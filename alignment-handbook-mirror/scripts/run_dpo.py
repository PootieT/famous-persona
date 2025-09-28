#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
import sys
from typing import List, Optional, Tuple

from tqdm import tqdm
import evaluate
import numpy as np
import pandas as pd
import torch
import transformers
import wandb
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, set_seed, TrainingArguments, TrainerCallback, TrainerState, \
    TrainerControl

from alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    decontaminate_humaneval,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
    OurDPOTrainer
)
from peft import PeftConfig, PeftModel

from alignment.data import construct_persona_inference_dataset
from alignment.vpl_trainer import VPLTrainer

logger = logging.getLogger(__name__)


class FreezeEmbeddingCallback(TrainerCallback):
    def __init__(self, model, tokenizer, num_added_tokens: int):
        # makes a copy of the original embedding
        self.tokenizer = tokenizer
        self.orig_vocab_size = len(tokenizer.vocab) - num_added_tokens
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model)
        self.orig_embeds_params = model.get_input_embeddings().weight.data.clone().bfloat16()
        self.orig_embeds_params.requires_grad = False

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Let's make sure we don't update any embedding weights besides the newly added token
        # code from textual inversion
        index_no_updates = torch.ones((len(self.tokenizer.vocab),), dtype=torch.bool)
        index_no_updates[self.orig_vocab_size:] = False
        with torch.no_grad():
            kwargs["model"].get_input_embeddings().weight[index_no_updates] = self.orig_embeds_params.cuda()


class PersonaInferenceCallback(TrainerCallback):
    def __init__(self, trainer: OurDPOTrainer, data_args: DataArguments, train_dataset: Dataset) -> None:
        super().__init__()
        self._trainer = trainer
        self.raw_train_dataset = train_dataset
        self.data_args = data_args
        self.persona_df = pd.read_json(data_args.persona_data_path, lines=True, orient="records").set_index("name")
        self.persona_df_original = self.persona_df.copy()

    @staticmethod
    def trim_generations(generations, prompts):
        phrase = "<|assistant|>\n"
        return [g[max(g.rfind(phrase)+len(phrase), len(p)):].replace("|>", "").strip() for g, p in zip(generations, prompts)]

    def infer_persona(self, state, dataloader: DataLoader, bs:int, max_num_inference: Optional[int]=20) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        filter out datapoints to just generate personas, 1 per person, then optionally evaluate it
        by calculating rouge score to ground truth persona
        """
        print(f"Inferring Persona at the end of epoch {state.epoch}")
        # Sample persona subset, 1 datapoint per person
        ds = dataloader.dataset.filter(lambda x: x["data_type"]=="persona")
        sample_indices = []
        for name in set(ds["name"]):
            sample_indices.append(np.random.choice(np.where(np.array(ds["name"])==name)[0]))
        if max_num_inference is not None:
            sample_indices = random.sample(sample_indices, k=min(max_num_inference, len(sample_indices)))
        ds = ds.select(sample_indices)

        # Use dataloader.dataset.select to get the random batch without iterating over the DataLoader
        policy_outputs, ref_outputs = [], []
        num_batches = (len(ds) // bs) + 1 if len(ds) % bs != 0 else len(ds) // bs
        for b in tqdm(range(num_batches), desc="Generating personas"):
            batch = ds[b*bs:(b+1)*bs]
            batch_tokens = self._trainer.tokenizer(batch["prompt"], padding=True, return_tensors="pt", return_token_type_ids=False).to(self._trainer.model.device)
            policy_output_decoded = self._trainer.get_batch_samples(self._trainer.model, batch_tokens, policy_only=True)
            policy_outputs.extend(self.trim_generations(policy_output_decoded, batch["prompt"]))
            # just use the persona used to train as the reference ones
            ref_outputs.extend(self.persona_df_original.loc[batch["name"], self.data_args.inference_persona].tolist())

        self._trainer.log(
            {
                "game_log": wandb.Table(
                    columns=["Epoch", "Name", "Prompt", "Policy", "Ref Model"],
                    rows=[
                        [state.epoch, name, prompt, pol, ref]
                        for name, prompt, pol, ref in zip(
                            ds["name"], ds["prompt"], policy_outputs, ref_outputs
                        )
                    ],
                ),
                "epoch": state.epoch
            }
        )
        self._trainer.state.log_history.pop()
        return ds["name"], ds["prompt"], policy_outputs, ref_outputs

    def evaluate_and_log_persona_quality(self, state, names: List[str], policy_outputs: List[str], ref_outputs: List[str]):
        print("Evaluating and logging persona quality")
        rouge = evaluate.load('rouge')
        ref_cols = ["persona_xy_4s_gpt4", "persona_xy_4s_gpt4_seed1", "persona_gold_gpt4", "persona_gold_gpt4_seed1"]
        # for each person, 4 references
        references = [[r for r in refs if r is not None] for refs in self.persona_df_original.loc[names, ref_cols].to_dict("split")["data"]]
        policy_metrics = rouge.compute(predictions=policy_outputs, references=references)
        ref_metrics = rouge.compute(predictions=ref_outputs, references=references)
        policy_metrics = {f"persona_policy_{k}": v for k, v in policy_metrics.items()}
        ref_metrics = {f"persona_ref_{k}": v for k, v in ref_metrics.items()}
        policy_metrics.update(ref_metrics)
        self._trainer.log({"epoch": state.epoch, **policy_metrics})

    def on_epoch_end(self, args, state, control, **kwargs):
        if (self.data_args.evaluate_persona_rate != 0 and state.epoch % self.data_args.evaluate_persona_rate == 0) or \
           (self.data_args.refresh_persona_rate != 0 and state.epoch % self.data_args.refresh_persona_rate == 0):
            # Infer persona by generation
            names, prompts, policy_outputs, ref_outputs = self.infer_persona(state, self._trainer.get_train_dataloader(), 1) #args.eval_batch_size)

            # Optionally compute some metric on how good persona is
            if (self.data_args.evaluate_persona_rate != 0 and state.epoch % self.data_args.evaluate_persona_rate == 0):
                self.evaluate_and_log_persona_quality(state, names, policy_outputs, ref_outputs)

            # Optionally update persona in dataset with the generation
            if (self.data_args.refresh_persona_rate != 0  and state.epoch % self.data_args.refresh_persona_rate == 0):
                # update in the persona in latest persona_df
                persona_df_previous = self.persona_df.copy()
                self.persona_df.loc[names, self.data_args.inference_persona] = policy_outputs

                # update persona that is built into prompts in preference data
                def replace_persona(example):
                    old_persona = persona_df_previous.loc[example["name"], self.data_args.inference_persona]
                    new_persona = self.persona_df.loc[example["name"], self.data_args.inference_persona]
                    if example["data_type"] == "preference":
                        # in preference data, persona appear in prompt
                        if example["name"] in names:
                            example["prompt"] = example["prompt"].replace(old_persona, new_persona)
                    elif example["data_type"] == "persona":
                        # in persona inference data, persona appear in either chosen / rejeceted
                        if example["name"] in names:
                            example["chosen"] = example["chosen"].replace(old_persona, new_persona)
                        else:
                            example["rejected"] = example["rejected"].replace(old_persona, new_persona)
                    return example
                self.raw_train_dataset = self.raw_train_dataset.map(replace_persona, num_proc=self._trainer.dataset_num_proc).shuffle(seed=42)
                # TODO could also optionally mix in previous rounds of personas
                train_dataset_tokenized = self.raw_train_dataset.map(self._trainer.tokenize_row, num_proc=self._trainer.dataset_num_proc)
                self._trainer.train_dataset = train_dataset_tokenized


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    raw_datasets, test_splits, tokenizer = load_datasets(data_args, model_args, training_args,
                                                         data_args.dataset_splits,
                                                         data_args.combine_eval_splits,
                                                         data_args.max_eval_data,
                                                         log=True,
                                                         inference_persona=data_args.inference_persona)

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path
    if is_adapter_model(model, model_args.model_revision) is True:
        logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
        model_kwargs = dict(
            revision=model_args.base_model_revision,
            trust_remote_code=model_args.trust_remote_code,
            use_flash_attention_2=model_args.use_flash_attention_2,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )

        model = PeftModel.from_pretrained(
            base_model,
            model_args.model_name_or_path,
            revision=model_args.model_revision,
            is_trainable=not training_args.merge_pretrained_peft
        )

        model_kwargs = None

    ref_model = model
    ref_model_kwargs = model_kwargs

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None

    #########################
    # Instantiate DPO trainer
    #########################
    trainer_cls = VPLTrainer if training_args.use_vpl else OurDPOTrainer
    trainer = trainer_cls(
        model=model,
        ref_model=ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        model_args=model_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"] if data_args.combine_eval_splits
                     else {k: raw_datasets[k] for k in test_splits},
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        peft_config=get_peft_config(model_args),
        loss_type=training_args.loss_type,
        generate_during_eval=training_args.generate_during_eval,
        callbacks= None
    )

    # add callbacks
    callbacks = []
    if model_args.add_personal_token_embedding > 0:
        callbacks.append(FreezeEmbeddingCallback(model, tokenizer, model_args.add_personal_token_embedding))
    if data_args.refresh_persona_rate or data_args.evaluate_persona_rate:
        callbacks.append(PersonaInferenceCallback(trainer, data_args, raw_datasets["train"]))
    for callback in callbacks:
        trainer.add_callback(callback)

    ###############
    # Training loop
    ###############
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(raw_datasets["train"])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        logger.info("*** Training complete ***")

        ##################################
        # Save model and create model card
        ##################################
        logger.info("*** Save model ***")
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

        # Save everything else on main process
        kwargs = {
            "finetuned_from": model_args.model_name_or_path,
            "dataset": list(data_args.dataset_mixer.keys()),
            "dataset_tags": list(data_args.dataset_mixer.keys()),
            "tags": ["alignment-handbook"],
        }
        if trainer.accelerator.is_main_process:
            trainer.create_model_card(**kwargs)
            # Restore k,v cache for fast inference
            trainer.model.config.use_cache = True
            trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate (default reference model) ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Evaluate (default reference model) on full test set ***")
        logger.info("loading full eval_dataset_splits")
        import wandb
        wandb.init(mode="disabled")
        trainer.ref_model = None
        raw_datasets, test_splits, _ = load_datasets(data_args, model_args, training_args,
                                                     data_args.eval_dataset_splits,
                                                     split_eval_by_name=data_args.split_predict_by_name,
                                                     combine_eval_splits=False, max_eval_data=None, log=False,
                                                     inference_persona=None)
        trainer.eval_dataset = {n: raw_datasets[n].map(trainer.tokenize_row, num_proc=trainer.dataset_num_proc) for n in test_splits}
        # if vpl exclude people during training, then we mark these eval data with OOD at front
        if model_args.vpl_exclude_people is not None:
            if isinstance(model_args.vpl_exclude_people, str):
                trainer.vpl_exclude_people_list = model_args.vpl_exclude_people.split(",")
            trainer.eval_dataset = {
                f"OOD_{n}" if d["name"][0].replace(' ', '').lower() in trainer.vpl_exclude_people_list and not n.startswith("OOD_")
                else n: d for n, d in trainer.eval_dataset.items()
            }
        metrics = trainer.evaluate()
        trainer.log_metrics("eval_ref_free", metrics)
        trainer.save_metrics("eval_ref_free", metrics)
        # I can just change the ref_model and compute DPO reward on eval sets (baseline model=Mistral-7B)
        if training_args.alternative_ref_models is not None:
            for ref_model_name in training_args.alternative_ref_models:
                logger.info(f"*** Evaluate (alternative reference model = {ref_model_name}) ***")
                eval_prefix = "ref_free" if ref_model_name == "ref_free" else ref_model_name.split("/")[1]
                ref_model = None if ref_model_name == "ref_free" else AutoModelForCausalLM.from_pretrained(ref_model_name).cuda()
                trainer.ref_model = ref_model
                with torch.no_grad():
                    metrics = trainer.evaluate()
                trainer.log_metrics(f"eval_{eval_prefix}", metrics)
                trainer.save_metrics(f"eval_{eval_prefix}", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


def load_datasets(
    data_args,
    model_args,
    training_args,
    dataset_splits: List[str],
    combine_eval_splits: bool,
    max_eval_data: Optional[int],
    log: bool = True,
    split_eval_by_name: bool=False,
    inference_persona: Optional[str] = None
):
    """

    Args:
        data_args:
        model_args:
        training_args:
        dataset_splits:
        combine_eval_splits: if input multiple evaluation files, whether to combine them into single
            eval data (this is nice for eval during training)
        max_eval_data: maximum number of datapoints to evaluate (randomly subsample)
        log:
        split_eval_by_name: if given generic combined train and eval files (2), split the combined
            evaluation dataset into dict of name mapped to eval dataset by itself (this is nice
            for reporting)
        inference_persona: whether to add persona inference data

    Returns:

    """
    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label", "name"],
        combine_eval_splits=combine_eval_splits,
        max_eval_data=max_eval_data,
        inference_persona=inference_persona,
    )
    if log:
        logger.info(
            f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
        )
    column_names = list(raw_datasets["train"].features)
    if "data_type" in column_names:
        column_names.remove("data_type")
    column_names.remove("name")
    test_splits = [split for split in raw_datasets.keys() if "train" not in split]

    if split_eval_by_name:
        test_splits = []
        # assert len(raw_datasets) == 2
        non_train_datasets = {k:ds for k, ds in raw_datasets.items() if "train" not in k}
        new_raw_datasets = DatasetDict()
        new_raw_datasets["train"] = raw_datasets["train"]
        train_names = set(raw_datasets["train"]["name"])
        eval_names = set(list(non_train_datasets.values())[0]["name"])
        for name in list(eval_names):
            name_str = name.replace(' ', '').lower()
            eval_split_name = f"eval_{name_str}" if name in train_names else f"OOD_eval_{name_str}"
            if len(raw_datasets) == 2:
                new_raw_datasets[eval_split_name] = list(non_train_datasets.values())[0].filter(lambda example: example["name"] == name)
                test_splits.append(eval_split_name)
            else:
                for test_file_name, non_train_dataset in non_train_datasets.items():
                    new_eval_split_name = eval_split_name + "_" + test_file_name.split("_")[1]
                    new_raw_datasets[new_eval_split_name] = non_train_dataset.filter(lambda example: example["name"] == name)
                    test_splits.append(new_eval_split_name)
        raw_datasets = new_raw_datasets
        if log:
            logger.info(f"Split combined eval data by name into {len(eval_names)} splits, "
                        f"average of {np.mean([len(new_raw_datasets[n]) for n in new_raw_datasets.keys() if n!='train'])}")

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)
    if model_args.add_personal_token_embedding > 0:
        special_tokens_dict = {'additional_special_tokens': [f"<special_person_tag_{i}>" for i in
                                                             range(model_args.add_personal_token_embedding)]}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "dpo",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )
    ##########################
    # Decontaminate benchmarks
    ##########################
    num_raw_train_samples = len(raw_datasets["train"])
    raw_datasets = raw_datasets.filter(
        decontaminate_humaneval,
        fn_kwargs={"text_column": "text_chosen"},
        batched=True,
        batch_size=10_000,
        num_proc=1,
        desc="Decontaminating HumanEval samples",
    )
    num_filtered_train_samples = num_raw_train_samples - len(raw_datasets["train"])
    if log:
        logger.info(
            f"Decontaminated {num_filtered_train_samples} ({num_filtered_train_samples / num_raw_train_samples * 100:.2f}%) samples from the training set."
        )
    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", *test_splits]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )
    # Log a few random samples from the training set:
    if log:
        for index in random.sample(range(len(raw_datasets["train"])), 3):
            logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
            logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
            logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")
    return raw_datasets, test_splits, tokenizer


if __name__ == "__main__":
    main()
