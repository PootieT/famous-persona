# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import os
import pdb
import random
from typing import List, Literal, Optional
import numpy as np
import pandas as pd
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk, Dataset
from datasets.builder import DatasetGenerationError

from util import format_persona_inference_prompt
from .configs import DataArguments
from util import format_persona_inference_prompt


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\nƒpd{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"


def maybe_insert_system_message(messages, tokenizer):
    if messages[0]["role"] == "system":
        return

    # chat template can be one of two attributes, we check in order
    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.default_chat_template

    # confirm the jinja template refers to a system message before inserting
    if "system" in chat_template or "<|im_start|>" in chat_template:
        messages.insert(0, {"role": "system", "content": ""})


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "dpo"],
    auto_insert_empty_system_msg: bool = True,
):
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True if task == "generation" else False
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(chosen_messages, tokenizer)
                maybe_insert_system_message(rejected_messages, tokenizer)

            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "dpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            # For DPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            prompt_messages = example["chosen"][:-1]
            # Prepend a system message if the first message is not a system message
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(prompt_messages, tokenizer)

            # Now we extract the final turn to define chosen/rejected responses
            chosen_messages = example["chosen"][-1:]
            rejected_messages = example["rejected"][-1:]
            example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            try:
                example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
                example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            except:
                # some tokenizer (mistral) does not like just tokenizing the response turn (has to start with user)
                # so can apply template to whole convo, then replace up to last turn one
                example["text_chosen"] = tokenizer.apply_chat_template(example["chosen"], tokenize=False).replace(example["text_prompt"], "")
                example["text_rejected"] = tokenizer.apply_chat_template(example["rejected"], tokenize=False).replace(example["text_prompt"], "")
        else:
            raise ValueError(
                f"Could not format example as dialogue for `dpo` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'rm', 'dpo']}"
        )
    return example


def get_datasets(
    data_config: DataArguments | dict,
    splits: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle: bool = True,
    combine_eval_splits: bool = True,
    max_eval_data: Optional[int] = None,
    inference_persona: Optional[str] = None
) -> DatasetDict:
    """
    Loads one or more datasets with varying training set proportions.

    Args:
        data_config (`DataArguments` or `dict`):
            Dataset configuration and split proportions.
        splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        configs (Optional[List[str]], *optional*, defaults to `None`):
            List of dataset config names. If given must be the same length as 'data_config' keys.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
        combine_eval_splits (`bool`, *optional*, defaults to `True`):
            Whether to combine test splits into one test dataset or leave as separate.
        max_eval_data (Optional[int], defaults to `None`):
            Whether to remove excess eval data
        inference_persona (Optional[str], defaults to `None`):
            Whether to add persona data
    Returns
        [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
    """
    if type(data_config) is DataArguments:
        # Structure of the config to read the datasets and their mix
        # datasets_mixer:
        #     - 'dataset1': 0.5
        #     - 'dataset2': 0.3
        #     - 'dataset3': 0.2
        dataset_mixer = data_config.dataset_mixer
    elif isinstance(data_config, dict):
        # Structure of the input is:
        #     dataset_mixer = {
        #             "dataset1": 0.5,
        #             "dataset1": 0.3,
        #             "dataset1": 0.2,
        #         }
        dataset_mixer = data_config
    else:
        raise ValueError(f"Data config {data_config} not recognized.")

    raw_datasets = mix_datasets(
        dataset_mixer, splits=splits, configs=configs, columns_to_keep=columns_to_keep, shuffle=shuffle,
        combine_eval_splits=combine_eval_splits, max_eval_data=max_eval_data
    )

    # optionally adding persona inference in the training split
    if inference_persona is not None:
        raw_datasets["train"] = raw_datasets["train"].add_column("data_type", ["preference"]*len(raw_datasets["train"]))
        persona_inference_dataset = construct_persona_inference_dataset(data_config, raw_datasets["train"])
        raw_datasets["train"] = concatenate_datasets([raw_datasets["train"], persona_inference_dataset])
        if shuffle:
            raw_datasets["train"] = raw_datasets["train"].shuffle(seed=42)

    return raw_datasets


def mix_datasets(
    dataset_mixer: dict,
    splits: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle=True,
    combine_eval_splits: bool = True,
    max_eval_data: Optional[int] = None
) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        configs (Optional[List[str]], *optional*, defaults to `None`):
            List of dataset config names. If given must be the same length as 'dataset_mixer' keys.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
        combine_eval_splits (`bool`, *optional*, defaults to `True`):
            Whether to combine test splits into one test dataset or leave as separate.
        max_eval_data (Optional[int], defaults to `None`):
            Whether to remove excess eval data
    """
    splits = ["train", "test"] if splits is None else splits
    configs = [None] * len(dataset_mixer) if not configs else configs
    columns_to_keep = [] if columns_to_keep is None else columns_to_keep

    if configs is not None and len(configs) != len(dataset_mixer):
        raise ValueError("The number of given dataset config names must be the same as the given number of datasets.")

    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    val_split_names = ["test_" + n for n in simplify_split_names([s for s in splits if "test" in s])]
    fracs = []
    for (ds, frac), ds_config in zip(dataset_mixer.items(), configs):
        fracs.append(frac)
        for split in splits:
            try:
                # Try first if dataset on a Hub repo
                dataset = load_dataset(ds, ds_config, split=split)
            # except DatasetGenerationError:  # somehow this doesn't catch the exception?
            except Exception:
                # If not, check local dataset
                # dataset = load_from_disk(os.path.join(ds, split))
                dataset = load_dataset('json', data_files={"some_split": os.path.join(ds, split)}, split="some_split")
                
            # Remove redundant columns to avoid schema conflicts on load
            dataset = dataset.remove_columns([col for col in dataset.column_names if col not in columns_to_keep])
            if "train" in split:
                raw_train_datasets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            # non-deterministic shuffle, fix seed with trainer arg
            dataset = dataset.shuffle()
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if combine_eval_splits:
            if shuffle:
                raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(seed=42)
                if max_eval_data is not None and max_eval_data < len(raw_datasets["test"]):
                    raw_datasets["test"] = raw_datasets["test"].select(range(max_eval_data))
            else:
                raw_datasets["test"] = concatenate_datasets(raw_val_datasets)
        else:
            for i, raw_val_dataset in enumerate(raw_val_datasets):
                if shuffle:
                    raw_datasets[val_split_names[i]] = raw_val_datasets[i].shuffle(seed=42)
                    if max_eval_data is not None and max_eval_data < len(raw_val_datasets[i]):
                        raw_datasets[val_split_names[i]] = raw_datasets[val_split_names[i]].select(range(max_eval_data))
                else:
                    raw_datasets[val_split_names[i]] = raw_val_datasets[i]

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with splits {splits}. Check the dataset has been correctly formatted."
        )

    return raw_datasets


def simplify_split_names(splits: List[str]):
    # remove longest common strings in split inputs. Assuming differentiating part is contiguous in middle
    # greedily advance from left and right to obtain middle chunk
    if len(splits) == 1:
        return splits
    start = 0
    while len(set([split[start] for split in splits])) == 1:
        start += 1

    end = -1
    while len(set([split[end] for split in splits])) == 1:
        end -= 1

    new_splits = [split[start:end+1] for split in splits]
    return new_splits


def remove_persona_prefix(prompt):
    prompt = prompt.replace("### Preferred Response:", "").strip()
    return prompt[prompt.rfind("\n"):].strip()


def construct_persona_inference_dataset(data_config, train_dataset: Dataset) -> Dataset:
    """
    Given path to persona.jsonl (or latest persona generation), we construct preference pair:
    where prompt is few-shot sampled from training data, chosen is persona, and rejected is
    a persona of the same kind randomly sampled from another person.
    Args:
        data_config:
        train_dataset:

    Returns:
        dataset object
    """
    persona_df = pd.read_json(data_config.persona_data_path, lines=True, orient="records")
    train_df = train_dataset.to_pandas()
    # parsing the prefix types into arguments
    prefix_type = data_config.inference_persona
    prefix_type = prefix_type.replace("_gpt4", "")
    if "seed" in prefix_type:
        prefix_type = prefix_type.rsplit("_",1)[0]
    if prefix_type[-1] == "s" and prefix_type[-2].isdigit():
        num_shots = int(prefix_type.split("_")[-1].replace("s", ""))
    else:
        num_shots = 0
    reveal_name = "gold" in prefix_type
    x_only = "xy" not in prefix_type and not reveal_name

    # if prompt already has prefix in it, then get rid of it
    if len(train_df.prompt[0]) > 500:
        train_df["prompt"] = train_df.prompt.apply(remove_persona_prefix)

    # augment data
    aug_df = []
    for name in set(train_dataset["name"]):
        other_personas = persona_df.loc[(persona_df.name != name) & (~persona_df[data_config.inference_persona].isna()), data_config.inference_persona].tolist()
        
        # if number of inference data is a fraction, we probabilistically include this person's data (PRISM)
        # otherwise, we add multiple datapoints for the person, shuffling preference datapoints
        if data_config.num_persona_inference_per_person < 1:
            data_per_person = 1 if random.random() < data_config.num_persona_inference_per_person else 0
        else:
            data_per_person = int(data_config.num_persona_inference_per_person)
        
        for i in range(data_per_person):
            prompt = format_persona_inference_prompt(
                train_df, persona_df, True, name, num_shots, False,
                reveal_name, x_only, seed=i)
            self_persona = persona_df[persona_df.name == name][data_config.inference_persona].tolist()[0]
            other_persona = np.random.choice(other_personas)
            chosen = [*prompt["messages"], {"role": "assistant", "content": self_persona}]
            rejected = [*prompt["messages"], {"role": "assistant", "content": other_persona}]
            aug_df.append({
                "name": name,
                "prompt": prompt["messages"][1]["content"],
                "chosen": chosen,
                "rejected": rejected,
                "data_type": "persona"
            })

    dataset = Dataset.from_pandas(pd.DataFrame(aug_df))
    return dataset

