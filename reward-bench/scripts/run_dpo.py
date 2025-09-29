# Copyright 2023 AllenAI. All rights reserved.
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

import argparse
import copy
import json
import logging
import os
import pdb
import sys
import time
import pprint

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from fastchat.conversation import get_conv_template
from tqdm import tqdm
from trl.trainer.utils import DPODataCollatorWithPadding

from rewardbench import DPO_MODEL_CONFIG, DPOInference, load_eval_dataset, save_to_hub
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section, load_preference_dataset

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login
#     for try_cnt in range(50):
    try:
        _login(token=HF_TOKEN, add_to_git_credential=False)
    except Exception as e:
        # print(f"Try number {try_cnt}: Error logging in to HF Hub: {e}"
        #       f"sleeping for 30 seconds and trying again..")
        # time.sleep(30)
        print("logging failed but eval can still happen")


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", type=str, default="", help="comma separated subsets")
    parser.add_argument("--dataset", type=str, default="", help="external dataset to load")
    parser.add_argument("--split", type=str, default=None, help="The split to evaluate on.")
    parser.add_argument("--load_json", action="store_true", default=False, help="Load dataset as json.")
    parser.add_argument("--model", type=str, required=True, help="path to model")
    parser.add_argument("--ref_model", type=str, default=None, help="path to model")
    parser.add_argument(
        "--ref_free_type", type=str, default="avg", help="type of reference free normalization (norm, avg, or sum)"
    )
    parser.add_argument(
        "--add_prefix_type", type=str, default=None, help="if not None, the type of prefix, "
                                                           "added to policy model only and not reference model"
    )
    parser.add_argument(
        "--prefix", type=str, default=None, help="the prefix"
    )
    parser.add_argument(
        "--prefix_df_path", type=str, default=None, help="the prefix"
    )
    parser.add_argument("--tokenizer", type=str, default=None, help="path to non-matching tokenizer")
    parser.add_argument("--chat_template", type=str, default="tulu", help="path to chat template")
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument("--batch_size", type=int, default=6, help="batch size for inference")
    parser.add_argument(
        "--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument("--debug", type=bool, default=False, help="use only 10 examples, and print not save")
    parser.add_argument(
        "--disable_beaker_save", action="store_true", help="disable saving the main results in a file for AI2 Beaker"
    )
    parser.add_argument("--output_dir", type=str, default="", help="path to output")
    parser.add_argument("--force_overwrite", action="store_true", help="if result file exist, overwrite or not")
    parser.add_argument("--full_results", action="store_true", help="output full output dataset scores")
    parser.add_argument("--eval_names", type=str, default=None, help="comma separated list of names to evaluate only")
    parser.add_argument("--max_len", type=int, default=2500, help="max sequence length")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    accelerator = Accelerator()

    ###############
    # Setup logging
    ###############
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")
    if args.trust_remote_code:
        logger.info("Loading model with Trust Remote Code")

    if args.model in DPO_MODEL_CONFIG:
        config = DPO_MODEL_CONFIG[args.model]
    else:
        config = DPO_MODEL_CONFIG["default"]
    logger.info(f"Using dpo model config: {config}")

    model_builder = config["model_builder"]
    tokenizer_builder = config["tokenizer_builder"]

    if isinstance(args.prefix, str) and len(args.prefix) == 0:
        args.add_prefix_type = None
        args.prefix = None
    if isinstance(args.add_prefix_type, str) and args.add_prefix_type.startswith("shuffle"):
        args.add_prefix_type = args.add_prefix_type.replace("shuffle_", "")

    # load chat template
    chat_template = args.chat_template
    conv = get_conv_template(chat_template)

    # define reference free
    if args.ref_model is None:
        ref_free = True
        logger.info("Running reference free DPO - no reference model provided")
    else:
        ref_free = False
        logger.info(f"Running DPO with reference model {args.ref_model}")

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = tokenizer_builder(tokenizer_path, trust_remote_code=args.trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    # if no BOS token, set as pad token, e.g. QWEN models
    if tokenizer.bos_token is None:
        tokenizer.bos_token_id = tokenizer.eos_token_id
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if not args.dataset:
        dataset, subsets = load_eval_dataset(
            core_set=not args.pref_sets,
            conv=conv,
            tokenizer=tokenizer,
            logger=logger,
            keep_columns=["text_chosen", "text_rejected", "id", "prompt", "name"],
        )
        if subsets:
            args.subsets = args.subsets.split(",")
            subset_mask = [i for i, s in enumerate(subsets) if s in args.subsets]
            dataset = dataset.select(subset_mask)
            subsets = [s for s in subsets if s in args.subsets]
        dataset = dataset.remove_columns("id")
    else:
        if args.eval_names is not None:
            args.eval_names = args.eval_names.split(",")
        dataset = load_preference_dataset(
            args.dataset, split=args.split, json=args.load_json, tokenizer=tokenizer, conv=conv,
            eval_names=args.eval_names, keep_columns=["prompt", "text_chosen", "text_rejected", "name"]
        )
        dataset_name = args.dataset.split("/")[-1].replace(".json","")
        subsets = [dataset_name for _ in range(len(dataset))]

    # debug: use only 10 examples
    if args.debug:
        dataset = dataset.select(range(10))
        subsets = subsets[:10]
    ############################
    # Check if result is there
    ############################
    if not args.subsets:
        print("========== example data point (prompt) ==========")
        print(dataset[0]["prompt"])
    postfix = "reasoning" if "math-prm" in args.subsets else "safety" if args.subsets else subsets[0]
    ref = f"ref_free_{args.ref_free_type}" if args.ref_model is None else "ref_sft"
    out_path = f"{args.output_dir}/all_results_{ref}_{postfix}.json"
    if os.path.exists(out_path) and not (args.force_overwrite or args.debug):
        print("result file exists, skipping ..")
        exit(1)

    ############################
    # Load reward model pipeline
    ############################
    BATCH_SIZE = args.batch_size

    model_kwargs = {
        "load_in_8bit": True,
        "device_map": "auto",
        "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        "use_auth_token": HF_TOKEN,
    }
    model = model_builder(
        args.model,
        trust_remote_code=args.trust_remote_code,
        **model_kwargs,
    )

    if ref_free:
        ref_model = None
    else:
        model_kwargs_ref = {
            "load_in_8bit": True,
            "device_map": "auto",
            "torch_dtype": torch.float16 if torch.cuda.is_available() else None,
        }
        ref_model = model_builder(
            args.ref_model,
            trust_remote_code=args.trust_remote_code,
            **model_kwargs_ref,
        )

    # use internal inference functions in DPO trainer
    dpo = DPOInference(
        model,
        ref_model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        ref_free_norm=args.ref_free_type,
        # norm is norm, avg is average, sum is sum
        add_prefix_type=args.add_prefix_type,
        prefix=args.prefix,
        prefix_df_path=args.prefix_df_path,
        max_len=args.max_len,
    )
    # tokenize dataset
    column_names = list(dataset.features)

    tokenized_dataset = dataset.map(dpo.tokenize_row, remove_columns=column_names)
    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=DPODataCollatorWithPadding(
            pad_token_id=tokenizer.pad_token_id,
            label_pad_token_id=dpo.label_pad_token_id,
            is_encoder_decoder=dpo.is_encoder_decoder,
        ),
        # collate_fn = lambda x: x, # fix weird batching error
        shuffle=False,
        drop_last=False,
    )
    results = []
    scores_chosen = []
    scores_rejected = []
    pol_chosen_score, pol_rejected_score, ref_chosen_score, ref_rejected_score = [], [], [], []

    def get_sub_batch(batch, bs, start_idx):
        half_batch = {}
        for k, v in batch.items():
            half_batch[k] = v[start_idx:start_idx+bs]
        return half_batch

    for step, batch in enumerate(tqdm(dataloader, desc="RM batch steps")):
        logger.info(f"RM inference step {step}/{len(dataloader)}")
        rewards_chosen, rewards_rejected = torch.tensor([]), torch.tensor([])
        policy_chosen_logps, policy_rejected_logps = torch.tensor([]), torch.tensor([])
        ref_chosen_logps, ref_rejected_logps = torch.tensor([]), torch.tensor([])
        tmp_batch = copy.deepcopy(batch)
        bs = batch["chosen_input_ids"].shape[0]
        while len(rewards_chosen) < batch["chosen_input_ids"].shape[0]:
            try:
                if args.full_results:
                    rewards_chosen_tmp, rewards_rejected_tmp, policy_chosen_logps_tmp, policy_rejected_logps_tmp, ref_chosen_logps_tmp, ref_rejected_logps_tmp = dpo.inference_step(tmp_batch, ref_free=ref_free, return_all=True)
                    policy_chosen_logps = torch.concat([policy_chosen_logps, policy_chosen_logps_tmp])
                    policy_rejected_logps = torch.concat([policy_rejected_logps, policy_rejected_logps_tmp])
                    ref_chosen_logps = torch.concat([ref_chosen_logps, ref_chosen_logps_tmp])
                    ref_rejected_logps = torch.concat([ref_rejected_logps, ref_rejected_logps_tmp])
                else:
                    rewards_chosen_tmp, rewards_rejected_tmp = dpo.inference_step(tmp_batch, ref_free=ref_free)
                rewards_chosen = torch.concat([rewards_chosen, rewards_chosen_tmp])
                rewards_rejected = torch.concat([rewards_rejected, rewards_rejected_tmp])
            except RuntimeError as e:
                print(f"Runtime error: {e}")
                if bs == 1:
                    print("OOM with batch size == 1, need to reduce context len")
                    exit(1)
                print(f"OOM error! reducing batch size from {bs} to {bs//2}, "
                      f"max response length is {tmp_batch['chosen_input_ids'].shape[1]},")
                if args.prefix:
                      print(f"max prompt+response length is {tmp_batch['chosen_input_ids'].shape[1]+tmp_batch['prefixed_prompt_input_ids'].shape[1]}")
                bs = bs // 2
                torch.cuda.empty_cache()
            tmp_batch = get_sub_batch(batch, bs, len(rewards_chosen))

        # for each item in batch, record 1 if chosen > rejected
        # extra score from dict within batched results (e.g. logits)
        # [{'label': 'LABEL_1', 'score': 0.6826171875},... ]
        if isinstance(rewards_chosen[0], dict):
            scores_chosen_batch = [result["score"] for result in rewards_chosen]
            scores_rejected_batch = [result["score"] for result in rewards_rejected]
        # for classes that directly output scores (custom code)
        else:
            scores_chosen_batch = rewards_chosen.cpu().numpy().tolist()
            scores_rejected_batch = rewards_rejected.cpu().numpy().tolist()

        [
            results.append(1) if chosen > rejected else results.append(0)
            for chosen, rejected in zip(scores_chosen_batch, scores_rejected_batch)
        ]

        scores_chosen += scores_chosen_batch
        scores_rejected += scores_rejected_batch

        if args.full_results:
            pol_chosen_score += policy_chosen_logps.cpu().numpy().tolist()
            pol_rejected_score += policy_rejected_logps.cpu().numpy().tolist()
            ref_chosen_score += ref_chosen_logps.cpu().numpy().tolist()
            ref_rejected_score += ref_rejected_logps.cpu().numpy().tolist()


    ############################
    # Print & process results
    ############################
    # add column for results for easy printing
    out_dataset = dataset.add_column("results", results)

    # add subsets back (removed so it's not handled by cuda)
    out_dataset = out_dataset.add_column("subset", subsets)
    # add scores_chosen and scores_rejected to the dataset
    out_dataset = out_dataset.add_column("scores_chosen", scores_chosen)
    out_dataset = out_dataset.add_column("scores_rejected", scores_rejected)
    if args.full_results:
        for var in ["pol_chosen_score", "pol_rejected_score", "ref_chosen_score", "ref_rejected_score"]:
            out_dataset = out_dataset.add_column(var, locals()[var])

    results_grouped = {}
    results_grouped["dataset_size"] = len(dataset)
    results_grouped["model"] = args.model
    results_grouped["ref_model"] = args.ref_model
    results_grouped["model_type"] = "DPO"  # TODO add options for references free, DPO-ref-free, or DPO-normalized
    if ref_free:
        results_grouped["model_type"] = "DPO Ref. Free"
        save_modifier = "_ref_free"
    else:
        save_modifier = ""
    results_grouped["chat_template"] = args.chat_template if not hasattr(tokenizer, "chat_template") else "tokenizer"
    # print per subset and log into results_grouped file
    present_subsets = np.unique(subsets)
    for subset in present_subsets:
        subset_dataset = out_dataset.filter(lambda example: example["subset"] == subset)
        num_correct = sum(subset_dataset["results"])
        num_total = len(subset_dataset["results"])
        print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
        results_grouped[subset] = num_correct / num_total

    # log leaderboard aggregated results
    results_leaderboard = {}
    if not args.pref_sets:
        results_leaderboard = calculate_scores_per_section(EXAMPLE_COUNTS, SUBSET_MAPPING, results_grouped)
        print(results_leaderboard)


    ############################
    # Upload results to hub
    ############################
    sub_path = "eval-set/" if not args.pref_sets else "pref-sets/"

    # upload chosen-rejected with scores
    # create new json with scores and upload
    scores_df = out_dataset.to_pandas()
    all_results = {**results_leaderboard, **results_grouped}
    if not args.debug:
        with open(out_path, "w") as f:
            json.dump(all_results, f)
    else:
        pprint.pprint(all_results)

    if args.full_results:
        del scores_df["prompt"], scores_df["text_chosen"], scores_df["text_rejected"]
        if not args.debug:
            scores_df.to_json(out_path.replace(".json", "_scores_dict.json"), orient="records", lines=True)
        else:
            pprint.pprint(scores_df.to_dict(orient="records"))

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
