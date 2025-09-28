import os
import pdb
import json
import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Literal

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch

from context_cite import ContextCiter
from context_cite.context_partitioner import SimpleContextPartitioner

from qualitative_difference_persona_vs_fs import load_data_and_prefices

DUMP_DIR = str(Path(__file__).parents[1].joinpath('dump').joinpath("pp-50-final").joinpath(
    "one_model_for_all_cv").resolve().absolute())
OUT_DIR = str(Path(__file__).parent.joinpath('attributions').resolve().absolute())

MODEL_SHORT_TO_FULL = {
    "zephyr": "HuggingFaceH4/zephyr-7b-beta",
    "llama321b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama323b": "meta-llama/Llama-3.2-3B-Instruct",
    "ministral8b": "mistralai/Ministral-8B-Instruct-2410",
}

def cache_source_attributions(
    prefix: str = "persona_xy_4s",
    use_finetuned_models: bool = False,
    attribution_source_type: Literal["sentence", "word"] = "sentence",
    cv: Optional[int] = None,
    model_prefix: Optional[str] = None,
    model: str = "zephyr",
):
    """
    save source attribution vectors of models on our dataset

    Args:
        prefix: prefix type (of data and finetuned model, unless model_prefix is specified)
        use_finetuned_models: whether to use finetuned models
        attribution_source_type: one of "sentence", "word"
        cv: which cross-validation folds
        model_prefix: if prefix is different than that used during finetuning, this is to specify model's finetuning prefix
        model: which model to use, default is zephyr. options: {zephyr,llama321b,llama323b,ministral8b}
    """
    # some customization if base model is not zehpyr
    model_hf_path = MODEL_SHORT_TO_FULL[model]
    global DUMP_DIR
    DUMP_DIR = DUMP_DIR if model == "zephyr" else f"{DUMP_DIR}_{model}"
    if prefix == "persona_xy_4s" and model != "zephyr":
        print(f"{model} is not trained with persona_xy_4s, switching to self-generated prefix {prefix}_{model}.")
        prefix = f"{prefix}_{model}"

    # check if computed, if so, skip
    model_prefix = None if model_prefix == prefix else model_prefix
    if model_prefix is None:
        out_path = f"{OUT_DIR}/{model}/{prefix}{'_finetuned' if use_finetuned_models else ''}_{attribution_source_type}{'_cv' + str(cv) if cv is not None else ''}.json"
    else:
        out_path = f"{OUT_DIR}/{model}/{model_prefix}{'_finetuned' if use_finetuned_models else ''}_alt{prefix}_{attribution_source_type}{'_cv' + str(cv) if cv is not None else ''}.json"
    if os.path.exists(out_path):
        print(f"output file already exists, skipping {out_path}")
        exit(0)

    data_df, persona_df = load_data_and_prefices(prefix=prefix)

    # make output dir
    os.makedirs(f"{OUT_DIR}/{model}", exist_ok=True)

    output_df = []
    cvs = range(5) if cv is None else [cv]
    for curr_cv in cvs:
        print(f"Caching attributions for {prefix}, CV={curr_cv}, {use_finetuned_models=}, {model_prefix=}...")
        # find subset of the people
        cv_persona_df = persona_df[persona_df["cv"] == curr_cv]
        cv_data_df = data_df[data_df["name"].isin(cv_persona_df["name"])]

        # load model
        model_name = model_hf_path if not use_finetuned_models else \
            f"{DUMP_DIR}/overton_{prefix if model_prefix is None else model_prefix}_cv{curr_cv}"
        citer = ContextCiter.from_pretrained(model_name, context="", query="", prompt_template="{context}\n\n{query}",
                                             device="cuda", num_ablations=64, batch_size=2)

        # loop through each datapoints and attribute
        for i, row in tqdm(cv_data_df.iterrows(), total=len(cv_data_df)):
            if "xyw" in prefix:
                query = row["prompt"][row["prompt"].rfind("## Prompt"):].strip()
                context = row["prompt"][:row["prompt"].rfind("## Prompt")].strip()
            else:
                query = row["prompt"].strip().split("\n")[-1]
                context = row["prompt"].replace(query, "").strip()
            citer.query = query
            citer.context = context
            citer.partitioner = SimpleContextPartitioner(context, source_type=attribution_source_type)  # word
            _, prompt = citer._get_prompt_ids(return_prompt=True)
            res_row = {}
            for res_field in ["chosen", "rejected"]:
                response = row[res_field][-1]["content"]
                with torch.no_grad():
                    scores = try_attribute_score(citer, prompt, query, response, row)
                torch.cuda.empty_cache()
                res_row.update({
                    "name": row["name"],
                    "question_type": row["question_type"],
                    "query": query,
                    "sources": citer.partitioner.sources,
                    f"{res_field}_response": response,
                    f"{res_field}_scores": scores,
                })

                # clear cache for next example
                citer._cache = {}
            output_df.append(res_row)

    output_df = pd.DataFrame(output_df)
    output_df.to_json(out_path, orient="records", lines=True)
    return


def try_attribute_score(citer: ContextCiter, prompt: str, query: str, response:str, row):
    citer._cache["output"] = prompt + response
    if len(response.strip()) == 0:
        print(f"Encounter empty response for {row['name']}:\nprompt={query}")
        scores = np.zeros(len(citer.partitioner.sources))
    else:
        scores = None
        og_batch_size = citer.batch_size
        while scores is None:
            try:
                scores = citer.get_attributions(verbose=False)
            except Exception as e:
                if citer.batch_size != 1:
                    print(f"Exception: {e}\n. Current batch size is {citer.batch_size}, retrying halfing batch size")
                    citer.batch_size /= 2
                elif citer.batch_size == 1 and len(response.strip()) > 4:
                    print(f"Exception: {e}\n. Current batch size is 1, trying halfing response length")
                    response = response[:len(response)//2]
                else:
                    print(f"Exception: {e}, batch size is already 1. , default to zero attributions")
                    scores = np.zeros(len(citer.partitioner.sources))
        citer.batch_size = og_batch_size
    return scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="persona_xy_4s")
    parser.add_argument("--use_finetuned_models", action="store_true")
    parser.add_argument("--attribution_source_type", type=str, default="sentence")
    parser.add_argument("--cv", type=int, default=None)
    parser.add_argument("--model_prefix", type=str, default=None)
    parser.add_argument("--model", type=str, default="zephyr")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cache_source_attributions(**vars(args))

