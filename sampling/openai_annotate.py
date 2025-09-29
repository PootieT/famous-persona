import json
import os
from copy import deepcopy
from operator import itemgetter
from pathlib import Path
from typing import List, Dict, Set, Literal

import torch
import yaml
from toposort import toposort, toposort_flatten
from tqdm import tqdm
import numpy as np
import pandas as pd
import tiktoken
from transformers import AutoTokenizer, pipeline

from alpaca_farm.auto_annotations import PairwiseAutoAnnotator
from alpaca_eval.annotators.pairwise_evaluator import PairwiseAnnotator

from prompts import SAMPLE_WIKI_PREFERENCE_GIVEN_X_Y_2S_5B_CHAT
THIS_FILE_PATH = Path(__file__)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def calculate_rate(annotators, data):
    # Set-up the tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    total_token_num = 0
    for row in data:
        text_input = str(row['instruction']) + str(row['input']) + str(row['output_1']) + str(row['output_2'])
        tokens = tokenizer.encode(text_input)
        total_token_num += len(tokens)
    if 'gpt-3.5-turbo' in annotators or 'chatgpt' in annotators:
        price_per_1k = 0.0015
    elif 'gpt-4o' in annotators:
        price_per_1k = 0.005
    elif 'gpt-4-turbo' in annotators:
        price_per_1k = 0.01
    elif 'text-davinci-003' in annotators or 'gpt-4' in annotators:
        price_per_1k = 0.03
    elif 'annotator_pool_v0' in annotators:
        price_per_1k = 0.03 # Estimate for Alpacafarm annotator pool w/ 14 annotator combinations
    else:
        raise Exception('Please include the model name in the annotators config.')
    estimated_pricing = (total_token_num / 1000 ) * price_per_1k
    # estimated_pricing_with_demonstrations = estimated_pricing * 4 # Estimated price for the prompts appeneded
    estimated_pricing_with_demonstrations = estimated_pricing * 1.633  # Estimated price for the prompts appeneded (in personal preference data)
    print(f'The esimtated cost for this annotation is ${estimated_pricing_with_demonstrations:.2f} with {total_token_num} number of tokens (excluding demonstrations).')
    # consent = input(f'The esimtated cost for this annotation is ${estimated_pricing_with_demonstrations:.2f} with {total_token_num} number of tokens (excluding demonstrations). Would you like to go ahead? \nReply with y/n: ')
    # if consent == 'y':
    #    print('proceeding with the annotation!')
    # else:
    #    exit('not proceeding with the annotation because it is too expensive..! :(')


def format_to_alpaca_df(df: pd.DataFrame, simulate_personal_preference: bool) -> List[Dict[str, str]]:
    if isinstance(df.y[0], str):
        df.y = df.y.apply(lambda x: eval(x))
    assert isinstance(df.y[0], list)
    if simulate_personal_preference:
        # instruction: please simulate {NAME}'s preferrence over ..., input: the question x user is asking
        df = df.rename(columns={"prompt": "input", "name": "instruction"})
        df.instruction = df.instruction.apply(lambda name: f"Please simulate {name}'s preference over the answers for the questions below.")
    else:
        # instruction: the question x user is asking, input: empty
        df = df.rename(columns={"prompt": "instruction"})
        df["input"] = ""
    df["output_1"] = df.y.apply(lambda y: y[0])
    df["output_2"] = df.y.apply(lambda y: y[1])
    del df["y"]
    return df.to_dict("records")


def get_annotator_config():
    """deprecated for now, but in case we need personalized annotator per person, we may bring it up again"""
    config_path = THIS_FILE_PATH.parent.joinpath(f"annotator_configs/annotator_config_multi_name.yaml")
    if not os.path.isdir(config_path.parent):
        os.makedirs(config_path.parent)

    if not os.path.isfile(config_path):
        config = deepcopy(DEFAULT_CONFIG)
        with open(config_path, 'w+') as f:
            yaml.dump(config, f)
        prompt_path = config["gpt4_3"]["with_inputs"]
        prompt = SAMPLE_WIKI_PREFERENCE_GIVEN_X_Y_2S_5B_CHAT
        with open(prompt_path, "w") as f:
            f.write(prompt)

    return config_path


def format_final_dataset(df: pd.DataFrame, annotated_results: List[Dict]) -> pd.DataFrame:
    assert len(df) == len(annotated_results)
    filled_annotated_results = deepcopy(annotated_results)
    for annotation in filled_annotated_results:
        if annotation["preference"] not in [1, 2]:
            annotation["preference"] = np.random.choice([1,2])
    df["yw"] = df.apply(lambda r: r.y[0] if filled_annotated_results[r.name]["preference"] == 1
                        else r.y[1] if filled_annotated_results[r.name]["preference"] == 2 else None, 1)
    df["yl"] = df.apply(lambda r: r.y[1] if filled_annotated_results[r.name]["preference"] == 1
                        else r.y[0] if filled_annotated_results[r.name]["preference"] == 2 else None, 1)
    df["raw_completion"] = [d["raw_completion"] for d in annotated_results]
    assert sum(df["yw"].isna()) == 0
    return df


def annotate_generations(data_path: str, simulate_personal_preference: bool=True):
    assert len(OPENAI_API_KEY) > 1

    df = pd.read_json(data_path, lines=True, orient="records")
    all_output_pairs = format_to_alpaca_df(df, simulate_personal_preference)
    calculate_rate("gpt-4", all_output_pairs)

    # annotator_config = "gpt4_personal_preference_multi/configs.yaml" if simulate_personal_preference else "alpaca_eval_gpt4"
    annotator_config = "gpt4_personal_preference_multi_cot/configs.yaml" if simulate_personal_preference else "alpaca_eval_gpt4"
    annotator = PairwiseAutoAnnotator(annotators_config=annotator_config)
    annotated = annotator.annotate_pairs(all_output_pairs)
    prefs = np.array([d["preference"] for d in annotated])
    print(f"Annotation finished, total of {len(prefs)} pairs, "
          f"{sum(prefs==1)} prefer option 1, {sum(prefs==2)} prefer option 2, "
          f"and {len(prefs)-sum(prefs==2)-sum(prefs==1)} no preferences")

    # format output
    final_df = format_final_dataset(df, annotated)
    final_df.to_json(data_path.replace(".json", f"_{'cot_' if simulate_personal_preference else 'default_'}annotated.json"), lines=True, orient="records")


def calculate_candidate_comparisons(comparisons: Dict[int, Set[int]]) -> List[int]:
    counts = [0 for _ in range(len(comparisons))]
    for winner, loosers in comparisons.items():
        for looser in loosers:
            counts[looser] += 1
            counts[winner] += 1
    return counts


def select_next_pairs_for_pairwise_annotation(df: pd.DataFrame, y_field) -> pd.DataFrame:
    # topo_results are of the format:
    # [{1}, {2}, {3, 4}] ==> meaning 3 and 4 are tied for top priority
    topo_results = df.pairwise_comparisons.apply(lambda x: list(toposort(x)))
    # candidate_comparison_counts is list of integers as count of pairwise comparisons a candidate has gone through
    candidate_comparison_counts = df.pairwise_comparisons.apply(lambda x: calculate_candidate_comparisons(x))

    total_candidates = []
    for i in range(len(topo_results)):
        # greedily find highest priority-tied pairs after topological sort
        candidate_hierarchy = -1
        candidate_pool = topo_results[i][candidate_hierarchy]
        while len(candidate_pool) <= 1:
            candidate_hierarchy -= 1
            candidate_pool = topo_results[i][candidate_hierarchy]

        # then find the least compared to candidates, and compair them
        assert len(candidate_pool) >= 2
        candidate_pool_w_counts = zip(candidate_pool, [candidate_comparison_counts[i][c] for c in candidate_pool])
        candidate_pool_w_counts = sorted(candidate_pool_w_counts, key=itemgetter(1))

        # return top 2 candidates with the least comparisons so far
        candidates = [df[y_field][i][c[0]] for c in candidate_pool_w_counts[:2]]
        total_candidates.append(candidates)

    df["y"] = total_candidates
    return df


def update_dataset_ranks(df: pd.DataFrame, annotated: List[Dict], y_field: str) -> pd.DataFrame:
    assert len(df) == len(annotated)
    # update pairwise_comparisons with pairwise annotation result
    filled_annotated_results = deepcopy(annotated)
    if "raw_completions" not in df.columns:
        df["raw_completions"] = [{} for _ in range(len(df))]
    for i, row in df.iterrows():
        candidate_indices = [row[y_field].index(c) for c in row.y]
        if filled_annotated_results[i]["preference"] not in [1, 2]:
            filled_annotated_results[i]["preference"] = np.random.choice([1, 2])
        if filled_annotated_results[i]["preference"] == 1:
            row["pairwise_comparisons"][candidate_indices[0]].add(candidate_indices[1])
            row["raw_completions"][f"{candidate_indices[0]}_beats_{candidate_indices[1]}"] = \
                filled_annotated_results[i]["raw_completion"]
        else:
            row["pairwise_comparisons"][candidate_indices[1]].add(candidate_indices[0])
            row["raw_completions"][f"{candidate_indices[1]}_beats_{candidate_indices[0]}"] = \
                filled_annotated_results[i]["raw_completion"]
    return df


def format_final_dataset_ranks(df: pd.DataFrame, y_field: str, yl_strategy: str) -> pd.DataFrame:
    topo_result = df.pairwise_comparisons.apply(lambda x: list(toposort(x)))
    topo_result_flat = df.pairwise_comparisons.apply(lambda x: toposort_flatten(x))
    df["yw"] = df.apply(lambda r: r[y_field][next(iter(topo_result[r.name][-1]))], 1)
    if yl_strategy == "random":
        df["yl"] = df.apply(lambda r: r[y_field][np.random.choice(topo_result_flat[r.name][:-1])], 1)
    elif yl_strategy == "worst":
        df["yl"] = df.apply(lambda r: r[y_field][np.random.choice(topo_result[r.name][0])], 1)
    elif yl_strategy == "best":
        df["yl"] = df.apply(lambda r: r[y_field][np.random.choice(topo_result[r.name][-2])], 1)
    assert sum(df["yw"].isna()) == 0
    return df


def rank_generations(data_path: str, simulate_personal_preference: bool=True, top_only: bool=True,
                     y_field: str = "diverse_ys", yl_strategy: Literal["random", "worst", "best"]="random"):
    """
    Given more than 2 response candidates, rank them using pairwise comparison and topological sort.
    ranking strategy: round-robin, say we have 4 candidates, ABCD, we compare first AB, CD, then winner of those two
    the comparisons column keeps track of pairwise comparisons of each generation. It's a dict of dicts, tracking
    list of (winner, loser) generation indices. e.g. {1:{2},3:{0}} => index 1 solution beats 2, whereas index 3 beats 0.
    There is no 1st place yet. Dataformat following https://pypi.org/project/toposort/
    Args:
        data_path: path to .json panda dataframe. dataframe should consist of columns [name, prompt, {y_field}]
        simulate_personal_preference: whether to use personal gpt4 annotator or default annotator
        top_only: if we only care about finding the top candidate O(log(n)+1) vs. sorting out all rank orders O(n*(n-1)
        y_field: the field where the y candidates are from
        yl_strategy: one of {random, worst}. Random just randomly pick from non-top options. worst picks from the worst
            pool of candidates. worst is deterministic when top_only is False.

    Returns:
        saves the dataframe with pairwise comparison results. Added columns include [pairwise_comparisons,
        raw_completions, yw, yl]
    """
    assert len(OPENAI_API_KEY) > 1
    df = pd.read_json(data_path, lines=True, orient="records")
    assert "name" in df.columns and df.name.isna().sum() == 0, "maybe a common question data but not distributed?"
    n_candidates = len(df[y_field][0])
    rounds = np.log2(n_candidates) + 1 if top_only else n_candidates * (n_candidates-1)
    assert rounds == int(rounds), "Can only rank 2^n number of generations at the moment"
    rounds = int(rounds)

    if "pairwise_comparisons" not in df.columns:
        df["pairwise_comparisons"] = [{i: set([]) for i in range(n_candidates)} for _ in range(len(df))]

    for r in range(rounds):
        print(f"Starting annotatation round {r} out of {rounds} rounds")
        df = select_next_pairs_for_pairwise_annotation(df, y_field)
        all_output_pairs = format_to_alpaca_df(df, simulate_personal_preference)

        # annotator_config = "gpt4_personal_preference_multi/configs.yaml" if simulate_personal_preference else "alpaca_eval_gpt4"
        annotator_config = "gpt4_personal_preference_multi_cot/configs.yaml" if simulate_personal_preference else "alpaca_eval_gpt4"
        annotator = PairwiseAutoAnnotator(annotators_config=annotator_config)
        calculate_rate(list(annotator.annotators.values())[0].completions_kwargs["model_name"], all_output_pairs)
        annotated = annotator.annotate_pairs(all_output_pairs)
        prefs = np.array([d["preference"] for d in annotated])
        print(f"Annotation finished, total of {len(prefs)} pairs, "
              f"{sum(prefs==1)} prefer option 1, {sum(prefs==2)} prefer option 2, "
              f"and {len(prefs)-sum(prefs==2)-sum(prefs==1)} no preferences")
        df = update_dataset_ranks(df, annotated, y_field)

    # format output
    final_df = format_final_dataset_ranks(df, y_field, yl_strategy)
    out_data_path = data_path.replace(".json", f"_yl-{yl_strategy}_{'cot_' if simulate_personal_preference else 'default_'}annotated.json")
    final_df.to_json(out_data_path, lines=True, orient="records")


def rank_generations_automodel(
    data_path: str,
    model_name: str="sfairXC/FsfairX-LLaMA3-RM-v0.1",
    y_field: str = "diverse_ys",
    yl_strategy: Literal["random", "worst", "best"] = "random"
):
    df = pd.read_json(data_path, lines=True, orient="records")
    out_data_path = data_path.replace(".json", f"_automodel_annotated.json")

    rm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    rm_pipe = pipeline(
        "sentiment-analysis",
        model="sfairXC/FsfairX-LLaMA3-RM-v0.1",
        device="cuda",
        tokenizer=rm_tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16}
    )

    pipe_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 16
    }

    def generate_pref_graphs(rewards: np.array):
        # format the preference in like {1: {2}, 3: {0}}
        prefs = {i:{} for i in range(len(rewards))}
        sorted_indices = np.argsort(rewards)
        for i in range(1, len(rewards)):
            prefs[sorted_indices[i]].add(sorted_indices[i-1])
        return prefs

    for i, row in tqdm(df.iterrows(), total=len(df)):
        chats = [[{"role": "user", "content": row.prompt}, {"role": "assistant", "content": y}] for y in row[y_field]]
        test_texts = [rm_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False).replace(rm_tokenizer.bos_token,"") for chat in chats]
        pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
        rewards = np.array([output[0]["score"] for output in pipe_outputs])
        df.iloc[i, "pairwise_comparisons"] = generate_pref_graphs(rewards)

    final_df = format_final_dataset_ranks(df, y_field, yl_strategy)
    final_df.to_json(out_data_path, lines=True, orient="records")


if __name__ == "__main__":
    fs = [
        "../data/pp-50-final/all_50p_200d_common_50r_tem2.0_top0.8_cot_gold_filtered20m4k_distributed.json",
        "../data/pp-50-final/all_50p_200d_50r_tem2.0_top0.8_cot_filtered20m4k.json",
    ]
    for f in fs:
        rank_generations(
            f, simulate_personal_preference=True, yl_strategy="random"
        )



