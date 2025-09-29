
import os
import pdb
import pprint
import random

import argparse
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Literal

from sentence_transformers import SentenceTransformer  # I think this takes a while
import sklearn.cluster
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

from openai_model import chat_completions
from prompts import *
from transformers import AutoTokenizer, pipeline

from util import format_persona_inference_prompt, deepseek_chat_completions, query_questions, merge_persona_axis, \
    get_category, join_names, generate_completions_with_gold_categories, \
    extract_persona_name, RESPONSE_SAMPLING_KWARGS, AUTOMODEL_RESPONSE_SAMPLING_KWARGS, remove_empty_responses, extract_cot_and_response

THIS_FILE_PATH = Path(__file__)
DATA_DIR = "../data/pp-50-final"

YL_FORMATS = ["default", "self", "default-other", "other", "mixed"]

AXES = """sports, diet, politics, religion, age, profession, 
geographical location, gender, education level, AI professors, 
family marriage status""".split(", ")

MODEL_TO_SHORT = {
    "HuggingFaceH4/zephyr-7b-beta": "",
    "mistralai/Ministral-8B-Instruct-2410":  "_ministral8b",
    "meta-llama/Llama-3.2-3B-Instruct": "_llama323b",
    "meta-llama/Llama-3.2-1B-Instruct": "_llama321b",
}


def generate_dialogues_from_wiki_people_with_axis(
    n_responses: int = 100,
    max_batch_size: int = 20,
    existing_data_path: Optional[str] = None,
    data_dir: str = f"../data/pp-50-final",
) -> pd.DataFrame:
    """
    given path to existing personas.json (which contains persona names and axess),
    generates personal questions (X) for each person batches at a time

    Args:
        n_responses (int): number of responses to generate
        max_batch_size (int): maximum number of responses to generate per API call
        existing_data_path (str): path to existing personas.json
        data_dir (str): directory to save data in
        persona_subset (List[str]): list of persona names to generate

    Returns:
        pd.DataFrame: dataset with prompts column added to each persona
    """
    os.makedirs(data_dir, exist_ok=True)
    persona_df = pd.read_json(f"{data_dir}/personas.json", lines=True, orient="records")
    persona_df = persona_df.sample(frac=1)  # randomly shuffle order
    num_personas = len(persona_df)
    DATA_PATH = f"{data_dir}/all_{num_personas}p_{n_responses}d.json"
    if not os.path.isfile(DATA_PATH):
        df = pd.DataFrame(columns=["name", "prompt"])
    else:
        df = pd.read_json(DATA_PATH, lines=True, orient="records")
    if existing_data_path is not None:
        df_exist = pd.read_json(existing_data_path, lines=True, orient="records")
        df = pd.concat([df, df_exist])
        persona_df = persona_df[~persona_df.name.isin(df_exist.name.unique())]

    added_rows = []
    sampling_kwargs = RESPONSE_SAMPLING_KWARGS
    sampling_kwargs["max_tokens"] = 2000
    rounds = int(np.ceil(n_responses / max_batch_size))
    for i, row in tqdm(persona_df.iterrows(), total=len(persona_df)):
        if row["name"] in df.name:
            continue
        prompt_template = SAMPLE_X_GIVEN_WIKI_NAME_AXIS_PP_CHAT_3S
        prompt = prompt_template.format(NAME=row["name"], AXES=", ".join(row.axis), N_RESPONSES=max_batch_size)
        qs = query_questions([prompt] * rounds)
        added_rows.extend([{"name": row["name"], "prompt": q} for q in qs[:n_responses]])

    df = pd.concat([df, pd.DataFrame(added_rows)])
    df.to_json(DATA_PATH, lines=True, orient="records")
    return df



def generate_common_dialogues_from_wiki_people_with_axis(
    n_responses: int = 70,
    max_batch_size: int = 20,
    axes_subset: Optional[List[str]]=None,
    data_dir: str = "../data/pp-50-final"
) -> pd.DataFrame:
    """
    given path to existing personas.json (which contains persona names and axess),
    generates personal questions (X) for each person batches at a time

    Args:
        n_responses (int): number of responses to generate
        max_batch_size (int): maximum number of responses to generate per API call
        data_dir (str): directory to save data in
        axes_subset (List[str]): list of axes subset to consider

    Returns:
        pd.DataFrame: dataset with prompts column added to each persona
    """

    persona_df = pd.read_json(f"{data_dir}/personas.json", lines=True, orient="records")
    axes_to_consider = AXES
    if axes_subset is not None:
        axes_to_consider = axes_subset

    DATA_PATH = f"{data_dir}/all_{len(axes_to_consider)}a_{n_responses}d_common.json"
    os.makedirs(Path(DATA_PATH).parent, exist_ok=True)
    if not os.path.isfile(DATA_PATH):
        df = pd.DataFrame(columns=["name", "prompt"])
    else:
        df = pd.read_json(DATA_PATH, lines=True, orient="records")

    added_rows = []
    sampling_kwargs = RESPONSE_SAMPLING_KWARGS
    sampling_kwargs["max_tokens"] = 2000
    rounds = int(np.ceil(n_responses / max_batch_size))
    for axis in tqdm(axes_to_consider):
        axis_df = persona_df[persona_df.axis.apply(lambda x: True if f"{axis}: " in str(x) else False)]
        names = axis_df.name.unique()
        axis_subcategories = [get_category(name, axis, persona_df) for name in names]
        prompt_template = SAMPLE_X_GIVEN_WIKI_NAME_AXIS_COMMON_PP_CHAT_3S

        prompts = [prompt_template.format(
            NAMES=join_names(names), AXIS=axis,
            PERSON_CATEGORIES=join_names(axis_subcategories),
            N_RESPONSES=max_batch_size)
            for _ in range(rounds)]

        qs = query_questions(prompts)
        np.random.shuffle(qs)
        qs = qs[:n_responses]
        added_rows.extend([{"axis": axis, "prompt": q} for q in qs])

    df = pd.concat([df, pd.DataFrame(added_rows)])
    df.to_json(DATA_PATH, lines=True, orient="records")
    return df


def generate_axis_specific_wiki_persona():
    """
    Sample personas using axis to create conflicting opinions

    Returns:
        Saving the persona df in the output path
    """
    out_path = "../data/pp-50-final/personas.json"
    os.makedirs(Path(out_path).parent, exist_ok=True)

    rows = []
    for axis in tqdm(AXES):
        axis = axis.strip()
        prompt_template = SAMPLE_WIKI_PERSONA_AXIS_PP_CHAT.replace(
            "{sub-category}, {name}, {1-sentence brief description}", "REPLACE_BACK")
        prompt = prompt_template.format(AXIS=axis)
        prompt = prompt.replace("REPLACE_BACK", "{sub-category}, {name}, {1-sentence brief description}")
        sampling_kwargs = RESPONSE_SAMPLING_KWARGS
        completions = chat_completions(
            prompts=[prompt],
            **sampling_kwargs
        )
        for item in completions[0].split("\n-"):
            # category might be separated with :
            item = item.replace(":", ",") + item[20:]
            category, rest = item[:item.find(",")].replace("- ", "").strip(), item[item.find(",") + 1:].strip()
            name, description = rest[:rest.find(",")].strip(), rest[rest.find(",") + 1:].strip()
            rows.append({
                "axis": axis,
                "category": category,
                "name": name,
                "description": description
            })
    df = pd.DataFrame(rows)
    df = merge_persona_axis(df)
    df.to_json(out_path, lines=True, orient="records")


def fill_df_completions(df: pd.DataFrame, completions: List[str], n_responses: int,
                        cot_completions: Optional[List[str]] = None) -> pd.DataFrame:
    while completions:
        for i, row in df.iterrows():
            if cot_completions:
                assert len(row.y) == len(row.y_cot)
            priority_field = row.y if cot_completions is None else row.y_cot
            existing_responses = len(priority_field)

            if existing_responses < n_responses:
                num_filled = n_responses - existing_responses
                df.loc[i, "y"].extend(completions[:num_filled])
                completions = completions[num_filled:]
                if cot_completions:
                    df.loc[i, "y_cot"].extend(cot_completions[:num_filled])
                    cot_completions = cot_completions[num_filled:]

            if len(completions) == 0:
                break

        if df.y.apply(lambda x: len(x)).min() >= n_responses:
            break
    return df


def generate_automodel_completion_dataset(
        data_path,
        model_name,
        n_responses: int = 2,
        bs: int = 4,
        save_every_n_batch: int = 5,
        prompt_template: str = "",
        start_index: int = 0,
        number_of_prompts_to_complete: int = -1,
        **sampling_kwargs
):
    """take in dataset which has {name, x}, and generate 2 y for each row"""
    from automodel import Model, LlamaModel
    from alignment.data import apply_chat_template

    df = pd.read_json(data_path, lines=True, orient="records")

    completion_kwargs = AUTOMODEL_RESPONSE_SAMPLING_KWARGS.copy()
    completion_kwargs.update(sampling_kwargs)
    diff_str = "" if completion_kwargs == AUTOMODEL_RESPONSE_SAMPLING_KWARGS else "_" + "_".join(
        f"{k[:3]}{v}" for k, v in completion_kwargs.items() if v != AUTOMODEL_RESPONSE_SAMPLING_KWARGS[k])
    index_postfix = ""
    end_index = min(start_index + number_of_prompts_to_complete, len(df))
    if start_index != 0 or number_of_prompts_to_complete != -1:
        index_postfix = f"_{start_index}-{end_index}"

    df = df[start_index:end_index]
    out_data_path = data_path if "r_" in data_path else data_path.replace(".json",
                                                                          f"_{n_responses}r{diff_str}{prompt_template}{index_postfix}.json")
    if os.path.isfile(out_data_path):
        # df = pd.read_json(out_data_path, lines=True, orient="records")
        # because sometimes we may artificially change file name, so may not contain all x's, this way we only append y
        df["y"] = pd.read_json(out_data_path, lines=True, orient="records")["y"].tolist()
        df.loc[df.y.isna(), "y"] = [[] for _ in range(len(df[df.y.isna()]))]

    if "y" not in df.columns:
        df["y"] = [[] for _ in range(len(df))]

    model = LlamaModel(model_name, None, model_name, None) if "llama" in model_name.lower() else Model(model_name, None,
                                                                                                       model_name, None)

    dialogues = []
    for i, row in df.iterrows():
        if isinstance(row.y, list):
            non_empty_completions = len([y for y in row.y if len(y) > 1])
            if non_empty_completions < n_responses:
                dialogues.extend([row.prompt] * (n_responses - non_empty_completions))
    print(f"total of {len(df)} prompts, {sum(df.y.apply(lambda x: 1 if len(x) < n_responses else 0))} to complete, "
          f"{len(dialogues)} of responses to generate")
    # generate in batches, save each batch into df iteratively
    num_batches = int(np.ceil(len(dialogues) / bs))
    completions = []
    for i in tqdm(range(num_batches), desc="Batches"):
        dialogues_batch = dialogues[i * bs:i * bs + bs]
        if prompt_template == "":
            prompts = [
                apply_chat_template({"messages": [{"content": d, "role": "user"}]}, model.tokenizer, "generation")[
                    "text"] for d in dialogues_batch]
        elif prompt_template == "diverse":
            prompts = [apply_chat_template({"messages": [{
                                                             "content": f"You are a helpful assistant who is considerate of different people's preferences on response styles. Different people might prefer responses that are of different length, detail, complexity, content, writing style, opinion, etc. Simulate a set of preferences, and generate a responses to the following question that follows the preference. Do not output the preference, directly generate the response and do not mention the preferences in your response.\n{d}",
                                                             "role": "user"}]}, model.tokenizer, "generation")["text"]
                       for d in dialogues_batch]
        ys = model.completions(prompts, **completion_kwargs)
        completions.extend(ys)

        if i != 0 and i % save_every_n_batch == 0:
            df = fill_df_completions(df, completions, n_responses)
            df.to_json(out_data_path, lines=True, orient="records")
            completions = []

    df = fill_df_completions(df, completions, n_responses)
    df.to_json(out_data_path, lines=True, orient="records")


def generate_automodel_completion_diverse_cot_dataset(
        data_path,
        model_name,
        n_responses: int = 2,
        bs: int = 4,
        save_every_n_batch: int = 5,
        start_index: int = 0,
        number_of_prompts_to_complete: int = -1,
        number_of_cot: int = 5,
        new_cot_prompt: bool = False,
        gold_categories: bool = False,
        **sampling_kwargs
):
    """take in dataset which has {name, x}, and generate 2 y for each row"""
    from automodel import Model, LlamaModel
    from alignment.data import apply_chat_template
    print(f"Sampling responses from {model_name=}")
    print(f"{gold_categories=}")
    print(f"{n_responses=}")
    print(f"{start_index=}")
    print(f"{number_of_prompts_to_complete=}")

    df = pd.read_json(data_path, lines=True, orient="records")

    completion_kwargs = AUTOMODEL_RESPONSE_SAMPLING_KWARGS.copy()
    completion_kwargs.update(sampling_kwargs)
    diff_str = "" if completion_kwargs == AUTOMODEL_RESPONSE_SAMPLING_KWARGS else "_" + "_".join(
        f"{k[:3]}{v}" for k, v in completion_kwargs.items() if v != AUTOMODEL_RESPONSE_SAMPLING_KWARGS[k])
    index_postfix = ""
    end_index = min(start_index + number_of_prompts_to_complete, len(df))
    if start_index != 0 or number_of_prompts_to_complete != -1:
        index_postfix = f"_{start_index}-{end_index}"

    df = df[start_index:end_index]
    out_data_path = data_path if "r_" in data_path else \
        data_path.replace(".json", f"_{n_responses}r{diff_str}_cot"
                                   f"{'_new' if new_cot_prompt else '_gold' if gold_categories else ''}"
                                   f"{index_postfix}.json")
    if os.path.isfile(out_data_path):
        df["y"] = pd.read_json(out_data_path, lines=True, orient="records")["y"].tolist()
        df["y_cot"] = pd.read_json(out_data_path, lines=True, orient="records")["y_cot"].tolist()
        df.loc[df.y.isna(), "y"] = [[] for _ in range(len(df[df.y.isna()]))]
        df.loc[df.y_cot.isna(), "y_cot"] = [[] for _ in range(len(df[df.y_cot.isna()]))]

    if "y" not in df.columns:
        df["y"] = [[] for _ in range(len(df))]
        df["y_cot"] = [[] for _ in range(len(df))]

    model = LlamaModel(model_name, None, model_name, None) if "llama" in model_name.lower() else \
        Model(model_name, None, model_name, None)

    # first, get all CoT generations
    print("Generating CoT generations to get assumed axis and categories...")
    while any(df.y_cot.apply(lambda y_cots: len([y for y in y_cots if len(y) > 1])) < number_of_cot):
        cot_dialogues = []
        for i, row in df.iterrows():
            assert isinstance(row.y_cot, list)
            existing_completions = len(row.y_cot)
            if existing_completions < number_of_cot:
                system_msg = SAMPLE_Y_GIVEN_AXIS_DIVERSE_CHAT.split('<<<USER>>>')[0].replace("<<<SYSTEM>>>", "")
                prompt = apply_chat_template({"messages": [
                    {"content": system_msg, "role": "system"},
                    {"content": row.prompt, "role": "user"}]},
                    model.tokenizer, "generation")["text"]
                cot_dialogues.extend([prompt] * (number_of_cot - existing_completions))
        generate_completions(bs, completion_kwargs, df, cot_dialogues, model, True, number_of_cot, out_data_path,
                             save_every_n_batch, gold_cot=gold_categories)

        # remove empty completions (likely malformed CoT)
        remove_empty_responses(df, number_of_cot)

    print("now generating ys ...")
    # then, generate all regular generations after randomly sampling sub-axis
    dialogues = []
    # since we don't actually want original COT produced y, we generate 5 more y with new prompt, and
    # remove the 5 original COT produced 1
    n_responses += number_of_cot if new_cot_prompt or gold_categories else 0
    for i, row in df.iterrows():
        assert isinstance(row.y, list) and len(row.y_cot) == 5
        non_empty_completions = len([y for y in row.y if len(y) > 1])
        if non_empty_completions < n_responses:
            cot_index = np.repeat(range(number_of_cot), int(n_responses / number_of_cot))
            for j in range(non_empty_completions, n_responses):
                y_cot = row.y_cot[cot_index[j]]
                random_category = random.choice(y_cot[y_cot.lower().find("categories: "):].split(",")).strip()
                if new_cot_prompt:
                    system_msg = SAMPLE_Y_GIVEN_AXIS_DIVERSE_NEW_CHAT.split('<<<USER>>>')[0].replace("<<<STSTEM>>>", "")
                    axis = y_cot[y_cot.lower().find("axis: ")+5:].split("\n")[0].strip()
                    system_msg = system_msg.format(AXIS=axis, CATEGORY=random_category)
                    prompt = apply_chat_template({"messages": [{"content": system_msg, "role": "system"},
                                                               {"content": row.prompt, "role": "user"}]},
                                                 model.tokenizer, "generation")["text"]
                else:
                    system_msg = SAMPLE_Y_GIVEN_AXIS_DIVERSE_CHAT.split('<<<USER>>>')[0].replace("<<<STSTEM>>>", "")
                    prompt = apply_chat_template({"messages": [{"content": system_msg, "role": "system"},
                                                               {"content": row.prompt, "role": "user"},
                                                               {"content": f"{y_cot}\nChosen Category: {random_category}",
                                                                "role": "assistant"},
                                                               {"content": "", "role": "user"}]},
                                                 model.tokenizer, "generation")["text"]
                dialogues.append(prompt)
    generate_completions(bs, completion_kwargs, df, dialogues, model, False, n_responses, out_data_path,
                         save_every_n_batch, True)
    if new_cot_prompt or gold_categories:
        n_responses -= number_of_cot
        # we don't actually want the first few ys generated in one go with COT
        df["y"] = df.y.apply(lambda ys: ys[number_of_cot:] if len(ys)>n_responses else ys)
        df["y_cot"] = df.y_cot.apply(lambda y_cots: y_cots[number_of_cot:] if len(y_cots) > number_of_cot else y_cots)
        df.to_json(out_data_path, lines=True, orient="records")


def generate_completions(
    bs, completion_kwargs, df, dialogues, model, cot: bool,
    n_responses, out_data_path,
    save_every_n_batch, parse_cot=False, gold_cot=False
):
    y_field = "y_cot" if cot else "y"
    print(
        f"total of {len(df)} {'cot' if cot else 'regular'} prompts, {sum(df[y_field].apply(lambda x: 1 if len(x) < n_responses else 0))} to complete, "
        f"{len(dialogues)} of responses to generate")
    # generate in batches, save each batch into df iteratively
    num_batches = int(np.ceil(len(dialogues) / bs))
    completions = []
    cot_completions = []
    if gold_cot:
        persona_df = pd.read_json(f"{THIS_FILE_PATH.parents[1].absolute()}/data/pp-200/personas.json", lines=True,
                                  orient="records")
    for i in tqdm(range(num_batches), desc="Batches"):
        prompts = dialogues[i * bs:i * bs + bs]
        if gold_cot:
            ys = generate_completions_with_gold_categories(prompts, df, persona_df)
        else:
            ys = model.completions(prompts, **completion_kwargs)
        if cot or parse_cot:
            y_cots, ys = extract_cot_and_response(ys)
            cot_completions.extend(y_cots)
        completions.extend(ys)
        if i != 0 and i % save_every_n_batch == 0:
            df = fill_df_completions(df, completions, n_responses, cot_completions if cot else None)
            df.to_json(out_data_path, lines=True, orient="records")
            completions = []
            cot_completions = []

    df = fill_df_completions(df, completions, n_responses, cot_completions if cot else None)
    df.to_json(out_data_path, lines=True, orient="records")


def aggregate_automodel_completion_dataset(
        data_path, start: int = 0, num_indices: int = 200,
        num_files: int = 7, postfix: str = "", delete: bool = False
):
    data_dir = str(Path(data_path).parent)
    file_prefix = Path(data_path).name.replace(postfix, "").replace(".json", "")
    completion_files = [f for f in os.listdir(data_dir) if ("-" in f) and f.endswith(postfix)]
    end = start + num_indices * num_files
    ordered_files = []
    while start < end:
        fs = [f for f in completion_files if f"{file_prefix}_{start}-" in f]
        if len(completion_files) == 1 and f"_{start}-" in completion_files[0]:
            f = completion_files[0]
        else:
            assert len(fs) > 0, f"file with postfix {start}-XX not found"
            f = fs[0]
        ordered_files.append(f)
        completion_files.remove(f)
        start = start + num_indices
    print("joining following files together")
    pprint.pprint(ordered_files)
    df = pd.concat([pd.read_json(f"{data_dir}/{f}", lines=True, orient="records") for f in ordered_files],
                   ignore_index=True)
    # out_name = ordered_files[0].split("-")[0]+"-"+ordered_files[-1].split("-")[1]
    df.to_json(f"{data_path}", lines=True, orient="records")

    if delete:
        for f in ordered_files:
            os.remove(f"{data_dir}/{f}")


def select_other_persona_yw(df: pd.DataFrame, default_persona: bool = False) -> List[List[Dict]]:
    rejected = []
    for i, row in df.iterrows():
        if default_persona:
            df_sub = df[(df.prompt_id == row.prompt_id) & (df.persona.isna())]
        else:
            df_sub = df[(df.prompt_id == row.prompt_id) & (df.index != i) & (~df.persona.isna())]
        new_yl = random.choice(df_sub.yw.to_list())
        rejected.append([{"content": row.prompt, "role": "user"}, {"content": new_yl, "role": "assistant"}])
    return rejected


def select_mixed_yw_yl_pairs(df: pd.DataFrame, weight: np.array) -> Tuple[List[str], List[str]]:
    """Most general case of mixing different pairs of yw/yl given the assumption that
    yw_self > y_default > yw_others > yl_self. The weight vector indicates how much of
    each generation to sample from. yw and yl sampled without replacement"""
    assert len(weight) == 4
    res = {"chosen": [], "rejected": []}
    for i, row in df.iterrows():
        choices = sorted(np.random.choice(4, 2, replace=False, p=weight))  # choice for yw, yl
        for y_field, option in zip(["chosen", "rejected"], choices):
            if option == 0:  # yw_self
                y_str = row.yw
            elif option == 1:  # y_default
                y_str = df[(df.prompt_id == row.prompt_id) & (df.persona.isna())].iloc[0].yw
            elif option == 2:  # yw_others
                df_sub = df[(df.prompt_id == row.prompt_id) & (df.index != i) & (~df.persona.isna())]
                y_str = random.choice(df_sub.yw.to_list())
            else:  # yl_self
                y_str = row.yl

            response = [{"content": row.prompt, "role": "user"}, {"content": y_str, "role": "assistant"}]
            res[y_field].append(response)
    return res["chosen"], res["rejected"]


SHUFFLED_NAME_MAP = {}
NAME_TO_TAG = {}


def format_dataset(
    data_path: str,
    yl: str = "self",
    format: str = "prefs",
    mix_weight: Optional[Tuple[float, float, float, float]] = None,
    prefix_prompt_w_persona: Union[bool, str] = False,
    split_question_type: bool = False,
    overwrite_cache: bool = True,
    return_df: bool = False,
    persona_df: Optional[pd.DataFrame] = None,
):
    """convert from yw yl format to dpo or sft format, additionally decide what to use for rejected"""
    # make output directory
    file_name = Path(data_path).name
    dir_name = "_".join(file_name.split("_")[1:]).replace(".json", f'_{yl}-yl')
    dir_name += f"_{prefix_prompt_w_persona}-name-prefixed" if isinstance(prefix_prompt_w_persona, str) else \
        "_name-prefixed" if prefix_prompt_w_persona else ""
    if dir_name.startswith("_"):
        dir_name = dir_name[1:]
    split = file_name.split("_")[0].replace(".json", "")
    output_dir = Path(data_path).parent.joinpath(dir_name)
    
    if not return_df:
        if not overwrite_cache and os.path.isdir(output_dir):
            print(f"output directory exists and not overwriting, skip! {output_dir=}")
            return output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    df = pd.read_json(data_path, lines=True, orient="records")
    # add additional persona from personas.json if needed
    if persona_df is None:
        persona_df = pd.read_json(Path(data_path).parent.joinpath("personas.json"), lines=True, orient="records")
    if prefix_prompt_w_persona in persona_df.columns and prefix_prompt_w_persona not in df.columns:
        df = df.merge(persona_df[["name", prefix_prompt_w_persona]], on="name", how="left")
    if isinstance(prefix_prompt_w_persona,str) and prefix_prompt_w_persona.startswith("tag") and "tag" in persona_df.columns and "tag" not in df.columns:
        df = df.merge(persona_df[["name", "tag"]], on="name", how="left")

    # we directly put shuffled columns in persona df now
    # elif prefix_prompt_w_persona.replace("shuffle_", "") in persona_df.columns and prefix_prompt_w_persona!= "shuffle_name":
    #     df = df.merge(persona_df[["name", prefix_prompt_w_persona.replace("shuffle_", "")]], on="name", how="left")
    global SHUFFLED_NAME_MAP, NAME_TO_TAG

    if prefix_prompt_w_persona:
        if (prefix_prompt_w_persona not in df.columns) and (prefix_prompt_w_persona not in ["tag_plus"]):
            raise Exception(f"{prefix_prompt_w_persona} not in column, are you sure about this?")

        if "persona" in df.columns and "name" in df.columns and prefix_prompt_w_persona == "persona":
            df["prompt"] = df.apply(
                lambda r: SAMPLE_YW_GIVEN_PERSONA_PP.format(NAME=r['name'], PERSONA=r.persona, x=r.prompt), 1)
        if isinstance(prefix_prompt_w_persona, str) and "persona_" in prefix_prompt_w_persona and prefix_prompt_w_persona in df.columns:
            df["prompt"] = df.apply(
                lambda r: SAMPLE_YW_GIVEN_PERSONA_ANONYMOUS_PP.format(PERSONA=r[prefix_prompt_w_persona], x=r.prompt), 1)
        elif isinstance(prefix_prompt_w_persona, str) and "xyw" in prefix_prompt_w_persona and prefix_prompt_w_persona in df.columns:
            df["prompt"] = df.apply(
                lambda r: SAMPLE_YW_GIVEN_XY_PP.format(XY_LIST=r[prefix_prompt_w_persona], x=r.prompt), 1)
        elif isinstance(prefix_prompt_w_persona, str) and "tag" in prefix_prompt_w_persona and "tag" in df.columns:
            if prefix_prompt_w_persona == "tag":
                df["prompt"] = df.apply(lambda r: f"{r.tag} {r.prompt}", 1)
            elif prefix_prompt_w_persona == "tag_plus":
                df["prompt"] = df.apply(lambda r: SAMPLE_YW_GIVEN_NAME_PP.format(NAME=r.tag, x=r.prompt), 1)
            else:
                raise NotImplementedError
        elif isinstance(prefix_prompt_w_persona, str) and "name" in prefix_prompt_w_persona and prefix_prompt_w_persona in df.columns:
            df["prompt"] = df.apply(lambda r: SAMPLE_YW_GIVEN_NAME_PP.format(NAME=r[prefix_prompt_w_persona], x=r.prompt), 1)

        else:
            raise NotImplementedError

    if format == "prefs":
        if yl == "self":
            df["rejected"] = df.apply(
                lambda r: [{"content": r.prompt, "role": "user"}, {"content": r.yl, "role": "assistant"}], 1)
        elif yl == "other":
            df["rejected"] = select_other_persona_yw(df)
        elif yl == "default":
            df["rejected"] = select_other_persona_yw(df, default_persona=True)
        df["chosen"] = df.apply(
            lambda r: [{"content": r.prompt, "role": "user"}, {"content": r.yw, "role": "assistant"}], 1)
        if yl == "default-other":
            df["chosen"] = select_other_persona_yw(df, default_persona=True)
            df["rejected"] = df.apply(
                lambda r: [{"content": r.prompt, "role": "user"}, {"content": r.yl, "role": "assistant"}], 1)
        elif yl == "mixed":
            # weight is defined as proportions/probability of [yw_self, y_default, yw_others, yl_self]
            weight = np.array(mix_weight) if mix_weight is not None else np.array(
                [1, 1, len(df[~df.persona.isna()].persona.unique()) - 1, 1])
            if sum(weight) > 1:
                weight = weight / sum(weight)
            (df["chosen"], df["rejected"]) = select_mixed_yw_yl_pairs(df, weight)
    else:
        df["messages"] = df.apply(
            lambda r: [{"content": r.prompt, "role": "user"}, {"content": r.yw, "role": "assistant"}], 1)

    # remove unnecessary columns
    cols = ["name", "prompt", "chosen", "rejected"]
    for c in ["question_type", "cv"]:
        if c in df.columns:
            cols.append(c)
    df = df[cols]

    if return_df:
        return df
    
    if split_question_type and "question_type" in df.columns and split!="train": # :
        for question_type in ["personal", "common"]:
            df_sub = df[df.question_type==question_type]
            del df_sub["question_type"]
            df_sub.to_json(output_dir.joinpath(f"{split}_{question_type}_{format}.json"), lines=True, orient="records")
    else:
        df.to_json(output_dir.joinpath(f"{split}_{format}.json"), lines=True, orient="records")

    return output_dir


def format_dataset_factory(
    data_path: str,
    yl: str = "self",
    format: str = "prefs",
    mix_weight: Optional[Tuple[float, float, float, float]] = None,
    prefix_prompt_w_persona: Union[bool, str, List[str]] = False,
    split_question_type: bool = False,
    overwrite_cache: bool = True
):
    if not isinstance(prefix_prompt_w_persona, list):
        format_dataset(data_path, yl, format, mix_weight, prefix_prompt_w_persona, split_question_type, overwrite_cache)
        return

    prefixed_paths = []
    for prefix in prefix_prompt_w_persona:
        split_path = format_dataset(data_path, yl, format, mix_weight, prefix, split_question_type, overwrite_cache)
        prefixed_paths.append(split_path)

    # now combine all prefixed dataset paths
    print("combining prefixed datasets ...")
    combined_path = str(prefixed_paths[0]).replace("-name-prefixed", f"_agg{len(prefix_prompt_w_persona)}-name-prefixed")
    os.makedirs(combined_path, exist_ok=True)
    for f in tqdm(os.listdir(prefixed_paths[0]), total=len(os.listdir(prefixed_paths[0]))):
        p = f"{prefixed_paths[0]}/{f}"
        if os.path.isfile(p):
            combine_files([f"{d}/{f}" for d in prefixed_paths], out_dir=combined_path)
        elif os.path.isdir(p):
            # cross validation directory, go one depth further
            for f1 in os.listdir(p):
                os.makedirs(f"{combined_path}/{f}", exist_ok=True)
                combine_files([f"{d}/{f}/{f1}" for d in prefixed_paths], out_dir=f"{combined_path}/{f}")


def combine_files(paths: List[str], out_dir: str):
    """given multiple datasets, combine them and output aggregated data to output_dir"""
    assert len(set([p.split("/")[-1] for p in paths])) == 1
    out_path = out_dir + "/" + paths[0].split("/")[-1]
    lines = []
    for p in paths:
        with open(p, "r") as f:
            lines.extend(f.readlines())
    with open(out_path, "w") as f:
        f.writelines(lines)


def split_dialogue_train_valid(data_path, num_train, num_test):
    """Given a path do all_.....json dataset file, split it according to some dialogue subsets"""
    df = pd.read_json(data_path, lines=True, orient="records")
    # if x sourced from common dataset (ultrachat 200K), then we split by dialogue_id, which is common across different persona
    if "prompt_id" in df.columns:
        assert len(df.prompt_id.unique()) == (num_train + num_test)
        assert "all" in data_path
        prompt_ids = df.prompt_id.unique()
        np.random.shuffle(prompt_ids)
        train_ids, test_ids = prompt_ids[:num_train], prompt_ids[num_train:]
        df_train = df[df.prompt_id.isin(train_ids)]
        df_test = df[df.prompt_id.isin(test_ids)]
    else:  # otherwise, x is different across persona, so we randomly split per person
        df_train, df_test = pd.DataFrame(), pd.DataFrame()
        for name in df.name.unique():
            sub_df = df[df.name == name]
            df_train_sub, df_test_sub = train_test_split(sub_df, train_size=num_train / (num_train + num_test),
                                                         random_state=24)
            df_train = pd.concat([df_train, df_train_sub])
            df_test = pd.concat([df_test, df_test_sub])
    df_train.to_json(data_path.replace("all_", "train_"), lines=True, orient="records")
    df_test.to_json(data_path.replace("all_", "test_"), lines=True, orient="records")


def split_into_individual_json(data_dir, splits: List[str] = ["train", "test"]):
    """In this directory, we expect something like train_prefs.json, test_prefs.json"""
    # splits = ["train", "test"]
    files = [f for f in os.listdir(data_dir) if any(f.startswith(s) for s in splits)]
    cnt = 0
    for f in files:
        cnt += 1
        df = pd.read_json(f"{data_dir}/{f}", lines=True, orient="records")
        persona_column = "name" if "name" in df.columns else "persona"
        for persona in df[persona_column].unique():
            if "name" in df.columns:
                name = persona.lower().replace(" ", "")
            else:
                if persona is None:  # not exporting default response
                    continue
                name = extract_persona_name(persona).lower().replace(" ", "")
            sub_df = df[df[persona_column] == persona]
            if len(sub_df) == len(df):  # this means the file is already persona specific
                continue
            f_split = f.split("_")
            f_split[0] += f"_{name}"
            new_f = "_".join(f_split)
            sub_df.to_json(f"{data_dir}/{new_f}", lines=True, orient="records")



def filter_generic_reward(df: pd.DataFrame, model_name: str, num_to_keep: int = 30) -> pd.DataFrame:
    """Find N number of responses with least differences in their rewards calculated by a generic reward model"""
    print(f"filtering response with generic reward model: {model_name}")

    # copied from https://huggingface.co/sfairXC/FsfairX-LLaMA3-RM-v0.1
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
    filtered_y = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        chats = [[{"role": "user", "content": row.prompt}, {"role": "assistant", "content": y}] for y in row.y]
        test_texts = [rm_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False).replace(
            rm_tokenizer.bos_token, "") for chat in chats]
        pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
        rewards = np.array([output[0]["score"] for output in pipe_outputs])
        sorted_indices = np.argsort(rewards)

        # find a contiguous segment of 20 samples with minimum range in rewards
        sorted_rewards = np.sort(rewards) - min(rewards)  # make it positive so convolve doesn't have weird effects
        conv_filter = np.zeros(num_to_keep)
        conv_filter[0], conv_filter[-1] = 1, -1  # subtract first element by last element in span,
        range_diff = np.convolve(sorted_rewards, conv_filter, "valid")  # gives us range of the span
        start = range_diff.argmin()  # then we get the span with the smallest range
        end = start + num_to_keep

        ys = [row.y[i] for i in sorted_indices[start:end]]
        filtered_y.append(ys)
    df["y"] = filtered_y
    return df


def filter_diverse_responses(
    data_path: str,
    n_responses: int = 4,
    filter_generic_reward_model: Optional[str] = None,
    num_to_keep: int = 30,
    debug: bool = False,
):
    df = pd.read_json(data_path, lines=True, orient="records")
    if debug:
        df = df[:20]  # debug
    out_path = data_path.replace(".json",
                                 f"_filtered{str(num_to_keep) + 'm' if filter_generic_reward_model is not None else ''}{n_responses}k.json")

    if filter_generic_reward_model is not None:
        df = filter_generic_reward(df, filter_generic_reward_model, num_to_keep)

    sent_model = SentenceTransformer("sentence-transformers/sentence-t5-xxl")
    diverse_ys_all = []
    random_ys_all = []
    print("Now filtering by clustering and selecting diverse examples")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        embs = sent_model.encode(row.y, convert_to_tensor=True)
        embs = embs / torch.sqrt((embs ** 2).sum(axis=1))[:, None]  # normalize, so we can run kmeans
        km_model = sklearn.cluster.KMeans(n_clusters=4)
        km_model.fit(embs.cpu())
        diverse_ys = []
        random_ys = []
        for k in range(n_responses):
            embs, km_model.labels_ = embs.cuda(), torch.tensor(km_model.labels_).cuda()
            km_model.cluster_centers_ = torch.tensor(km_model.cluster_centers_).cuda()

            # for each cluster, find the example in the cluster farthest from all other clusters
            cluster_embs = embs[km_model.labels_ == k]
            # [n_examples, emb] X [emb, n_clusters] -> [n_examples, n_cluster] -> [n_examples]
            all_but_cluster_id = [ci for ci in range(n_responses) if ci != k]
            cluster_sum_distances = (cluster_embs @ km_model.cluster_centers_[all_but_cluster_id].T.float()).sum(1)

            # take argmin example (least similar to other clusters)
            try:
                sample_emb = cluster_embs[torch.argmin(cluster_sum_distances)]
            except Exception as e:
                print(f"Issue taking argmin, liklely empty cluster sum distances? {e}")
                print(f"resolve to taking random samples from clusters")
                pdb.set_trace()
                sample_emb = cluster_embs[torch.randint(len(cluster_embs), [1])]
            row_index_candidates = torch.where(torch.all(embs == sample_emb, dim=1))[0].cpu()
            row_index = row_index_candidates.item() if len(row_index_candidates) < 2 else row_index_candidates[0].item()
            diverse_ys.append(row.y[row_index])

            # now also randomly select y from each cluster
            sample_emb = cluster_embs[torch.randint(len(cluster_embs), [1])]
            row_index_candidates = torch.where(torch.all(embs == sample_emb, dim=1))[0].cpu()
            row_index = row_index_candidates.item() if len(row_index_candidates) < 2 else row_index_candidates[0].item()
            random_ys.append(row.y[row_index])

        diverse_ys_all.append(diverse_ys)
        random_ys_all.append(random_ys)
    df["diverse_ys"] = diverse_ys_all
    df["random_ys"] = random_ys_all
    df.to_json(out_path, lines=True, orient="records")


def split_into_cross_validation_for_meta_learning(data_dir: str, folds: Union[int, Literal["axis"]] = 5, subset=True):
    """
    split into cross validation for multitask training

    Args:
        data_dir (str): path to data directory
        folds (int | "axis"): number of folds to split the data, default 5, if "axis" leave-one-axis-out split
        subset (bool): True to subset the data, False to not subset the data, default True
    """
    all_train_df = pd.read_json(f"{data_dir}/train_prefs.json", lines=True, orient="records")
    common_test_df = pd.read_json(f"{data_dir}/test_common_prefs.json", lines=True, orient="records")
    personal_test_df = pd.read_json(f"{data_dir}/test_personal_prefs.json", lines=True, orient="records")
    if not subset:
        df_persona = pd.read_json(f"{Path(data_dir).parent}/personas.json", lines=True, orient="records")
    common_test_df["question_type"], personal_test_df["question_type"] = "common", "personal"
    all_test_df = pd.concat([common_test_df, personal_test_df])
    # names = np.random.permutation(all_train_df.name.unique())
    if subset:
        assert isinstance(folds, int), "subset only available with multi-fold CV"
        names = "Halle Berry,Donald Trump,Bernie Sanders,Jennifer Aniston,Alexandria Ocasio-Cortez,Joe Biden,Gwyneth Paltrow,Megan Fox,Rand Paul,Ellen DeGeneres".split(",")
        test_size = int(len(names) / folds)

    if not isinstance(folds, int):
        folds = ['profession', 'geographical location', 'family marriage status', 'age', 'politics', 'sports', 'diet', 'religion', 'education level', 'AI professors', 'gender']
    else:
        folds = range(folds)

    for cv_i in folds:
        cv_dir = f"{data_dir}/cv{cv_i.replace(' ','_') if isinstance(cv_i, str) else cv_i}"
        os.makedirs(cv_dir, exist_ok=True)
        if subset:
            train_names, test_names = names[test_size:], names[:test_size]
        elif isinstance(cv_i, int):
            train_names, test_names = df_persona[df_persona.cv!=cv_i].name.unique(), df_persona[df_persona.cv==cv_i].name.unique()
        else:
            train_names, test_names = (df_persona[df_persona.axis.apply(lambda x: f"{cv_i}:" not in str(x))].name.unique(),
                                       df_persona[df_persona.axis.apply(lambda x: f"{cv_i}:" in str(x))].name.unique())
        meta_train_df = all_train_df[all_train_df.name.isin(train_names)]
        meta_test_df = all_test_df[all_test_df.name.isin(train_names)]
        # inference_train_df = all_train_df[all_train_df.name.isin(test_names)]
        inference_test_df = all_test_df[all_test_df.name.isin(test_names)]
        meta_train_df.to_json(f"{cv_dir}/train_meta_prefs.json", lines=True, orient="records")
        meta_test_df.to_json(f"{cv_dir}/test_meta_prefs.json", lines=True, orient="records")
        # inference_train_df.to_json(f"{cv_dir}/train_inference_prefs.json", lines=True, orient="records")
        meta_train_df[meta_train_df.question_type == "common"].drop(columns=["question_type"]).to_json(
            f"{cv_dir}/train_meta_common_prefs.json", lines=True, orient="records")
        meta_train_df[meta_train_df.question_type == "personal"].drop(columns=["question_type"]).to_json(
            f"{cv_dir}/train_meta_personal_prefs.json", lines=True, orient="records")

        # split_into_individual_json(cv_dir, ["test_inference"])
        if subset:
            names = np.concatenate([names[test_size:], names[:test_size]])


def distribute_common_questions(common_df) -> pd.DataFrame:
    persona_df = pd.read_json(f"{THIS_FILE_PATH.parents[1].absolute()}/data/pp-200/personas.json", lines=True,
                              orient="records")
    axis_to_names = {}
    for i, row in persona_df.iterrows():
        for axis in row.axis:
            axis_name, subcategory = axis.split(": ")
            if axis_name not in axis_to_names:
                axis_to_names[axis_name] = [row['name']]
            else:
                axis_to_names[axis_name].append(row['name'])

    added_df = []
    for i, row in common_df.iterrows():
        for name in axis_to_names[row.axis]:
            added_df.append({"name": name, **{k: v for k, v in row.to_dict().items() if k != "name"}})
    added_df = pd.DataFrame(added_df)
    return added_df


def visualize_question_distribution(data_path, common_q_data_path=None, no_personal_questions: bool = False):
    import seaborn as sns
    import matplotlib.pyplot as plt
    df = pd.read_json(data_path, lines=True, orient="records")
    sent_model = SentenceTransformer("sentence-transformers/sentence-t5-xxl")
    embs = sent_model.encode(df.prompt, convert_to_tensor=True, show_progress_bar=True)
    embs = (embs / torch.sqrt((embs ** 2).sum(axis=1))[:, None]).cpu().numpy()
    tsne = TSNE(n_components=2, perplexity=30, verbose=1, random_state=10, n_iter=300)

    # Fit the t-SNE model to the data
    if common_q_data_path is None:
        X_tsne = tsne.fit_transform(embs)
        df["tsne dim1"], df["tsne dim2"] = X_tsne[:, 0], X_tsne[:, 1]
    else:
        common_df = pd.read_json(common_q_data_path, lines=True, orient="records")
        common_embs = sent_model.encode(common_df.prompt, convert_to_tensor=True, show_progress_bar=True)
        common_embs = (common_embs / torch.sqrt((common_embs ** 2).sum(axis=1))[:, None]).cpu().numpy()
        combined_embs = np.concatenate([embs, common_embs])
        X_tsne = tsne.fit_transform(combined_embs)
        common_df["tsne dim1"], common_df["tsne dim2"] = X_tsne[-len(common_df):, 0].tolist(), X_tsne[-len(common_df):,
                                                                                               1].tolist()
        df["tsne dim1"], df["tsne dim2"] = X_tsne[:len(df), 0].tolist(), X_tsne[:len(df), 1].tolist()
        added_df = distribute_common_questions(common_df)
        # optionally only include data in current dataset size (30 people)
        added_df = added_df[added_df.name.isin(df.name.unique())]

        if no_personal_questions:
            df = added_df
        else:
            df = pd.concat([df, added_df])

    # Create a scatter plot of the data, colored by group
    plt.figure(figsize=(16, 10))
    # rndperm = np.random.permutation(df.shape[0])
    sns.scatterplot(
        x="tsne dim1", y="tsne dim2",
        hue="name",
        palette=sns.color_palette("hls", len(df.name.unique())),
        data=df,  # df.loc[rndperm, :],
        legend="full",
        alpha=0.3
    )

    # Show the plot
    # plt.show()
    plt.savefig(data_path.replace(".json",
                                  f"_x_tsne{'' if common_q_data_path is None else '_common_no_personal' if no_personal_questions else '_common'}.png"))


def sample_persona_given_x_y(
    train_path,
    num_shots,
    model_name,
    x_only: bool = False,
    reveal_name: bool = False,
    opinion_qa_prompt: bool = False,
    persona_field_suffix: str = "",
    seed: Optional[int]=None,
    data_indices: Optional[Union[List[int], Dict[str, List[int]], Dict[str, List[List[int]]]]]=None,
    names: List[str]=None,
    model_name_str: str = "",
    return_single_persona_str: bool=False,
):
    """
    Sample a persona description (with online preferences) from model (can be baseline model or GPT4)
    Args:
        train_path: train dataset
        num_shots: how many shots to sample from training data
        model_name: name of the model (either hf name or gpt4)
        x_only: if only prompt with question (x) and not with responses (yw + yl)
        reveal_name: if true, reveal person's name,
        opinion_qa_prompt: if true, use a separate set of prompts specific to OpinionQA
        persona_field_suffix: if any suffix for persona's field,
        seed: random state for sampling
        data_indices: few-shot indices, if list[int], we are using fixed few-shot indices,
            if a dictionary of str->list[int], it is indices mapped by name
        names: Optional, if inputted, only generate persona for these names
        model_name_str: if any suffix for model name,
        return_single_persona_str: if true, do not infer persona for everyone, just do it for one person one prompt at
            a time and respond. useful for inferring retrieval specific persona per question
    Returns:
    """

    if return_single_persona_str:
        assert isinstance(data_indices, dict) and len(names) == 1, "If returning single persona, need to specify which shot to use for whom"
    persona_data_path = str(Path(train_path).parent.joinpath("personas.json"))
    persona_df = pd.read_json(persona_data_path, lines=True, orient="records")
    train_df = pd.read_json(train_path, lines=True, orient="records")
    if reveal_name:
        persona_field = "persona_gold"
    else:
        persona_field = f"persona_x_{num_shots}s" if x_only else f"persona_xy_{num_shots}s"
    if "gpt4" in model_name:
        persona_field += "_gpt4"
    elif "ds" in model_name:
        persona_field += "_ds"
    else:
        # if os.path.isdir(model_name):
        #   persona_field += persona_field_suffix
        from automodel import Model, LlamaModel
        from alignment.data import apply_chat_template
        model = LlamaModel(model_name, None, model_name, None) if ("llama" in model_name.lower().replace("llamagrp",""))\
            else None if "gpt4" in model_name.replace("llamagrp","") \
            else Model(model_name, None, None, None)   # else Model(model_name, None, model_name, None)
            # issue reloading finetuned model, but let's just load their original tokneizer for now
        persona_field += model_name_str
    if seed is not None:
        persona_field += f"_seed{seed}"
    if persona_field_suffix:
        persona_field += persona_field_suffix

    completion_kwargs = RESPONSE_SAMPLING_KWARGS if (("gpt4" in model_name) or ("ds" in model_name)) else AUTOMODEL_RESPONSE_SAMPLING_KWARGS
    if names is not None:
        train_df = train_df[train_df.name.isin(names)]

    for name in tqdm(train_df.name.unique(), total=len(train_df.name.unique())) if not return_single_persona_str else train_df.name.unique():
        if isinstance(data_indices, dict):
            person_indices = data_indices[name.replace(" ", "").lower()]
        else:
            person_indices = data_indices

        shot_indices_list = [person_indices] if isinstance(person_indices, int) else [None] if person_indices is None else person_indices
        completions = []
        if return_single_persona_str:
            print(f"inferencing {len(shot_indices_list)} prefixes for {name}...")
        for shot_indices in tqdm(shot_indices_list) if return_single_persona_str else shot_indices_list:
            prompt = format_persona_inference_prompt(train_df, persona_df,
                "gpt4" not in model_name, name, num_shots, opinion_qa_prompt,
                reveal_name, x_only, seed, shot_indices)

            if "gpt4" in model_name:
                completion = chat_completions([prompt], **completion_kwargs)[0].strip()
            elif "ds" in model_name:
                completion = deepseek_chat_completions([prompt], **completion_kwargs)[0].strip()
            else:
                prompt = apply_chat_template(prompt, model.tokenizer, "generation")["text"]
                completion = model.completions([prompt], **completion_kwargs)[0].strip()

            completions.append(completion)

        if return_single_persona_str:
            torch.cuda.empty_cache()
            return completions
        else:
            assert len(completions) == 1

        persona_df.loc[persona_df.name==name, persona_field] = completions[0]
    persona_df.to_json(persona_data_path, lines=True, orient="records")


def add_xy_fewshot_as_prefix_to_persona(
    train_path:str,
    num_shots: int,
    include_yl: bool=True,
    seed: Optional[int]=None,
    data_indices: Optional[List[int]]=None,
    return_df: bool=False,
    names: List[str]=None,
):
    if data_indices is not None:
        assert len(data_indices) == num_shots
        
    persona_data_path = str(Path(train_path).parent.joinpath("personas.json"))
    persona_df = pd.read_json(persona_data_path, lines=True, orient="records")
    train_df = pd.read_json(train_path, lines=True, orient="records")
    for i, row in tqdm(persona_df.iterrows(), total=len(persona_df)):
        if row["name"] not in train_df.name.unique():
            continue  # subset does not include the person
        if names is not None and row["name"] not in names:
            continue  # allow generating prefix for only selective people
        prefix_name, prompt = format_few_shot_prompt(
            train_df[train_df.name == row["name"]], num_shots, include_yl, seed, data_indices)
        persona_df.loc[i, prefix_name] = prompt

    if return_df:
        return persona_df
    persona_df.to_json(persona_data_path,  lines=True, orient="records")


def format_few_shot_prompt(
    train_name_df: pd.DataFrame,
    num_shots: int,
    include_yl: bool=True,
    seed: Optional[int]=None,
    data_indices: Optional[List[int]] = None,
):
    if data_indices is not None:
        train_sub_df = train_name_df.reset_index().iloc[data_indices]
    elif seed is None:
        train_sub_df = train_name_df.sample(n=min(num_shots, len(train_name_df)))
    else:
        train_sub_df = train_name_df.sample(n=min(num_shots, len(train_name_df)), random_state=seed)
    prompt = ""
    for j, train_row in train_sub_df.iterrows():
        # This follows prompt format SAMPLE_YW_GIVEN_XY_PP in sampling/prompt.py
        prompt += f"## Prompt:\n{train_row.prompt}\n"
        prompt += f"### Preferred Response:\n{train_row.yw}\n"
        if include_yl:
            prompt += f"### Dispreferred Response:\n{train_row.yl}\n"
        prompt += "\n"
    prefix_name = f"xyw_{num_shots}s" if not include_yl else f"xywyl_{num_shots}s"
    if seed is not None:
        prefix_name += f"_seed{seed}"
    if data_indices is not None:
        prefix_name += f"_idx{''.join(str(i) for i in data_indices)}"
    return prefix_name, prompt.strip()


def combine_and_train_test_split_personal_and_common_questions(
    personal_data_path:str,
    common_data_path:str,
    personal_eval:int=20,
    common_eval_per_axis:int=10,
    overlapping_common_questions: bool=True,
    max_common_train:int=50,
):
    personal_df = pd.read_json(personal_data_path, lines=True, orient="records")
    common_df = pd.read_json(common_data_path, lines=True, orient="records")
    personal_df["question_type"] = "personal"
    common_df["question_type"] = "common"
    df_train, df_test = pd.DataFrame(), pd.DataFrame()

    # first split personal questions
    for name in personal_df.name.unique():
        sub_df = personal_df[personal_df.name == name]
        sub_df = sub_df.drop_duplicates("prompt")
        df_train_sub, df_test_sub = train_test_split(sub_df, train_size=len(sub_df)-personal_eval, random_state=24)
        df_train = pd.concat([df_train, df_train_sub])
        df_test = pd.concat([df_test, df_test_sub])

    # then split common questions, do it per axis
    common_train_map={}
    for name in common_df.name.unique():
        df_axis_train = []
        df_axis_test = []
        for axis in common_df.axis.unique():
            sub_df = common_df[(common_df.axis == axis) & (common_df.name == name)]
            if len(sub_df) == 0:
                continue
            if overlapping_common_questions:
                # default scenario, where common question a might be in the train split for one person but test split
                # for another person. but if we keep the seed the same, we are essentially keeping train/test the same
                # across people in the same axis
                df_train_sub, df_test_sub = train_test_split(sub_df, train_size=len(sub_df)-common_eval_per_axis, random_state=24)
            else:
                # alternative scenario where eval questions are never seen in anyone's training
                if len(common_train_map) == 0:
                    df_train_sub, df_test_sub = train_test_split(sub_df, train_size=len(sub_df)-common_eval_per_axis, random_state=24)
                    common_train_map.update({p: True for p in df_train_sub.prompt})
                    common_train_map.update({p: False for p in df_test_sub.prompt})
                else:
                    df_train_sub = sub_df[sub_df.prompt.apply(common_train_map)]
                    df_test_sub = sub_df[~sub_df.prompt.apply(common_train_map)]
            df_axis_train.append(df_train_sub)
            df_axis_test.append(df_test_sub)
        df_axis_train = pd.concat(df_axis_train)
        if max_common_train < len(df_axis_train):
            df_axis_train = df_axis_train.sample(n=max_common_train, replace=False)
        df_train = pd.concat([df_train, df_axis_train])
        df_test = pd.concat([df_test, pd.concat(df_axis_test)])

    def get_num_dialogues(p):
        return int(p.split("/")[-1].split("_")[2].replace("d",""))
    personal_d = get_num_dialogues(personal_data_path)
    d = personal_d + get_num_dialogues(common_data_path)
    out_path = personal_data_path.replace(f"{personal_d}d", f"{d}d_total")
    df_train.to_json(out_path.replace("all_", "train_"), lines=True, orient="records")
    df_test.to_json(out_path.replace("all_", "test_"), lines=True, orient="records")


def add_shuffled_prefices(prefix: str, persona_df_path: str, recover_from_shuffle: bool=False):
    print(f"adding shuffles for prefix {prefix}")
    df = pd.read_json(persona_df_path, lines=True, orient="records")
    # to ensure shuffle for every prefix has same mapping from old person to new person
    # we shuffle name first, then just deterministically map from old prefix to new prefix
    if "shuffle_name" not in df.columns:
        og_names = df.loc[~df[prefix].isna(), f"name"]
        shuffled_names = np.random.permutation(og_names)
        while sum(shuffled_names == og_names) > 1: # ensure every value is different
            shuffled_names = np.random.permutation(og_names)
        df.loc[~df[prefix].isna(), f"shuffle_name"] = shuffled_names

    # deterministically shuffle based on name->shuffle_name mapping
    if recover_from_shuffle:
        # re-set original column from shuffled column
        mapping = dict(zip(df.loc[~df["name"].isna(), "shuffle_name"], df.loc[~df["name"].isna(), 'name']))
        df[prefix] = df['name'].map(mapping).map(df.set_index('name')[f'shuffle_{prefix}'])
    else:
        mapping = dict(zip(df.loc[~df["name"].isna(),'name'], df.loc[~df["name"].isna(), "shuffle_name"]))
        df[f'shuffle_{prefix}'] = df['name'].map(mapping).map(df.set_index('name')[prefix])
    df.to_json(persona_df_path, lines=True, orient="records")


def add_retrieval_shots(
    data_dir: str,
    top_k: int = 2,
    method: Union[Literal["emb"], Literal["bm25"]]="bm25",
    reverse: bool = False,
    infer_persona: bool = True,
    persona_inference_model: str="HuggingFaceH4/zephyr-7b-beta",
    splits: List[str] = ["test"],
):
    """
    for every datapoint, retrieve top k similar shots from train split of the person,
    and infer the persona for that shot as well as few-shot

    Args:
        data_dir (str): path to the data directory with no prefix
        top_k (int): number of top shots to retrieve, default is 2.
        method (str): method to use for retrieving top shots. Default is "bm25".
        reverse (bool): whether to get farthest shots (in terms of distance)
        infer_persona (bool): whether to infer persona for each shot
        persona_inference_model (str): which model to generate persona given retrieved shots
        splits: (list[str]): which splits to generate retrieval for (if only eval, test, need train as well if training on it)
    """
    import bm25s
    out_dir = f"{data_dir}_xyw_{top_k}s_{method}{'_reverse' if reverse else ''}-name-prefixed"
    os.makedirs(out_dir, exist_ok=True)
    model_str = MODEL_TO_SHORT[persona_inference_model]
    out_dir_persona = f"{data_dir}_persona_xy_{top_k}s{model_str}_{method}{'_reverse' if reverse else ''}-name-prefixed"
    os.makedirs(out_dir_persona, exist_ok=True)
    all_train_df = pd.read_json(f"{data_dir}/train_prefs.json", lines=True, orient="records")
    common_test_df = pd.read_json(f"{data_dir}/test_common_prefs.json", lines=True, orient="records")
    personal_test_df = pd.read_json(f"{data_dir}/test_personal_prefs.json", lines=True, orient="records")
    if "train" in splits:
        common_train_df = all_train_df[all_train_df["question_type"]=="common"]
        personal_train_df = all_train_df[all_train_df["question_type"]=="personal"]

    # add some features from raw data back to make it work with prompt formatter
    all_train_df["yw"] = all_train_df.chosen.apply(lambda x: x[1]["content"])
    all_train_df["yl"] = all_train_df.rejected.apply(lambda x: x[1]["content"])
    for name in tqdm(common_test_df.name.unique()):
        name_flat = name.replace(" ", "").lower()

        # create retriever
        corpus = all_train_df[all_train_df.name == name].prompt.tolist()
        if method == "bm25":
            retriever = bm25s.BM25(corpus=corpus)
            retriever.index(bm25s.tokenize(corpus))
        elif method == "emb":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sent_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device) #"sentence-transformers/all-MiniLM-L6-v2" "sentence-transformers/sentence-t5-xxl"
            embs = sent_model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)
            embs = (embs / torch.sqrt((embs ** 2).sum(axis=1))[:, None]).cpu().numpy()
        else:
            raise ValueError(f"method {method} not supported")

        df_splits = {}
        if "test" in splits:
            df_splits.update({("test","personal"): (personal_test_df, personal_test_df.copy()), ("test","common"): (common_test_df,common_test_df.copy())})
        if "train" in splits:
            df_splits.update({("train","personal"): (personal_train_df, personal_train_df.copy()), ("train","common"): (common_train_df,common_train_df.copy())})
        for (split, qtype), (ret_df, ret_df_persona) in df_splits.items():
            user_ret_df = ret_df[ret_df.name==name]
            user_ret_df_persona = ret_df_persona[ret_df_persona.name==name]
            shot_indices_list = []
            for i, ret_row in user_ret_df.iterrows():
                # retrieve and get the index
                if method == "bm25":
                    if reverse:
                        docs, scores = retriever.retrieve(bm25s.tokenize(ret_row.prompt), k=len(corpus))
                        docs, scores = docs[:,::-1][:,:top_k], scores[:,::-1][:,:top_k]
                    else:
                        if split == "test":
                            docs, scores = retriever.retrieve(bm25s.tokenize(ret_row.prompt), k=top_k)
                        else:
                            # if retrieval for test, top similar one will be data itself, so move one down
                            docs, scores = retriever.retrieve(bm25s.tokenize(ret_row.prompt), k=top_k+1)
                            docs, scores = docs[:, 1:], scores[:, 1:]
                    topk_indices = [corpus.index(d) for d in docs[0]]
                elif method == "emb":
                    test_emb = sent_model.encode(ret_row.prompt, convert_to_tensor=True, show_progress_bar=True).cpu().numpy()
                    if reverse:
                        topk_indices = torch.flip(np.argsort(sent_model.similarity(embs, test_emb).squeeze()),[0])[-top_k:].tolist()
                    else:
                        if split == "test":
                            topk_indices = np.argsort(sent_model.similarity(embs, test_emb).squeeze())[-top_k:].tolist()
                        else:
                            topk_indices = np.argsort(sent_model.similarity(embs, test_emb).squeeze())[-top_k-1:-1].tolist()
                # format the prompt and replace
                _, fs = format_few_shot_prompt(all_train_df[all_train_df.name == name], top_k, include_yl=False, data_indices=topk_indices[::-1])
                prompt = SAMPLE_YW_GIVEN_XY_PP.format(XY_LIST=fs, x=ret_row.prompt)
                user_ret_df.loc[i, "prompt"] = prompt
                user_ret_df.loc[i, "chosen"][0]["content"] = prompt
                user_ret_df.loc[i, "rejected"][0]["content"] = prompt

                # save shot indices for persona inference later
                shot_indices_list.append(topk_indices[::-1])

            user_ret_df.to_json(f"{out_dir}/{split}_{name_flat}_{qtype}_prefs.json", lines=True, orient="records")

            # infer personas
            if not infer_persona:
                continue
            out_data_path = f"{out_dir_persona}/{split}_{name_flat}_{qtype}_prefs.json"
            if os.path.exists(out_data_path):
                continue
            persona_strs = sample_persona_given_x_y(
                Path(data_dir).parent.joinpath(
                    "train_50p_200d_total_50r_tem2.0_top0.8_cot_filtered20m4k_yl-random_cot_annotated.json"),
                num_shots=top_k,
                model_name=persona_inference_model,
                data_indices={name_flat:shot_indices_list},
                names=[name],
                persona_field_suffix=f"{method}{'_reverse' if reverse else ''}",
                return_single_persona_str=True
            )
            for i, (row_idx, test_row_persona) in enumerate(user_ret_df_persona.iterrows()):
                prompt = SAMPLE_YW_GIVEN_PERSONA_ANONYMOUS_PP.format(PERSONA=persona_strs[i], x=test_row_persona.prompt)
                user_ret_df_persona.loc[row_idx, "prompt"] = prompt
                user_ret_df_persona.loc[row_idx, "chosen"][0]["content"] = prompt
                user_ret_df_persona.loc[row_idx, "rejected"][0]["content"] = prompt

            user_ret_df_persona.to_json(out_data_path, lines=True, orient="records")



def generate_prefix():
    df_persona = pd.read_json(f"{DATA_DIR}/personas.json", lines=True, orient="records")
    df_test = pd.read_json(f"{DATA_DIR}/test_50p_200d_total_50r_tem2.0_top0.8_cot_filtered20m4k_yl-random_cot_annotated.json", lines=True, orient="records")
    df_persona = df_persona[df_persona.name.isin(df_test.name.unique())]
    seeds = [None, 1, 2,3,4]
    models = {
        "HuggingFaceH4/zephyr-7b-beta": "",
        "meta-llama/Llama-3.2-3B-Instruct": "_llama323b",
        "meta-llama/Llama-3.2-1B-Instruct": "_llama321b",
        "mistralai/Ministral-8B-Instruct-2410": "_ministral8b",
    }

    for seed in seeds:
        seed_str = "" if seed is None else f"_seed{seed}"
        # few shot prefix
        if sum(df_persona[f"xyw_2s{seed_str}"] != df_persona[f"xyw_2s{seed_str}"])>0:
            add_xy_fewshot_as_prefix_to_persona(
                train_path=f"{DATA_DIR}/train_50p_200d_total_50r_tem2.0_top0.8_cot_filtered20m4k_yl-random_cot_annotated.json",
                num_shots=2,
                include_yl=False,
                seed=seed
            )
        if f"shuffle_xyw_2s{seed_str}" not in df_persona.columns or sum(df_persona[f"shuffle_xyw_2s{seed_str}"] != df_persona[f"shuffle_xyw_2s{seed_str}"]) > 0:
            add_shuffled_prefices(f"xyw_2s{seed_str}", f"{DATA_DIR}/personas.json")


        if f"shuffle_persona_xy_4s_gpt4{seed_str}" not in df_persona.columns or sum(df_persona[f"shuffle_persona_xy_4s_gpt4{seed_str}"] != df_persona[f"shuffle_persona_xy_4s_gpt4{seed_str}"]) > 0:
            add_shuffled_prefices(f"persona_xy_4s_gpt4{seed_str}", f"{DATA_DIR}/personas.json")

        # deepseek chat persona few shot prefix
        if f"persona_xy_4s_ds{seed_str}" not in df_persona.columns or sum(df_persona[f"persona_xy_4s_ds{seed_str}"] != df_persona[f"persona_xy_4s_ds{seed_str}"])>0:
            sample_persona_given_x_y(
                train_path=f"{DATA_DIR}/train_50p_200d_total_50r_tem2.0_top0.8_cot_filtered20m4k_yl-random_cot_annotated.json",
                num_shots=4, model_name="ds", reveal_name=False,
                seed=seed
            )

        # gpt4 gold prefix
        if sum(df_persona[f"persona_gold_gpt4{seed_str}"] != df_persona[f"persona_gold_gpt4{seed_str}"])>0:
            sample_persona_given_x_y(
                train_path=f"{DATA_DIR}/train_50p_200d_total_50r_tem2.0_top0.8_cot_filtered20m4k_yl-random_cot_annotated.json",
                num_shots=4, model_name="gpt4", reveal_name=True,
                seed=seed
            )
        if f"shuffle_persona_gold_gpt4{seed_str}" not in df_persona.columns or sum(df_persona[f"shuffle_persona_gold_gpt4{seed_str}"] != df_persona[f"shuffle_persona_gold_gpt4{seed_str}"]) > 0:
            add_shuffled_prefices(f"persona_gold_gpt4{seed_str}", f"{DATA_DIR}/personas.json")

        # zephyr prefix
        if sum(df_persona[f"persona_xy_4s{seed_str}"] != df_persona[f"persona_xy_4s{seed_str}"]) > 0:
            sample_persona_given_x_y(
                train_path=f"{DATA_DIR}/train_50p_200d_total_50r_tem2.0_top0.8_cot_filtered20m4k_yl-random_cot_annotated.json",
                num_shots=4, model_name="gpt4", reveal_name=True,
                seed=seed
            )
        if f"shuffle_persona_gold_gpt4{seed_str}" not in df_persona.columns or sum(df_persona[f"shuffle_persona_gold_gpt4{seed_str}"] != df_persona[f"shuffle_persona_gold_gpt4{seed_str}"]) > 0:
            add_shuffled_prefices(f"persona_gold_gpt4{seed_str}", f"{DATA_DIR}/personas.json")

        # zephyr (or other models) prefix
        for model, model_short in models.items():
            if f"persona_xy_4s{model_short}{seed_str}" not in df_persona.columns or sum(df_persona[f"persona_xy_4s{model_short}{seed_str}"] != df_persona[f"persona_xy_4s{model_short}{seed_str}"]) > 0:
                sample_persona_given_x_y(
                    train_path=f"{DATA_DIR}/train_50p_200d_total_50r_tem2.0_top0.8_cot_filtered20m4k_yl-random_cot_annotated.json",
                    num_shots=4, model_name=model, reveal_name=False,
                    seed=seed, model_name_str=model_short,
                )
            if f"shuffle_persona_xy_4s{model_short}{seed_str}" not in df_persona.columns or sum(df_persona[f"shuffle_persona_xy_4s{model_short}{seed_str}"] != df_persona[f"shuffle_persona_xy_4s{model_short}{seed_str}"]) > 0:
                add_shuffled_prefices(f"persona_xy_4s{model_short}{seed_str}", f"{DATA_DIR}/personas.json")


def split_into_train_eval_files():
    data_name = "50p_200d_total_50r_tem2.0_top0.8_cot_filtered20m4k_yl-random_cot_annotated"
    for prefix in [
        False, "name", "tag",
        "xyw_2s", "xyw_4s",
        "persona_xy_4s",
        "persona_xy_4s_llama321b", "persona_xy_4s_llama323b",
        "persona_xy_4s_ministral8b",
        "persona_xy_4s_gpt4", "persona_gold_gpt4"
        "xyw_2s_seed1", "persona_xy_4s_seed1"
        "xyw_2s_seed2", "persona_xy_4s_seed2",
        "xyw_2s_seed3", "persona_xy_4s_seed3",
        "xyw_2s_seed4", "persona_xy_4s_seed4",
    ]:
        for split in ["train", "test"]:
            format_dataset_factory(
                f"{DATA_DIR}/{split}_{data_name}.json",
                yl="self",
                format="prefs",
                prefix_prompt_w_persona=prefix,
                split_question_type=True
            )
        data_suffix = "" if prefix == False else f"_{prefix}-name-prefixed"
        ### Split into individual train test files (1 model/person, and 1 model for all) ###
        split_into_individual_json(f"{DATA_DIR}/{data_name}_self-yl{data_suffix}")  # _tag-name-prefixed
        ### Split into meta train, and individual test_files (5 fold cv) ###
        split_into_cross_validation_for_meta_learning(f"{DATA_DIR}/{data_name}_self-yl{data_suffix}", subset=False)
        ### Split into meta train, and individual test_files (leave-on-axis-out) ###
        split_into_cross_validation_for_meta_learning(f"{DATA_DIR}/{data_name}_self-yl{data_suffix}", subset=False, folds="axis")




def argparser():
    """
    useful for submitting batch jobs for sampling completions
    """
    args = argparse.ArgumentParser()
    args.add_argument(
        "--start", type=int, help="start index for generating completion", default=0
    )
    args.add_argument(
        "--num_completions", type=int, help="number of completion", default=200
    )
    args.add_argument(
        "--reward_model", type=str, help="reward model", default=""
    )
    args.add_argument(
        "--num_to_keep", type=int, help="number to keep", default=30
    )
    args.add_argument(
        "--data_path", type=str, help="", default=""
    )
    args.add_argument(
        "--gold_categories", action="store_true", help=""
    )
    return args




if __name__ == "__main__":
    random.seed(24)
    np.random.seed(24)
    args = argparser().parse_args()
    ###### Sample personas #######
    generate_axis_specific_wiki_persona()

    ###### Sample prompts (x) for each persona #######
    generate_dialogues_from_wiki_people_with_axis(n_responses=100)
    generate_common_dialogues_from_wiki_people_with_axis(n_responses=100)

    ###### Sample responses (y) for each prompts ######
    ## Divergent Questions
    generate_automodel_completion_diverse_cot_dataset(
        "{dataset_dir}/all_50p_200d_common.json",
        "HuggingFaceH4/zephyr-7b-beta",
        temperature=2.0, top_p=0.8, number_of_cot=5, n_responses=50, new_cot_prompt=False,
        gold_categories=True, save_every_n_batch=10,
        start_index=args.start,
        number_of_prompts_to_complete=args.num_completions
    )
    ## Personal Questions
    generate_automodel_completion_diverse_cot_dataset(
        "{dataset_dir}/all_50p_200d_common.json",
        "HuggingFaceH4/zephyr-7b-beta",
        temperature=2.0, top_p=0.8, number_of_cot=5, n_responses=50, new_cot_prompt=False,
        gold_categories=False, save_every_n_batch=10,
        start_index=args.start,
        number_of_prompts_to_complete=args.num_completions
    )

    ###### Filter responses ######
    ## Divergent questions
    filter_diverse_responses(
        f"{DATA_DIR}/all_50a_100d_common_50r_tem2.0_top0.8_cot_gold",
        n_responses=4,
        filter_generic_reward_model="sfairXC/FsfairX-LLaMA3-RM-v0.1",
        num_to_keep=20
    )
    ## Personal Questions
    filter_diverse_responses(
        f"{DATA_DIR}/all_50a_100d_50r_tem2.0_top0.8_cot",
        n_responses=4,
        filter_generic_reward_model="sfairXC/FsfairX-LLaMA3-RM-v0.1",
        num_to_keep=20
    )

    ##### Aggregate from batch jobs (depending on your script) ######
    aggregate_automodel_completion_dataset(
        f"{DATA_DIR}/all_50a_100d_common_50r_tem2.0_top0.8_cot_gold.json", start=0, postfix="_filtered20m4k.json",
        num_indices=200, num_files=5, delete=False
    )
    aggregate_automodel_completion_dataset(
        f"{DATA_DIR}/all_50a_100d_50r_tem2.0_top0.8_cot_gold.json", start=0, postfix="_filtered20m4k.json",
        num_indices=200, num_files=5, delete=False
    )

    ###### distribute divergent completions to individual person ####
    df = pd.read_json(f"{DATA_DIR}/all_50a_100d_common_50r_tem2.0_top0.8_cot_gold_filtered20m4k.json", lines=True, orient="records")
    df = distribute_common_questions(df)
    df.to_json(f"{DATA_DIR}/all_50a_100d_common_50r_tem2.0_top0.8_cot_gold_filtered20m4k_distributed.json", lines=True, orient="records")

    ##### Annotate Questions (obtaining preference label from GPT4) #######
    from openai_annotate import rank_generations
    rank_generations(f"{DATA_DIR}/all_50a_100d_common_50r_tem2.0_top0.8_cot_gold_filtered20m4k_distributed.json",simulate_personal_preference=True, yl_strategy="random")
    rank_generations(f"{DATA_DIR}/all_50p_100d_50r_tem2.0_top0.8_cot_filtered20m4k.json",simulate_personal_preference=True, yl_strategy="random")

    #### Generate personal prefixes ####
    generate_prefix()

    #manually split people into Cross validation so each split has 40:10 train:test

    ### format and split into train and test ###
    split_into_train_eval_files()