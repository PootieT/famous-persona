import os
import json
import pdb
import random
from pprint import pprint
from glob import glob
from typing import Optional, Dict, Any, List
import glob

import scipy
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
from sklearn.preprocessing import normalize
import matplotlib

from rlhf_exp.analysis.get_dataset_statistics import persona_quality, add_demographics_info, \
    DEMOGRAPHIC_ATTRIBUTE_CATEGORIES

try:
    matplotlib.use('TkAgg')
except:
    print("better not be using pycharm!")
import matplotlib.pyplot as plt

sns.set_theme(style="ticks")
PALETTE = "vlag"


def read_all_test_data():
    all_dfs = {}
    data_dir = "../data/pp-50-final/personas.json/50p_200d_total_50r_tem2.0_top0.8_cot_filtered20m4k_yl-random_cot_annotated_self-yl"
    for p in os.listdir(data_dir):
        if not p.startswith("test_"):
            continue
        name = p.split("_")[1]
        q_type = p.split("_")[2]
        df = pd.read_json(f"{data_dir}/{p}", lines=True, orient="records")
        all_dfs[(name, q_type)] = df

    return all_dfs


def aggregate_full_output_files(dump_dir):
    all_res = []
    test_dfs = read_all_test_data()
    for eval_path in tqdm(glob.glob(f"{dump_dir}/one_model_for_all_cv/*/*/*/*scores_dict.json")):
        file_name = eval_path.split("/")[-1]
        name = file_name.split("_")[5]
        q_type = file_name.split("_")[6]
        res_df = pd.read_json(eval_path, lines=True, orient="records")
        test_df = test_dfs[(name, q_type)]
        assert len(test_df) == len(res_df)
        # if "name" in res_df:
        #     del res_df["name"]
        # res_df["conversation_id"] = test_df["conversation_id"]
        res_df.rename(columns={"results": "accuracy"}, inplace=True)
        if "ref_sft" in eval_path:
            res_df.drop(columns=["subset"], inplace=True)
            res_df["ref_free"] = False
        else:
            res_df.drop(columns=["subset", "pol_chosen_score", "pol_rejected_score", "ref_chosen_score", "ref_rejected_score"], inplace=True)
            res_df["ref_free"] = True
        group_dir, exp_dir = eval_path.split("/")[-5], eval_path.split("/")[-4]
        res_df["specific_method"] = exp_dir.replace("overton_", "").rsplit("_",1)[0]
        res_df["group"] = group_dir
        res_df["prompt"] = test_df.prompt
        res_df["question_type"] = q_type
        res_df["seen_during_train"] = "eval_OOD" in eval_path
        all_res.append(res_df)
    all_res_df = pd.concat(all_res).reset_index().drop(columns=["index"])
    all_res_df.to_csv(f"{dump_dir}/aggregated_results_detailed.csv", index=False)


def aggregate_harness_results(dump_dir, row):
    paths = glob.glob(f"{dump_dir}/llm_harness_results/*/results*.json")
    if paths:
        with open(paths[0]) as f:
            res = json.load(f)

        reasoning_results = []
        factual_results = []
        for eval_name, eval_res in res["results"].items():
            if eval_name == "truthfulqa_gen":
                continue
            row[eval_name] = eval_res["acc,none"]
            if "truthfulqa" in eval_name:
                factual_results.append(eval_res["acc,none"])
            else:
                reasoning_results.append(eval_res["acc,none"])
        row["reasoning_harness"] = np.mean(reasoning_results)
        row["factual_harness"] = np.mean(factual_results)


def aggregate_eval_files(dump_dir: str, name: str, row: Dict[str, Any], ref_free_metric: str = ""):
    row["eval_path"] = dump_dir
    _ref_free_metric = "" if not ref_free_metric else f"_{ref_free_metric}"
    eval_files = [
        f"all_results_ref_free{_ref_free_metric}_safety.json",
        f"all_results_ref_free{_ref_free_metric}_reasoning.json",
        f"all_results_ref_free{_ref_free_metric}_test_prefs.json",
        f"all_results_ref_free{_ref_free_metric}_test_{name}_common_prefs.json",
        f"all_results_ref_free{_ref_free_metric}_test_{name}_personal_prefs.json",
        f"all_results_ref_sft_reasoning.json",
        f"all_results_ref_sft_safety.json",
        f"all_results_ref_sft_test_{name}_common_prefs.json",
        f"all_results_ref_sft_test_{name}_personal_prefs.json",
    ]

    metric_fields = ["hep-python", "math-prm", "Reasoning", "Safety", f"test_{name}_common_prefs",
                     f"test_{name}_personal_prefs", "test_prefs"]
    for field in metric_fields:
        for ref in ["free", "dpo"]:
            field_name = f"ref-{ref}_{field}".replace(f"test_{name}_", "").lower()
            if field_name not in row:
                row[field_name] = 0

    for eval_f in eval_files:
        eval_path = f"{dump_dir}/{eval_f}"
        if not os.path.exists(eval_path):
            # if not ref_free_metric:
            #     print(f"{eval_path} not found, skipping")
            continue
        ref = "ref-free" if "free" in eval_f else "ref-dpo"
        try:
            with open(eval_path, "r") as f_io:
                eval_result = json.load(f_io)
        except Exception as e:
            print(f"parsing error on {eval_path}: {e}")
            continue

        if ref_free_metric:
            row["ref_free_agg_method"] = ref_free_metric
        for metric_field in metric_fields:
            if metric_field in eval_result:
                recorded_field = f"{ref}_{metric_field}".replace(f"test_{name}_", "").lower()
                row[recorded_field] = max(row[recorded_field], eval_result[metric_field])
        if f"test_{name}_common_prefs" in eval_result:
            row["common_size"] = eval_result["dataset_size"]
        if f"test_{name}_personal_prefs" in eval_result:
            row["personal_size"] = eval_result["dataset_size"]

    for ref in ["ref-free", "ref-dpo"]:
        if "common_size" in row and "personal_size" in row:
            row[f"{ref}_combined_pref"] = (row[f"{ref}_common_prefs"] * row["common_size"] +
                                           row[f"{ref}_personal_prefs"] * row["personal_size"]) / \
                                          (row["common_size"] + row["personal_size"])
        elif f"{ref}_test_prefs" in row:
            row[f"{ref}_combined_pref"] = row[f"{ref}_test_prefs"]
            del row[f"{ref}_test_prefs"]

    # add llm harness results
    aggregate_harness_results(dump_dir, row)


def aggregate_eval_files_from_default_eval(dump_dir: str, og_row: Dict[str, Any], oqa: bool = False) -> List[Dict[str, Any]]:
    row_dict = {}  # name mapped to results

    acc_pf = "eval_rewards/accuracies_eval" if oqa else "eval_rewards/accuracies_test"
    acc_ood_pf = "eval_rewards/accuracies_OOD_eval" if oqa else "eval_rewards/accuracies_OOD_test"
    eval_files = [f"{dump_dir}/eval_results.json", f"{dump_dir}/eval_ref_free_results.json"]
    for ref, eval_file in zip(["ref-dpo", "ref-free"], eval_files):
        with open(eval_file) as f:
            res = json.load(f)

        names = list(set([k.split("_")[-1] if oqa else k.split("_")[-2] for k in res.keys() if
                          (k.startswith(acc_pf) or k.startswith(acc_ood_pf))]))
        for name in names:
            if name not in row_dict:
                row_dict[name] = og_row.copy()
            if oqa:
                if f"{acc_pf}_{name}" in res:
                    pf, seen_during_train = acc_pf, True
                else:
                    pf, seen_during_train = acc_ood_pf, False
            else:
                if f"{acc_pf}_{name}_common" in res:
                    pf, seen_during_train = acc_pf, True
                else:
                    pf, seen_during_train = acc_ood_pf, False

            row_dict[name].update({
                "common_size": res["dataset_size"], "personal_size": res["dataset_size"],
                "seen_during_train": seen_during_train
            })
            if oqa:
                row_dict[name].update({
                    f"{ref}_common_prefs": res[f"{pf}_{name}"],
                    f"{ref}_combined_pref": res[f"{pf}_{name}"],
                })
            else:
                row_dict[name].update({
                    f"{ref}_common_prefs": res[f"{pf}_{name}_common"],
                    f"{ref}_personal_prefs": res[f"{pf}_{name}_personal"],
                    f"{ref}_combined_pref": (res[f"{pf}_{name}_common"] + res[f"{pf}_{name}_personal"]) / 2,
                })

        total_rows = [{"name": n, "eval_path": dump_dir, **kvs} for n, kvs in row_dict.items()]
    return total_rows


def get_prompt_results(dump_dir: str):
    """
    directory here follows pattern {prompt}_{name}
    within each directory, we have 6 result files
    - all_results_ref_free_reasoning.json
    - all_results_ref_free_test_{name}_common_prefs.json
    - all_results_ref_free_test_{name}_personal_prefs.json
    - all_results_ref_sft_reasoning.json
    - all_results_ref_sft_test_{name}_common_prefs.json
    - all_results_ref_sft_test_{name}_personal_prefs.json
    """
    print(f"collecting prompting results {dump_dir}")

    def get_name(f: str):
        return f.split("_")[-1]

    rows = []
    names = set([get_name(f) for f in os.listdir(dump_dir)])
    for name in tqdm(names):
        name_fs = [f for f in os.listdir(dump_dir) if name in f]

        for f in name_fs:
            name = get_name(f)
            prompt = "_".join(f.split("_")[:-1])
            prompt = "no_prefix" if prompt == "baseline" else prompt
            if "pp_finetuned_" in prompt:
                prompt = prompt.replace("pp_finetuned_", "")
                row = {"method_category": "prompt", "specific_method": prompt, "name": name, "finetuned": "pp"}
            else:
                row = {"method_category": "prompt", "specific_method": prompt, "name": name}
            aggregate_eval_files(f"{dump_dir}/{f}", name, row, ref_free_metric="avg")
            if row["ref-free_combined_pref"] == 0:
                aggregate_eval_files(f"{dump_dir}/{f}", name, row)
            rows.append(row)
            for ref_free_agg_method in ["sum"]: # "avg", "sum", "norm"
                if os.path.exists(f"{dump_dir}/{f}/all_results_ref_free_{ref_free_agg_method}_reasoning.json"):
                    row = {"method_category": "prompt", "specific_method": prompt, "name": name}
                    aggregate_eval_files(f"{dump_dir}/{f}", name, row, ref_free_metric=ref_free_agg_method)
                    rows.append(row)

    df = pd.DataFrame(rows)
    return df


def get_per_person_results(dump_dir: str):
    print(f"collecting personal model results in {dump_dir.split('/')[-1]}")
    rows = []
    for exp_dir in tqdm(os.listdir(dump_dir)):
        if "_" in exp_dir:
            name, prefix = exp_dir.split("_")[0], "_".join(exp_dir.split("_")[1:])
        else:
            name, prefix = exp_dir, "no_prefix"
        # prefix = prefix.split("-")[0]
        row = {"method_category": "personal_model", "specific_method": prefix, "name": name, "seen_during_train": True}
        if "seed" in prefix:
            row["seed"] = int(prefix[prefix.find("seed") + 4:])
            prefix = prefix.replace(f"_seed{row['seed']}", "")
            row["specific_method"] = prefix
        aggregate_eval_files(f"{dump_dir}/{exp_dir}", name, row, "avg")
        rows.append(row)

        # if OOD eval found
        ood_eval_dir = f"{dump_dir}/{exp_dir}/eval_OOD"
        if os.path.isdir(ood_eval_dir):
            for test_name in os.listdir(ood_eval_dir):
                row = {"method_category": "personal_model", "specific_method": prefix, "name": test_name,
                       "seen_during_train": False, "trained_on": name}
                aggregate_eval_files(f"{ood_eval_dir}/{test_name}", test_name, row, "avg")
                rows.append(row)
    df = pd.DataFrame(rows)
    return df


def get_multitask_results(dump_dir: str, from_single_file: bool=False):
    print(f"collecting multitask model results {dump_dir}")

    if from_single_file and os.path.isfile(f"{dump_dir}/{os.listdir(dump_dir)[0]}/eval_ref_free_results.json"):
        return get_single_eval_file_results(dump_dir)

    rows = []
    for exp_dir in tqdm(os.listdir(dump_dir)):
        prefix = "_".join(exp_dir.split("_")[1:])
        prefix = f"no_prefix_{exp_dir[exp_dir.find('_cv')+1:]}" if "no_prefix" in prefix else prefix

        for eval_dir in glob.glob(f"{dump_dir}/{exp_dir}/*eval*"):
            if not os.path.isdir(eval_dir):
                continue
            seen_during_training = False if "_OOD" in eval_dir else True
            row = {"method_category": "multitask_model", "specific_method": prefix,
                   "seen_during_train": seen_during_training}
            dir_name = eval_dir.split("/")[-1]
            if dir_name.startswith("altprefix"):
                # eval on a prefix not seen during training
                alt_prefix = dir_name.replace("_eval_OOD", "").replace("_eval", "").replace("altprefix_", "")
                row["alt_prefix"] = alt_prefix
            elif not dir_name.startswith("eval") and "agg" in eval_dir:
                # multi-prefix tuning, eval on a prefix seen during train
                row["specific_method"] = dir_name.replace("_eval_OOD", "").replace("_eval", "") + \
                                         exp_dir[exp_dir.find("_cv"):]
                row["multi_prefix"] = int(exp_dir[exp_dir.find("agg") + 3: exp_dir.find("_", exp_dir.find("agg"))])
            # otherwise, the eval dir is named eval or eval_OOD, which is for single prefix train, eval on that prefix

            for name in os.listdir(eval_dir):
                for agg_method in [None, "sum", "avg"]:
                    if not any(
                            f.startswith(f"all_results_ref_free{'_' + agg_method if agg_method is not None else ''}_")
                            for f in os.listdir(f"{eval_dir}/{name}")):
                        continue
                    name_row = row.copy()
                    name_row["name"] = name
                    aggregate_eval_files(f"{eval_dir}/{name}", name, name_row, ref_free_metric=agg_method)
                    if name_row is not None:
                        rows.append(name_row)
                    rows.append(name_row)

    df = pd.DataFrame(rows)
    return df


def get_single_eval_file_results(dump_dir: str):
    """For VPL or opinionQA results"""
    print("collecting aggregated eval results from single file")
    rows = []
    for exp_dir in tqdm(os.listdir(dump_dir)):
        prefix = f"vpl_{exp_dir.split('_')[-1]}" if "vpl" in exp_dir else "_".join(exp_dir.split("_")[1:])
        # prefix = "no_prefix" if "no_prefix" in prefix else "vpl" if "vpl" in prefix else prefix
        id_eval_dir = f"{dump_dir}/{exp_dir}"
        row = {"method_category": "multitask_model", "specific_method": prefix, "seen_during_train": True}
        rows.extend(aggregate_eval_files_from_default_eval(id_eval_dir, row, oqa="oqa" in exp_dir))
    df = pd.DataFrame(rows)
    return df


def aggregate(dump_dir: str):
    """
    dump directory which should include three subdirs:
    - one_model_per_person, prompt, one_model_for_all

    In aggregated table, we want columns:
    - {method_category, specific_methode, name, ref_free_code, ref_free_math, ref_free_reason, ref_free_personal, ref_free_common, .. (and ref_sft versions)}
    every row
    """
    method_dirs = os.listdir(dump_dir)
    df = pd.DataFrame()

    prompt_dirs = [d for d in method_dirs if d.startswith("prompt")]
    for prompt_dir in prompt_dirs:
        prompt_df = get_prompt_results(f"{dump_dir}/{prompt_dir}")
        if prompt_dir == "prompt":
            prompt_df["model"] = "Zephyr"
        else:
            prompt_df["model"] = prompt_dir.replace("prompt_", "")
            prompt_df["specific_method"] = prompt_df.apply(lambda r: r.specific_method.replace(f"_{r.model}", ""), 1)
        df = pd.concat([df, prompt_df])
    if "one_model_per_person" in method_dirs:
        df = pd.concat([df, get_per_person_results(f"{dump_dir}/one_model_per_person")])
    # if "one_model_for_all" in method_dirs:
    #     df = pd.concat([df, get_multitask_results(f"{dump_dir}/one_model_for_all")])

    # if "one_model_per_person_scaling" in method_dirs:
    #     exp_df = get_per_person_results(f"{dump_dir}/one_model_per_person_scaling")
    #     # exp_df["seed"] = exp_df["specific_method"].apply(
    #     #     lambda x: 42 if "seed" not in x else int(x[x.find("seed") + 4:]))
    #     exp_df["fraction"] = exp_df["specific_method"].apply(
    #         lambda x: 1 if "frac" not in x else float(x[x.find("frac") + 4:]))
    #     exp_df["specific_method"] = "no_prefix"
    #     df = pd.concat([df, exp_df])

    # if "one_model_per_person_scaling_extra" in method_dirs:
    #     exp_df = get_per_person_results(f"{dump_dir}/one_model_per_person_scaling_extra")
    #     # TODO using it as regular personal model for now
    #     exp_df["seed"] = exp_df["specific_method"].apply(lambda x: int(x[x.find("seed")+4:]))
    #     exp_df["fraction"] = exp_df["specific_method"].apply(lambda x: float(x[x.find("frac")+4:x.find("_seed")]))
    #     exp_df["combo"] = exp_df["specific_method"].apply(lambda x: x[x.find("extra") + 5:x.find("_frac")])
    #     exp_df["specific_method"] = "no_prefix"
    #     df = pd.concat([df, exp_df])

    cv_dirs = [d for d in method_dirs if d.startswith("one_model_for_all_cv") and "_axis" not in d]
    for cv_dir in cv_dirs:
        if len(os.listdir(f"{dump_dir}/{cv_dir}"))==0:
            continue
        exp_df = get_multitask_results(f"{dump_dir}/{cv_dir}", False)
        # exp_df["cv"] = True
        exp_df["model"] = "Zephyr" if (cv_dir == "one_model_for_all_cv") else cv_dir.replace("one_model_for_all_cv_", "")
        exp_df["specific_method"] = exp_df.apply(lambda r: r.specific_method.replace(f"_{r.model}", ""), 1)
        exp_df["cv"] = exp_df["specific_method"].apply(lambda x: int(x[x.find("_cv") + 3:]))
        exp_df["specific_method"] = exp_df["specific_method"].apply(lambda x: x.rsplit("_", 1)[0])
        df = pd.concat([df, exp_df])

    if "one_model_for_all_cv_axis" in method_dirs:
        exp_df = get_multitask_results(f"{dump_dir}/one_model_for_all_cv_axis")
        # exp_df["cv"] = True
        exp_df["model"] = "Zephyr"
        exp_df["cv"] = exp_df["specific_method"].apply(lambda x: x[x.find("_cv_axis") + 8:]) # cv would be axis (e.g. age, diet)
        exp_df["specific_method"] = exp_df["specific_method"].apply(lambda x: x[:x.find("_cv")].replace("overton_", ""))
        df = pd.concat([df, exp_df])

    # if "one_model_for_all_vpl" in method_dirs:
    #     exp_df = get_multitask_results(f"{dump_dir}/one_model_for_all_vpl", True)
    #     df = pd.concat([df, exp_df])

    if "one_model_for_all_vpl_cv" in method_dirs:
        exp_df = get_multitask_results(f"{dump_dir}/one_model_for_all_vpl_cv", True)
        # exp_df["cv"] = True
        exp_df["cv"] = exp_df["specific_method"].apply(lambda x: int(x[x.find("_cv") + 3:]))
        exp_df["specific_method"] = exp_df["specific_method"].apply(lambda x: x.rsplit("_", 1)[0])
        df = pd.concat([df, exp_df])

    # if "hyperparameter_tunning" in method_dirs:
    #     exp_df = get_multitask_results(f"{dump_dir}/hyperparameter_tunning")
    #     # exp_df["cv"] = True
    #     exp_df["cv"] = exp_df["specific_method"].apply(lambda x: int(x[x.find("_cv") + 3:]))
    #     exp_df["inference_persona"] = exp_df["specific_method"].apply(
    #         lambda x: x[x.find("_infer") + 6:x.find("_numinf")])
    #     # exp_df["max_len"] = exp_df["specific_method"].apply(
    #     #     lambda x: int(x[x.find("_max") + 4:x.find("_", x.find("_max") + 1)]))
    #     # exp_df["refresh"] = exp_df["specific_method"].apply(
    #     #     lambda x: np.nan if "refresh" not in x else int(x[x.find("_refresh") + 8:x.find("_", x.find("_refresh") + 1)]))
    #     exp_df["num_persona_inference_per_person"] = exp_df["specific_method"].apply(
    #         lambda x: float(x[x.find("_numinf") + 7:x.find("_", x.find("_numinf") + 1)]))
    #     # exp_df[["specific_method", "max_len", "inference_persona", "num_persona_inference_per_person"]]
    #     exp_df["specific_method"] = exp_df["specific_method"].apply(lambda x: x[:x.find("_infer")])
    #     df = pd.concat([df, exp_df])

    df.loc[df.ref_free_agg_method.isna(), "ref_free_agg_method"] = "avg"
    if "common_size" in df.columns and ("personas.json" not in dump_dir):
        df = df[(~df.common_size.isna()) | (df.method_category=="personal_model")]

    df.to_csv(f"{dump_dir}/aggregated_results.csv", index=False)


def remove_specific_experiment_rows(df) -> pd.DataFrame:
    for col in ["fraction", "cv"]:
        if col in df.columns:
            df = df[df[col].isna()]
    return df


def what_is_the_best_prompt(dump_dir: str):
    """
    RQ1: what's the best prompting method?
    aggregate across people, look at reasoning/combined, ref-free/ref-dpo
    """
    exp_dir = f"{dump_dir.split('/')[-1]}/best_prompt"
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.read_csv(f"{dump_dir}/aggregated_results.csv")

    random_baseline = df[df.method_category=="baseline"]
    df = remove_specific_experiment_rows(df)
    df = df[df.method_category.isin(["prompt"])]
    baseline = df[df.specific_method.isin(["baseline","no_prefix"])]
    df = df[~df.specific_method.isin(["baseline","no_prefix"])]

    # compare different model's prompt performance
    df[df.ref_free_agg_method=="avg"].pivot_table(index=["model", "specific_method"], values=["ref-free_common_prefs", "ref-free_personal_prefs"], aggfunc=["mean", "std"])
    # save the retrieval vs non-retrieval df out
    retrieval_df = df[(df.specific_method.str.contains("bm25")) | (df.specific_method.str.contains("emb")) | (df.specific_method=="xyw_2s")]
    retrieval_df.groupby(["model", "specific_method"])["ref-free_common_prefs"].std()

    baseline = baseline[(baseline["ref-free_hep-python"]!=0)]
    name_upperbound = df[df.specific_method=="name"]
    gpt4_upperbound = df[df.specific_method.str.contains("4s_gpt4")]
    gpt4_gold_upperbound = df[df.specific_method.str.contains("gold_gpt4")]
    df = df[~df.specific_method.str.contains("_gpt4")]
    df["shots"] = df.specific_method.apply(lambda x: 1 if "_" not in x else int(x.replace("_bm25", "").replace("_emb","").split("_")[-1][:-1]))
    df["specific_method"] = df.specific_method.apply(lambda x: "_".join(x.split("_")[:-1]))
    df = df[df.ref_free_agg_method == "avg"]
    df.loc[(df.method_category=="prompt")&(df.specific_method==""), "specific_method"] = "name"
    df = df.drop_duplicates(["eval_path", "name"])
    col_map = {
        "specific_method": "Prefix",
        "ref-free_combined_pref": "Total Accuracy",
        "ref-free_personal_prefs": "Personal Accuracy",
        "ref-free_common_prefs": "Divergent Accuracy"
    }
    baseline_val_map = {
        "persona_gold_gpt4": "persona gold",
        "majority_class": "random",
        "name": "name",
        "no_prefix": "no prefix",
    }
    exp_val_map = {
        "xyw": "few-shot",
        "persona_xy": "persona",
    }
    df.rename(columns=col_map, inplace=True)
    df["Prefix"] = df.Prefix.map(exp_val_map)
    df = df[~df.Prefix.isna()]

    df_baselines = pd.concat([random_baseline, baseline, name_upperbound, gpt4_gold_upperbound])
    df_baselines.rename(columns=col_map, inplace=True)
    df_baselines["Prefix"] = df_baselines.Prefix.map(baseline_val_map)

    ref="ref-free"
    fig, axs = plt.subplots(2, 2, sharey=True)
    for i, task in enumerate(["personal_prefs", "common_prefs"]): #"combined_pref", #    "reasoning", "combined_pref",

        # first we plot the baselines
        sns.violinplot(
            df_baselines, x="Prefix", y=col_map[f"{ref}_{task}"], ax=axs[i,0],
            order=["no prefix", "name", "persona gold"], # random <- moving it to dashed line
            palette=PALETTE
        )
        axs[i,0].axhline(y=df_baselines[df_baselines.Prefix=="random"][col_map[f"{ref}_{task}"]].mean(), color="black", ls="--")

        # then we plot the few shot results
        print(df[["Prefix","shots"]].value_counts())
        sns.violinplot(df, x="shots", y=col_map[f"{ref}_{task}"], hue="Prefix", ax=axs[i,1], palette=PALETTE)
        axs[i, 1].axhline(y=df_baselines[df_baselines.Prefix == "random"][col_map[f"{ref}_{task}"]].mean(), color="black", ls="--")

        # ax.axhline(y=baseline[col_map[f"{ref}_{task}"]].tolist()[0], linewidth=2, color='orange', ls=':')
        # ax.axhline(y=name_upperbound[col_map[f"{ref}_{task}"]].mean(), linewidth=2, color='red', ls=':')
        # ax.axhline(y=gpt4_upperbound[col_map[f"{ref}_{task}"]].mean(), linewidth=2, color='blue', ls=':')
        # ax.axhline(y=gpt4_gold_upperbound[col_map[f"{ref}_{task}"]].mean(), linewidth=2, color='green', ls=':')
        if i == 0:
            axs[i,1].set_title("few-shot and persona prompts")
            # axs[1].set(xscale='log')
            # axs[1].set(xticks=[1, 4, 8, 16, 30])
            # axs[1].set(xticklabels=[1, 4, 8, 16, 30])
    axs[0, 0].set_title("baseline and upperbounds")
    axs[0, 0].set_xticks([])
    axs[0, 0].set_xlabel("")
    # axs[1, 0].tick_params(axis='x', rotation=15)

    axs[0, 1].set_xticks([])
    axs[0, 1].set_xlabel("")
    axs[1, 1].get_legend().remove()
    plt.tight_layout()
    # plt.savefig(f"{exp_dir}/paper_prompt_performance_combined.png")
    plt.savefig(f"{exp_dir}/paper_prompt_performance_combined.pdf", dpi=600)
    plt.close()

    # plt.figure()
    # sns.lineplot(data=df, x="shots", y=f"reasoning_harness", hue="Prefix")
    # plt.savefig(f"{exp_dir}/prompt_performance_reasoning_harness.png")
    # plt.close()



def prompt_upperbounds(dump_dir: str):
    """
    RQ1: What is the upperbound we can get on prompting by revealing the name of the person (these benefits
    would not be applicable to regular humans)
    aggregate across people, look at reasoning/combined, ref-free/ref-dpo
    """
    exp_dir = f"{dump_dir.split('/')[-1]}/prompt_upperbounds"
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.read_csv(f"{dump_dir}/aggregated_results.csv")
    df = remove_specific_experiment_rows(df)
    df = df[df.method_category == "prompt"]
    # baseline = df[df.specific_method.isin(["baseline","no_prefix"])]
    # df = df[~df.specific_method.isin(["baseline","no_prefix"])]
    # baseline = baseline[(baseline["ref-free_hep-python"]!=0)]
    df = df[(df.specific_method.str.contains("gpt4")) | (df.specific_method == "name")]
    df = df[(df["ref-dpo_hep-python"] != 0) & (df["ref-free_hep-python"] != 0)]  # those would
    for ref in ["ref-free", "ref-dpo"]:
        for task in ["reasoning", "combined_pref"]:
            plt.figure()
            ax = sns.barplot(df, x="specific_method", y=f"{ref}_{task}")
            plt.savefig(f"{exp_dir}/prompt_performance_{ref}_{task}.png")
            plt.close()


def personal_model_performance_vs_data_size(dump_dir: str):
    """compare persona performance given amount of their training data"""
    exp_dir = f"{dump_dir.split('/')[-1]}/performance_vs_data_size"
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.read_csv(f"{dump_dir}/aggregated_results.csv")
    df = remove_specific_experiment_rows(df)
    df["training_size"] = df.common_size / 50 * 10 + df.personal_size / 50 * 20 + df.name.isin(
        ["donaldtrump", "halleberry"]) * 70
    for method_category in ["multitask_model", "personal_model"]:
        df_sub = df[df.method_category == method_category]
        for ref in ["ref-free", "ref-dpo"]:
            plt.figure()
            cols = ["training_size", "name", f"{ref}_reasoning", f"{ref}_personal_prefs", f"{ref}_common_prefs"]
            df_melt = df_sub[cols].melt(id_vars=["training_size", "name"],
                                        var_name="tasks",
                                        value_name="performance (accuracy)")
            sns.lmplot(data=df_melt, x="training_size", y=f"performance (accuracy)", hue="tasks")
            plt.savefig(f"{exp_dir}/{method_category}_agg-specific-method_{ref}.png")
            plt.close()

            for task in ["reasoning", "combined_pref"]:
                plt.figure()
                sns.lmplot(data=df_sub, x="training_size", y=f"{ref}_{task}", hue="specific_method")
                plt.savefig(f"{exp_dir}/{method_category}_{ref}_{task}.png")
                plt.close()


def personal_model_performance(dump_dir: str):
    """compare persona performance"""
    exp_dir = f"{dump_dir.split('/')[-1]}/personal_model_performance"
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.read_csv(f"{dump_dir}/aggregated_results.csv")
    # df = remove_specific_experiment_rows(df)
    df_baseline = df[(df.specific_method=="no_prefix") & (df.method_category=="prompt") & (df.ref_free_agg_method=="avg")]
    df_prompt_gold = df[
        (df.specific_method == "persona_gold_gpt4") & (df.method_category == "prompt") & (df.ref_free_agg_method == "avg")]
    df = df[df.method_category == "personal_model"]
    df = df.drop_duplicates(["eval_path", "name"])

    df, col_map, _ = rename_df_for_plotting(df)
    df_baseline, _, _ = rename_df_for_plotting(df_baseline)
    df_prompt_gold, _, _ = rename_df_for_plotting(df_prompt_gold)

    ref="ref-free"
    metrics=["Total Accuracy"]#,"Personal Accuracy","Common Accuracy"]
    for m in metrics:
        df_self = df[df.seen_during_train==True]
        # violin plot performance side by side as aggregate
        # plt.figure()
        # sns.violinplot(data=pd.concat([df_baseline, df_self]), x="method_category", y=m)
        # plt.savefig(f"{exp_dir}/personal_model_{m}.png")
        # plt.close()

        # correlating between no finetune performance vs finetune performance
        # plt.figure()
        # df_join = pd.merge(df_self, df_baseline[["name",*metrics]], on="name", suffixes=[" (PM)", " (Zephyr no prefix)"])
        # sns.lmplot(data=df_join, x=f"{m} (Zephyr no prefix)", y=f"{m} (PM)", palette=PALETTE,
        #            height=3.2, aspect=1)
        # plt.savefig(f"{exp_dir}/personal_model_{m}_correlation.png")
        # plt.tight_layout()
        # plt.close()

        # correlating between no finetune performance vs finetune performance (on other people)
        # df_other = df[df.seen_during_train == False]
        # plt.figure()
        # df_join = pd.merge(df_other, df_baseline[["name", *metrics]], on="name", suffixes=[" finetuned", " baseline"])
        # sns.lmplot(data=df_join, x=f"{m} baseline", y=f"{m} finetuned")
        # plt.savefig(f"{exp_dir}/personal_model_{m}_correlation_w_other.png")
        # plt.close()

    fig, axs = plt.subplots(2, 1, sharex=True)
    for i, m in enumerate(["Personal Accuracy", "Divergent Accuracy"]):
        # violin plot where x=seen_during_train, hue=prefix
        # df_baseline["seen_during_train"] = "Zephyr"
        # df["seen_during_train"] = df.seen_during_train.apply(lambda s: f"PM ({s})")
        sns.violinplot(
            data=df[df[m]>0], x="seen_during_train", y=m, ax=axs[i],
            hue="Prefix", hue_order=["no prefix", "persona gold"],
            order=["Persona trained", "Persona not trained"],
            palette=PALETTE
        )
        axs[i].axhline(y=df_baseline[m].mean(), linewidth=2, color=sns.color_palette(PALETTE,2)[0], ls='--')
        axs[i].axhline(y=df_prompt_gold[m].mean(), linewidth=2, color=sns.color_palette(PALETTE,2)[1], ls='--')

    # add dash vs var
    handles, labels = axs[0].get_legend_handles_labels()
    handles.extend([
        matplotlib.patches.Patch(facecolor='white', edgecolor='black', hatch='', label='PM'),
        matplotlib.lines.Line2D([],[],color="black", linestyle='--', label='Zephyr'),
    ])
    labels.extend(["PM", "Zephyr"])
    axs[0].legend(handles=handles, labels=labels)

    sns.move_legend(
        axs[0], "lower center", frameon=False,
        bbox_to_anchor=(.5, 1), ncol=4, title="Prefix Types                                   Model",
    )
    axs[1].get_legend().remove()
    plt.xlabel("")
    plt.tight_layout()
    # plt.savefig(f"{exp_dir}/personal_model_both_IDOOD.png")
    plt.savefig(f"{exp_dir}/personal_model_both_IDOOD.pdf",dpi=600)
    plt.close()


def performance_across_political_axis(dump_dir: str):
    exp_dir = f"{dump_dir.split('/')[-1]}/performance_political_axis"
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.read_csv(f"{dump_dir}/aggregated_results.csv")
    # df = remove_specific_experiment_rows(df)
    df = df[df.specific_method == "persona_xy_4s_gpt4"]
    df = df[df.inference_persona.isna()]
    df = df[df.seen_during_train!=True]
    names = ["alexandriaocasio-cortez", "berniesanders", "joebiden", "randpaul","donaldtrump"]
    df = df[df.name.isin(names)]
    for ref in ["ref-free"]:
        for task in ["combined_pref"]:
            agg = "sum" if task in ["reasoning", "safety"] else "avg"
            sub_df = df[df[f"{ref}_{task}"]>0]
            sub_df = sub_df[sub_df.ref_free_agg_method==agg]
            sub_df = sub_df.drop_duplicates(["name", "eval_path"])
            print(sub_df.method_category.value_counts())
            plt.figure()
            sns.barplot(sub_df, x="name", y=f"{ref}_{task}", hue="method_category", order=names)
            plt.savefig(f"{exp_dir}/bar_{ref}_{task}.png")
            plt.close()


def overall_method_performance_comparison(dump_dir: str):
    assert "final" in dump_dir
    exp_dir = f"{dump_dir.split('/')[-1]}/overall_performance_comparison"
    os.makedirs(exp_dir, exist_ok=True)
    df50 = pd.read_csv(f"{dump_dir}/aggregated_results.csv")
    df50 = df50[
        (df50.specific_method.isin(["persona_gold_gpt4", "no_prefix"]))
        & (df50.method_category == "multitask_model") & (df50.ref_free_agg_method=="avg")
        & (df50.alt_prefix.isna()) & (df50['ref-free_combined_pref'] > 0)
        ].drop_duplicates(["name", "eval_path"])

    df10 = pd.read_csv(f"{dump_dir.replace('personas.json', 'pp-dpo')}/aggregated_results.csv")
    df10 = df10[
        (df10.specific_method.isin(["persona_gold_gpt4", "no_prefix"])) &
        (df10.ref_free_agg_method=="avg") & (df10['ref-free_combined_pref'] > 0)
        & (df10.alt_prefix.isna()) & (df10.multi_prefix.isna()) & (df10.inference_persona.isna())
        & (~df10.eval_path.str.contains("/one_model_for_all/"))
        & ((df10.trained_on.isna()) | (df10.method_category=="personal_model"))
        ].drop_duplicates(["name", "eval_path"])
    if "frac" in df10.columns:
        df10 = df10[df10.frac.isna()]

    df50 = df50[df50.name.isin(df10.name.unique())]

    df50, col_map, _ = rename_df_for_plotting(df50)
    df10, col_map, _ = rename_df_for_plotting(df10)
    df10.loc[df10.Model=="multi-task model (MT)", "Model"] += " small"
    df50["Model"] = df50.Model + " all"
    df = pd.concat([df10, df50])

    ref="ref-free"
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12,8))
    for col, trained in enumerate(["Persona trained", "Persona not trained"]):
        for row,task in enumerate(["Personal Accuracy", "Divergent Accuracy"]):  # "combined_pref", "reasoning"
            sns.boxplot(
                df[df.seen_during_train==trained], x="Prefix", y=task, ax=axs[row, col],
                order=["no prefix", "persona gold"],
                hue="Model",
                hue_order=["Zephyr-7B", "personal model (PM)", "multi-task model (MT) small", "multi-task model (MT) all"],
                palette=PALETTE
            )
        axs[1, col].set_xlabel(trained)

    # sns.move_legend(
    #     axs[0,0],
    #     loc="upper center",
    #     bbox_to_anchor=(-0.1, 1.1),
    #     ncol=4,
    #     title="Model Type",
    #     frameon=False,
    # )
    # axs[0,0].autoscale()
    # axs[1,0].autoscale()
    # axs[0, 1].autoscale()
    # axs[1, 1].autoscale()
    # axs[0, 0].get_legend().remove()
    axs[1,0].get_legend().remove()
    axs[0, 1].get_legend().remove()
    axs[1, 1].get_legend().remove()

    # Add a single legend in the middle of the subplots
    # handles, labels = axs[0,0].get_legend_handles_labels()
    # axs[0, 0].get_legend().remove()
    # fig.legend(handles, labels, loc='center', title='Model Type')

    # axs[0, 0].set_ylim(0.4)
    # axs[0, 1].set_ylim(0.4)


    plt.tight_layout()
    plt.savefig(f"{exp_dir}/bar_overall_comparison.png")
    plt.close()

    # plt.figure()
    # sns.barplot(df, x="method_category", y=f"reasoning_harness", hue="specific_method",
    #             order=["prompt", "personal_model", "multitask_model"])
    # plt.savefig(f"{exp_dir}/bar_reasoning_harness.png")
    # plt.close()


def performance_data_scaling(dump_dir: str):
    exp_dir = f"{dump_dir.split('/')[-1]}/performance_data_scaling"
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.read_csv(f"{dump_dir}/aggregated_results.csv")
    df_zephyr_no_prefix = df[
        (df.method_category=="prompt") & (df.specific_method=="no_prefix")
        &(df.ref_free_agg_method=="avg")
    ]
    df_halle_baseline = df_zephyr_no_prefix[df_zephyr_no_prefix.name=="halleberry"]
    df_donald_baseline = df_zephyr_no_prefix[df_zephyr_no_prefix.name == "donaldtrump"]
    df = df[(~df.fraction.isna()) & (~df.common_size.isna())]
    # df["training_size"] = df.common_size / 50 * 10 + df.personal_size / 50 * 20 + df.name.isin(
    #     ["donaldtrump", "halleberry"]) * 70
    df["# training data"] = 100
    df["# training data"] = df["# training data"] * df.fraction
    df, col_map, _ = rename_df_for_plotting(df)
    df_halle_baseline, _, _ = rename_df_for_plotting(df_halle_baseline)
    df_donald_baseline, _, _ = rename_df_for_plotting(df_donald_baseline)
    ref = "ref-free" #, "ref-dpo"
    for task in ["Total Accuracy"]:  # "combined_pref", "common_prefs", "personal_prefs",
        plt.figure()
        sns.lmplot(data=df, x="# training data", y=task, hue="name", palette=PALETTE)
        plt.axhline(y=df_donald_baseline[task].mean(), color=sns.color_palette(PALETTE,2)[0], linestyle='--', linewidth=2)
        plt.axhline(y=df_halle_baseline[task].mean(), color=sns.color_palette(PALETTE,2)[1], linestyle='--',  linewidth=2)
        plt.savefig(f"{exp_dir}/performance_data_scaling_{task}.png")
        plt.close()


def performance_data_scaling_extra(dump_dir: str):
    """results on 100 more question (common/personal/gold/not gold axis)"""
    exp_dir = f"{dump_dir.split('/')[-1]}/performance_data_scaling_extra"
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.read_csv(f"{dump_dir}/aggregated_results.csv")
    df = df[~df.combo.isna()]
    df["# training data"] = 200 + 100 * df.combo.str.contains("-")
    df["# training data"] = df["# training data"] * df.fraction
    df_agg = df[(df.fraction == 1.0)].groupby(["name", "combo"])["ref-dpo_combined_pref"].mean()
    for combo in ["cg-p"]:  # "p", "pg", "cg", , "cg-pg"
        for ref in ["ref-free"]:  # , "ref-dpo"
            for task in ["combined_pref"]:  # ["combined_pref", "reasoning", "personal_prefs", "common_prefs"]:
                df_sub = df[df.combo == combo]
                df_sub.rename(columns={"ref-free_combined_pref": "Total Accuracy"}, inplace=True)
                df_sub["name"] = df_sub["name"].map({"donaldtrump": "Donald Trump", "halleberry": "Halle Berry"})
                plt.figure()
                sns.lmplot(data=df_sub, x="# training data", y="Total Accuracy", hue="name")
                plt.savefig(f"{exp_dir}/performance_data_scaling_{combo}_{ref}_{task}.png")
                plt.close()


def ood_performance_multitask(dump_dir: str, cv="cv", model:Optional[str]="zephyr"):
    exp_dir = f"{dump_dir.split('/')[-1]}/ood_multitask_performance{'' if cv=='cv' else f'_{cv}'}{'' if model=='zephyr' else f'_{model}'}"
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.read_csv(f"{dump_dir}/aggregated_results.csv")
    df_multi_no_prefix = df[df.name == "no_prefix"]
    df_prompt = df[df.method_category=="prompt"]
    df_multi_no_prefix_pref = df[(df.alt_prefix=="no_prefix")&(df.ref_free_agg_method=="avg")&(~df.common_size.isna())].drop_duplicates()
    df = df[~df.cv.isna()]
    if cv == "cv":
        df = df[df.cv.apply(lambda c: not str(c).replace("_","").isalpha())]
    else:
        df = df[df.cv.apply(lambda c: str(c).replace("_","").isalpha())]
    if "model" in df.columns and model is not None:
        df.loc[df.model.isna(), "model"] = "Zephyr"
        df = df[df.model.str.lower() == model.lower()]
    if "inference_persona" in df.columns:
        df = df[df.inference_persona.isna()]
    # for 10 people
    df = df[~df.name.isin(["no_prefix"])]
    if "alt_prefix" in df.columns:
        df = df[df.alt_prefix.isna()]
    if "multi_prefix" in df.columns:
        df = df[df.multi_prefix.isna()]
    # df["specific_method"] = df.apply(lambda r: r.specific_method if r.multi_prefix!=r.multi_prefix else f"multi_prefix_{int(r.multi_prefix)}",1)

    # 2 shot result seems to be best amongst all xyw
    # df = df[df.specific_method.str.startswith("xyw")]
    df = df[~df.specific_method.isin(["xyw_1s", "xyw_8s", "xyw_4s"])]

    df = df[(~df.specific_method.str.endswith("_only")) & (~df.specific_method.str.contains("_max"))]
    if "refresh" in df.columns:
        df = df[(df.refresh.isna()) & (df.max_len.isna()) & (df.inference_persona.isna()) & (df.num_persona_inference_per_person.isna())]
    df = df.drop_duplicates()

    df, col_map, _ = rename_df_for_plotting(df)
    df_prompt, _, _ = rename_df_for_plotting(df_prompt)
    df_multi_no_prefix, _, _ = rename_df_for_plotting(df_multi_no_prefix)
    df_multi_no_prefix["Inference Prefix"] = df_multi_no_prefix["name"] = "no prefix"
    df_multi_no_prefix_pref, _, _ = rename_df_for_plotting(df_multi_no_prefix_pref)
    df_multi_no_prefix_pref["seen_during_train"] = df_multi_no_prefix_pref.seen_during_train + " (no prefix)"
    #
    ref = "ref-free"  # "ref-dpo"
    fig, axs = plt.subplots(2, 1, sharex=True)
    for i,task in enumerate(["personal_prefs", "common_prefs"]): ##"combined_pref",  "reasoning",  "safety"
        agg_method = "avg"
        sub_df = df[df.ref_free_agg_method == agg_method]
        sub_df = sub_df[~sub_df.common_size.isna()]
        sub_df = sub_df[sub_df[col_map[f"{ref}_{task}"]]>0]
        sub_df = sub_df.drop_duplicates(["eval_path","name"])
        print(sub_df.Prefix.value_counts())
        # sub_df = pd.concat([sub_df, df_multi_no_prefix_pref])
        # leave axis out table 1
        # pd.pivot_table(sub_df[sub_df.Prefix == "persona gpt4"], values=col_map[f"{ref}_{task}"], index=['seen_during_train'], columns=['cv'], aggfunc="mean")
        # leave axis out table 2
        # pd.pivot_table(sub_df, values=col_map[f"{ref}_{task}"], index=['seen_during_train'], columns=['Prefix'], aggfunc="mean")
        # different models table 1
        # pd.pivot_table(sub_df, values=col_map[f"{ref}_{task}"], index=['model', 'seen_during_train'], columns=['Prefix'], aggfunc="mean")

        sns.barplot(
            sub_df, x="seen_during_train", y=col_map[f"{ref}_{task}"], hue="Prefix", #palette=PALETTE,
            hue_order=["no prefix", "tag", "vpl", "few-shot", "persona", "persona gpt4", "persona gold"], #"name", "persona_xy_8s_gpt4",
            # hue_order=["no prefix", "tag", "few-shot", "persona", "persona gpt4", "name", "persona gold"],
            # order=["Persona trained", "Persona not trained", "Persona trained (no prefix)", "Persona not trained (no prefix)",],
            order=["Persona trained", "Persona not trained"],
            ax=axs[i],
        )
        for bar_idx, bar in enumerate(axs[i].patches):
            if bar_idx in range(2,8):  # hacky way to make the bars with wrong prefix have hash
                try:
                    bar.set_hatch('//')
                except Exception as e:
                    print(f"weird bar cannot set hatch: {bar}. {e}")
            elif bar_idx in range(8,12):  # hacky way to make the bars with wrong prefix have hash
                try:
                    bar.set_hatch('*')
                except Exception as e:
                    print(f"weird bar cannot set hatch: {bar}. {e}")

        axs[i].set_ylim(0.45)
        prompt_no_prefix_baseline = df_prompt[(df_prompt.Prefix == "no prefix") & (df_prompt.ref_free_agg_method == "avg")][col_map[f"{ref}_{task}"]].mean()
        axs[i].axhline(y=prompt_no_prefix_baseline, linewidth=1, color='black', ls='--')

    # add the hash vs no hash
    handles, labels = axs[0].get_legend_handles_labels()
    handles.extend([
        matplotlib.patches.Patch(facecolor='white', edgecolor='black', hatch='//', label='passive'),
        matplotlib.patches.Patch(facecolor='white', edgecolor='black', hatch='*', label='active')
    ])
    labels.extend([ "passive prefix", "active prefix"])
    axs[0].legend(handles=handles, labels=labels)

    sns.move_legend(
        axs[0], "lower center",
        bbox_to_anchor=(.5, 1), ncol=4, title="Prefix Types", frameon=False,
    )
    axs[1].get_legend().remove()
    plt.xlabel("")
    # plt.xticks(rotation=-15)
    plt.tight_layout()
    # plt.savefig(f"{exp_dir}/bar_MT_common_personal_joint_subplots.png")
    plt.savefig(f"{exp_dir}/bar_MT_common_personal_joint_subplots.pdf", dpi=600)
    plt.close()

    ## now let's plot them individually for the best method
    # names in sports, ai professors, or politics
    subset_names = ['LeBron James', 'Serena Williams', 'David Beckham', 'Tiger Woods', 'Mike Trout', 'Bernie Sanders', 'Donald Trump', 'Rand Paul', 'Alexandria Ocasio-Cortez', 'Joe Biden', 'Timnit Gebru', 'Suchi Saria', 'Yoshua Bengio', 'Latanya Sweeney', 'Sebastian Thrun']
    if_subset_names = True
    for task in ["safety"]: #,"combined_pref", "reasoning"]:
        for prefix in ["persona gpt4"]:
            if task in ["reasoning", "safety"]:
                agg_method = "sum"
                # sub_df = df[df.common_size.isna()]
                sub_df = df
            else:
                agg_method = "avg"
                sub_df = df[~df.common_size.isna()]

            sub_df = sub_df[sub_df.ref_free_agg_method == agg_method]
            sub_df = sub_df[sub_df[col_map[f"ref-free_{task}"]]>0]
            sub_df = sub_df.drop_duplicates(["eval_path", "name"])
            sub_df = sub_df[sub_df.Prefix==prefix]
            # sub_df = sub_df[~sub_df.Prefix.str.startswith("shuffle")]
            sub_df = sub_df[sub_df.seen_during_train == "Persona not trained"]
            if if_subset_names:
                sub_df = sub_df[sub_df.name.isin(subset_names)]
                sub_df.loc[
                    sub_df.name.str.startswith("Alexandria"), "name"] = "A. Ocasio-Cortez"  # shorten name to make plot look nicer
            sub_df["Inference Prefix"] = "personal prefix"
            df_multi_no_prefix["Inference Prefix"] = df_multi_no_prefix["name"] = "no prefix"
            df_multi_no_prefix_baseline = df_multi_no_prefix[(df_multi_no_prefix.ref_free_agg_method == "sum")
                                                             &(df_multi_no_prefix.Prefix==prefix)].drop_duplicates(["name", "eval_path"])

            sub_df = pd.concat([sub_df, df_multi_no_prefix_baseline])
            sub_df = sub_df[~sub_df.name.isna()]
            print(sub_df.name.value_counts())
            hue_order = sub_df.sort_values(col_map[f"ref-free_{task}"]).name
            if if_subset_names:
                plt.figure(figsize=(8, 3))
            else:
                plt.figure(figsize=(20, 3))
            plt.ylim(sub_df[col_map[f"ref-free_{task}"]].min()-0.01, sub_df[col_map[f"ref-free_{task}"]].max()+0.01)
            ax=sns.barplot(
                sub_df,
                x="name", y=col_map[f"ref-free_{task}"],
                order=hue_order,
                palette = PALETTE,
                hue="Inference Prefix"
            )
            if task !="combined_pref":
                prompt_baseline = df_prompt[(df_prompt.Prefix=="no prefix") & (df_prompt.ref_free_agg_method=="sum")][col_map[f"ref-free_{task}"]].max()
                ax.axhline(y=prompt_baseline, linewidth=2, color='black', ls='--')
                plt.ylim(min(sub_df[col_map[f"ref-free_{task}"]].min(),prompt_baseline) - 0.01,
                         sub_df[col_map[f"ref-free_{task}"]].max() + 0.01)

            else:
                plt.ylim(sub_df[col_map[f"ref-free_{task}"]].min() - 0.01,
                         sub_df[col_map[f"ref-free_{task}"]].max() + 0.01)
            if if_subset_names:
                plt.xticks(rotation=-22, fontsize=10)
            else:
                plt.xticks(rotation=-25,fontsize=6)
            plt.legend().remove()
            plt.xlabel("")
            plt.tight_layout()
            # plt.savefig(f"{exp_dir}/bar_{'all' if not if_subset_names else 'subset'}_individuals_{agg_method}_{task}_{prefix}.png")
            plt.savefig(
                f"{exp_dir}/bar_{'all' if not if_subset_names else 'subset'}_individuals_{agg_method}_{task}_{prefix}.pdf", dpi=600)

            plt.close()

    df = df[~df.common_size.isna()]
    df_prompt = df_prompt[(~df_prompt.common_size.isna())]
    # llm harness evals
    for harness_task in ["reasoning_harness", "factual_harness"]:
        prompt_baseline = df_prompt[(df_prompt.Prefix == "no prefix")&(df_prompt.ref_free_agg_method=="avg")][col_map[harness_task]].max()
        # first we plot them aggregate across name
        # plt.figure()
        sub_df = df[df[col_map[harness_task]] > 0]
        sub_df = sub_df.drop_duplicates(["eval_path", "name"])
        df_multi_no_prefix_baseline = df_multi_no_prefix[(df_multi_no_prefix.ref_free_agg_method == "avg")].drop_duplicates(["name", "eval_path"])
        df_multi_no_prefix_baseline["seen_during_train"] = "no prefix at inference time"
        # ax = sns.barplot(
        #     data=pd.concat([sub_df, df_multi_no_prefix_baseline]),
        #     x="seen_during_train", y=col_map[harness_task],
        #     hue="Prefix", #palette=PALETTE,
        # )
        # ax.axhline(y=prompt_baseline, linewidth=2,
        #            color='black', ls='--')
        #
        # plt.xlabel("")
        # plt.ylim(0.5, 0.7)
        # plt.tight_layout()
        # plt.savefig(f"{exp_dir}/bar_{harness_task}.png")
        # plt.close()

        for prefix in ["persona gpt4"]:
            # then we plot them individually for each person
            sub_df = sub_df[sub_df.Prefix==prefix]
            sub_df = sub_df[sub_df.seen_during_train=="Persona not trained"]
            sub_df["Inference Prefix"] = "personal prefix"
            if if_subset_names:
                sub_df = sub_df[sub_df.name.isin(subset_names)]
                sub_df.loc[sub_df.name.str.startswith("Alexandria"),"name"] = "A. Ocasio-Cortez" # shorten name to make plot look nicer
            df_multi_no_prefix["Inference Prefix"] = "no prefix"
            df_multi_no_prefix_baseline = df_multi_no_prefix[(df_multi_no_prefix.ref_free_agg_method == "sum")&(df_multi_no_prefix.Prefix==prefix)].drop_duplicates(["name", "eval_path"])
            sub_df = pd.concat([sub_df, df_multi_no_prefix])
            print(sub_df.name.value_counts())
            hue_order = sub_df.groupby("name")[col_map[harness_task]].mean().sort_values().index
            if if_subset_names:
                plt.figure(figsize=(8, 3))
            else:
                plt.figure(figsize=(20, 3))
            plt.ylim(min(sub_df[col_map[harness_task]].min(),prompt_baseline) - 0.01, max(prompt_baseline,sub_df[col_map[harness_task]].max()) + 0.01)
            ax=sns.barplot(
                data=sub_df,
                x="name", y=col_map[harness_task],
                order=hue_order, palette=PALETTE,
                hue="Inference Prefix"
            )
            if if_subset_names:
                plt.xticks(rotation=-22, fontsize=10)
            else:
                plt.xticks(rotation=-25, fontsize=6)
            ax.axhline(y=prompt_baseline, linewidth=2,
                       color='black', ls='--')
            if harness_task != "reasoning_harness":
                plt.legend().remove()
            plt.xlabel("")
            plt.tight_layout()
            # plt.savefig(f"{exp_dir}/bar_{'all' if not if_subset_names else 'subset'}_individual_{prefix}_{harness_task}.png")
            plt.savefig(
                f"{exp_dir}/bar_{'all' if not if_subset_names else 'subset'}_individual_{prefix}_{harness_task}.pdf", dpi=600)

            plt.close()

    # plot scatterplot of prompting results vs multitask results per individual
    df_prompt = df_prompt[(df_prompt.ref_free_agg_method=="avg")].drop_duplicates(["eval_path", "name"])
    df_prompt = df_prompt[~df_prompt.Prefix.isna()]
    for m in ["Total Accuracy"]:  # "Personal Accuracy", "Common Accuracy"
        for seen_during_train in ["Persona not trained", "Persona trained"]:  # , "Persona trained"
            df_multi = df[(df.seen_during_train==seen_during_train) &
                          (df.ref_free_agg_method=="avg") &
                          (df[m]>0)].drop_duplicates(["eval_path", "name"])
            df_multi = df_multi[~df_multi.Prefix.isna()]

            # # here we compare multitask vs prompting (with the same prefix type)
            # df_join = pd.merge(df_multi, df_prompt[["name", "Prefix", m]], on=["name", "Prefix"], how="left",
            #                    suffixes=[" (multitask trained)", " (not trained)"])
            # df_join = df_join[~df_join[f"{m} (not trained)"].isna()]
            # plt.figure()
            # sns.lmplot(data=df_join, x=f"{m} (not trained)", y=f"{m} (multitask trained)", hue="Prefix")#, palette=PALETTE)
            # min_val = df_join[[f"{m} (not trained)", f"{m} (multitask trained)"]].min().min()
            # plt.axline((min_val, min_val), slope=1, color='black', linestyle='--')
            # plt.savefig(f"{exp_dir}/multitask_vs_prompt_{m}_{seen_during_train}_correlation.png")
            # plt.tight_layout()
            # plt.close()

            # instead for x-axis use baseline-no prefix
            df_join = pd.merge(df_multi, df_prompt[df_prompt.Prefix=="no prefix"][["name", m]], on=["name"], how="left",
                               suffixes=[" (MT)", " (Zephyr no prefix)"])
            df_join = df_join[~df_join[f"{m} (Zephyr no prefix)"].isna()]

            # calculate pearson correlation add to plot
            p_map = {}
            method_to_significance = {}
            for prefix in df_join.Prefix.unique():
                cor_df = df_join[df_join.Prefix == prefix]
                r, p = sp.stats.pearsonr(cor_df[f"{m} (Zephyr no prefix)"], cor_df[f"{m} (MT)"])
                # chi2_stat, p_val, dof, expected = sp.stats.chi2_contingency([cor_df[f"{ref}_{task}"], cor_df['avg_sim']])
                # r_kt, p_kt = sp.stats.kendalltau(cor_df[f"{ref}_{task}"], cor_df['avg_sim'])
                p_map[prefix] = f"{prefix} (pearson r={r:.2g}, p={p:.2g})"
                method_to_significance[p_map[prefix]] = p
            df_join["Prefix"] = df_join.Prefix.map(p_map)
            # hue_order = sorted(sub_df.specific_method.unique(), key=lambda x: method_to_significance[x])

            plt.figure() # figsize=(10,5)
            g = sns.lmplot(data=df_join, x=f"{m} (Zephyr no prefix)", y=f"{m} (MT)",
                           hue="Prefix", legend=True, height=5, aspect=1, markers='o',
                           hue_order=[p_map[p] for p in ["no prefix", "tag", "vpl", "few-shot", "persona", "persona gpt4" , "persona gold"]]# "name"
                           )  # , palette=PALETTE)
            min_val = df_join[[f"{m} (Zephyr no prefix)", f"{m} (MT)"]].min().min()
            plt.axline((min_val, min_val), slope=1, color='black', linestyle='--')

            # sns.move_legend(  # only works for ax plot, not facetGrid
            #     g, "lower center",
            #     bbox_to_anchor=(.5, 1), ncol=6, title=None, frameon=False,
            # )
            # plt.legend(loc='upper center', fontsize=9) # bbox_to_anchor=(0, 1),
            # g.fig.legend(loc='lower right', bbox_to_anchor=(1.25, 0))
            # plt.tight_layout()
            # plt.savefig(f"{exp_dir}/multitask_vs_prompt_no-prefix_{m}_{seen_during_train}_correlation.png")
            plt.savefig(f"{exp_dir}/multitask_vs_prompt_no-prefix_{m}_{seen_during_train}_correlation.pdf", dpi=600)
            plt.close()


def rename_df_for_plotting(df):
    df_persona = pd.read_json("../data/pp-50-final/personas.json", lines=True, orient="records")
    flat_name_to_formal_name = {n.lower().replace(" ", ""): n for n in df_persona.name}
    df["name"] = df.name.map(flat_name_to_formal_name)
    df["seen_during_train"] = df.seen_during_train.map({
        True: "Persona trained",
        False: "Persona not trained",
    }).fillna("Persona not trained")
    col_map = {
        "method_category": "Model",
        "specific_method": "Prefix",
        "ref-free_combined_pref": "Total Accuracy",
        "ref-free_personal_prefs": "Personal Accuracy",
        "ref-free_common_prefs": "Divergent Accuracy",
        "ref-free_reasoning": "Reasoning (RB)",
        "ref-free_safety": "Safety",
        "reasoning_harness": "Reasoning",
        "factual_harness": "Factuality",
    }
    baseline_val_map = {
        "persona_gold_gpt4": "persona gold",
        # "majority_class": "random",
        "name": "name",
        "no_prefix": "no prefix",
        "tag": "tag",
        "vpl": "vpl",
        "xyw_2s": "few-shot",
        "persona_xy_4s": "persona",
        "persona_xy_4s_llama321b": "persona",
        "persona_xy_4s_llama323b": "persona",
        "persona_xy_4s_ministral7b": "persona",
        "persona_xy_4s_gpt4": "persona gpt4",
        # retrieval trained
        "xyw_2s_bm25": "few-shot (retrieval)",
        "xyw_2s_bm25_reverse": "few-shot (retrieval, reverse)",
        "persona_xy_4s_bm25": "persona (retrieval)",
        "persona_xy_4s_bm25_reverse": "persona (retrieval, reverse)",
    }

    df.rename(columns=col_map, inplace=True)
    baseline_val_map = {k:v for k,v in baseline_val_map.items() if k in df.Prefix.unique()}
    df["Prefix"] = df.Prefix.map(baseline_val_map)
    df["Model"] = df.Model.map({
        "prompt": "Zephyr-7B",
        "personal_model": "personal model (PM)",
        "multitask_model": "multi-task model (MT)"
    })
    return df, col_map, baseline_val_map


def performance_in_each_demographics(dump_dir):
    exp_dir = f"{dump_dir.split('/')[-1]}/performance_in_each_demographics"
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.read_csv(f"{dump_dir}/aggregated_results.csv")
    df = df[~df.name.isin(["no_prefix"])]
    # df = df[df.inference_persona.isna()]
    df = df[df.cv.apply(lambda c: (not str(c).replace("_", "").isalpha()) or (c!=c))]

    # select value to plot
    df = df[df.specific_method == "persona_xy_4s"]  #  persona_xy_4s_gpt4
    df = df[(df.ref_free_agg_method == "avg") & (~df.common_size.isna()) & (df["ref-free_combined_pref"]>0)].drop_duplicates(["name", "eval_path"])
    df, _, _ = rename_df_for_plotting(df)
    df = df[df.seen_during_train == "Persona not trained"]
    # df = df[(df.alt_prefix.isna())|(df.Model == "Zephyr-7B")]
    df = df[df.alt_prefix.isna()]

    df_demo = pd.read_csv("../data/pp-50-final/personas.json/demographics.csv", index_col=False)
    df_demo["age"] = 2025 - df_demo["birth year"]


    # multitask performance vs demographics for each demographic attributes
    for attr in df_demo.columns:
        if attr in ["name", "birth year"]:
            continue

        print(f"plotting performance in {attr=}")
        if attr in ["age"]: # continuous values
            df_demo[f"{attr} binned"] = pd.cut(df_demo[attr], bins=5)
            df_join = pd.merge(df, df_demo[["name", f"{attr} binned"]], how="left", on="name")
            attr = f"{attr} binned"
        else:
            df_join = pd.merge(df, df_demo[["name", attr]], how="left", on="name")

        # explode data for multi-demographic values
        df_exp = []
        for i, row in df_join.iterrows():
            if isinstance(row[attr], str) and "/" in row[attr]:
                for attr_split in row[attr].split("/"):
                    row_split = row.copy()
                    row_split[attr] = attr_split
                    df_exp.append(row_split.to_dict())
            else:
                df_exp.append(row.to_dict())
        df_exp = pd.DataFrame(df_exp)

        # now we can plot: sort values by frequency, plot using bins
        df_gb = df_exp.groupby(attr)["Total Accuracy"]
        attr_map = {idx: f"{idx} ({row.count()/len(df_exp)*100:.0f}%)" for idx, row in df_gb}
        df_exp[attr] = df_exp[attr].map(attr_map)
        order = df_gb.count().sort_values().index
        plt.figure()
        g = sns.barplot(
            data=df_exp, x=attr, y=f"Total Accuracy",
            hue="Model", order=[attr_map[a] for a in order],
            palette=PALETTE
        )  # , )
        # for index, row in df_gb:
        #     g.text(index, row.mean()+0.01, round(row.count()/len(df_exp), 2), color='black', ha='center')

        # min_val = df_join[[f"{m} (not trained no prefix)", f"{m} (multitask trained)"]].min().min()
        # plt.axline((min_val, min_val), slope=1, color='black', linestyle='--')
        if len(order) <= 9:
            plt.xticks(rotation=-15)
        else:
            plt.xticks(rotation=-90)
        plt.tight_layout()
        plt.savefig(f"{exp_dir}/performance_vs_{attr}.png")
        plt.close()


def best_training_questions(dump_dir: str):
    exp_dir = f"{dump_dir.split('/')[-1]}/best_training_questions"
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.read_csv(f"{dump_dir}/aggregated_results.csv")
    df = df[~df.cv.isna()]
    df = df[~df.name.isin(["no_prefix", "timnitgebru", "yoshuabengio"])]
    df = df[df.alt_prefix.isna()]
    df = df[df.multi_prefix.isna()]
    df = df[(~df.specific_method.str.contains("gpt4")) & (~df.specific_method.str.contains("_max"))]
    df = df[(df.specific_method.str.startswith("persona_xy_4s")) | (df.specific_method.str.startswith("xyw_2s"))]
    df["training data"] = df.specific_method.apply(lambda s: "personal only" if "personal_only" in s else
                                                   "common only" if "common_only" in s else
                                                   "both")
    # for ref in ["ref-free"]:  # "ref-dpo"
    #     for task in ["combined_pref", "reasoning", "personal_prefs", "common_prefs", "safety"]:
    #         agg_method = "avg" if "pref" in task else "sum"
    #         plt.figure()
    #         # plt.ylim(0.5, 0.7)
    #         sub_df = df[df.ref_free_agg_method == agg_method]
    #         sns.barplot(sub_df[sub_df[f"{ref}_{task}"] > 0], x="seen_during_train", y=f"{ref}_{task}",
    #                     hue="training data")
    #         plt.savefig(f"{exp_dir}/bar_{ref}_{agg_method}_{task}.png")
    #         plt.close()


    # df.loc[df.name == "no_prefix", "seen_during_train"] = "No Prefix"
    # df.loc[df.name == "yoshuabengio", "seen_during_train"] = "Bengio"
    # df.loc[df.name == "timnitgebru", "seen_during_train"] = "Timnit Gebru"
    plt.figure()
    plt.ylim(0.5, 0.7)
    sub_df = df[(df.ref_free_agg_method == "avg") & (df.seen_during_train==False)]
    sub_df["specific_method"] = sub_df.specific_method.apply(lambda m: m.replace("_common_only", "").replace("_personal_only", ""))
    sns.barplot(data=sub_df[sub_df[f"reasoning_harness"] > 0], x="training data", y=f"reasoning_harness",
                hue="specific_method")
    plt.savefig(f"{exp_dir}/bar_reasoning_harness.png")
    plt.close()


def get_persona_similarity_from_preference_performance(df):
    df = df[df.method_category == "personal_model"]
    ref = "ref-free"
    personas = df.name.unique()
    sim_matrix = np.zeros([len(personas), len(personas)])

    for i, train_persona in enumerate(personas):
        for j, test_persona in enumerate(personas):
            if train_persona == test_persona:
                sim_matrix[i, j] = df[(df.name == train_persona) & (df.trained_on.isna())][
                    f"{ref}_combined_pref"].mean()
            else:
                sim_matrix[i, j] = df[(df.name == test_persona) & (df.trained_on == train_persona)][
                    f"{ref}_combined_pref"].mean()

    sim_dict = {}
    for i, train_persona in enumerate(personas):
        for j, test_persona in enumerate(personas):
            sim_dict[tuple(sorted([train_persona, test_persona]))] = (sim_matrix[i, j] + sim_matrix[j, i]) / 2

    return personas, sim_matrix, sim_dict


def persona_similarity_heatmap(dump_dir):
    exp_dir = f"{dump_dir.split('/')[-1]}/persona_similarity"
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.read_csv(f"{dump_dir}/aggregated_results.csv")
    personas, sim_matrix, sim_dict = get_persona_similarity_from_preference_performance(df)
    norm_sim_matrix = normalize(sim_matrix, axis=0, norm='l2')
    plt.figure()
    sns.heatmap(data=sim_matrix, xticklabels=personas, yticklabels=personas)
    plt.savefig(f"{exp_dir}/heatmap.png")
    plt.close()

    plt.figure()
    sns.heatmap(data=norm_sim_matrix, xticklabels=personas, yticklabels=personas)
    plt.savefig(f"{exp_dir}/heatmap_norm.png")
    plt.close()


CV_TEST_NAMES = [
    ["Donald Trump", "Halle Berry"], ["Bernie Sanders", "Jennifer Aniston"],
    ["Alexandria Ocasio-Cortez", "Gwyneth Paltrow"], ["Joe Biden", "Megan Fox"], ["Ellen DeGeneres", "Rand Paul"]
]
CV_TEST_NAMES2 = [[n.replace(" ", "").lower() for n in cv] for cv in CV_TEST_NAMES]


def get_other_test_persona(name) -> str:
    for cv in CV_TEST_NAMES2:
        if name in cv:
            other_name = [n for n in cv if n != name][0]
            return other_name
    raise ValueError("name not found")


def ood_multitask_performance_vs_persona_similarity(dump_dir):
    exp_dir = f"{dump_dir.split('/')[-1]}/ood_multitask_performance_vs_persona_similarity"
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.read_csv(f"{dump_dir}/aggregated_results.csv")
    personas, sim_matrix, sim_dict = get_persona_similarity_from_preference_performance(df)
    df = df[(~df.cv.isna()) & (df.multi_prefix.isna()) & (df.alt_prefix.isna()) & (df.seen_during_train==False)]
    df = df[(~df.specific_method.str.contains("only")) & (~df.specific_method.str.contains("_max"))]
    df = df[~df.specific_method.isin(["xyw_1s", "xyw_4s", "xyw_8s"])]
    df = df[df.refresh.isna()]
    df["specific_method"] = df.apply(lambda r: r.specific_method if r.inference_persona!=r.inference_persona else f"{r.specific_method}_inf{r.inference_persona}_max{r.max_len}_num{r.num_persona_inference_per_person}",1)
    df = df[(~df.specific_method.str.contains("infpersona")) | (df.specific_method=="persona_xy_4s_gpt4_infpersona_xy_4s_gpt4_max2048.0_num5.0")]
    df["specific_method"] = df.specific_method.apply(lambda x: "persona_xy_4s_gpt4 + inference_persona" if "infpersona" in x else x)
    df = df.drop_duplicates()
    for i, persona in enumerate(personas):
        other_test_persona = get_other_test_persona(persona)
        train_persona_indices = [i for i, p in enumerate(personas) if p not in {other_test_persona, persona}]
        avg_sim = (sim_matrix[i, train_persona_indices].mean() + sim_matrix[train_persona_indices, i].mean()) / 2
        df.loc[df.name == persona, "avg_sim"] = avg_sim

    ref = "ref-free"
    for task in ["combined_pref", "reasoning", "personal_prefs", "common_prefs", "safety"]: #["combined_pref", "reasoning", "personal_prefs", "common_prefs"]:
        agg_method = "avg" if "pref" in task else "sum"
        sub_df = df[df.ref_free_agg_method==agg_method]
        m_map = {}
        method_to_significance = {}
        for m in sub_df.specific_method.unique():
            cor_df = sub_df[sub_df.specific_method == m]
            r, p = sp.stats.pearsonr(cor_df[f"{ref}_{task}"], cor_df['avg_sim'])
            # chi2_stat, p_val, dof, expected = sp.stats.chi2_contingency([cor_df[f"{ref}_{task}"], cor_df['avg_sim']])
            r_kt, p_kt = sp.stats.kendalltau(cor_df[f"{ref}_{task}"], cor_df['avg_sim'])
            m_map[m] = f"{m} (r={r:.2g}, p={p:.2g}, p_kt={p_kt:.2g})"
            method_to_significance[m_map[m]] = (p+p_kt)/2
        sub_df["specific_method"] = sub_df.specific_method.map(m_map)
        hue_order = sorted(sub_df.specific_method.unique(), key=lambda x: method_to_significance[x])
        sub_df = sub_df.drop_duplicates(["eval_path","name"])
        print(sub_df.specific_method.value_counts())
        plt.figure()
        sns.lmplot(data=sub_df, x="avg_sim", y=f"{ref}_{task}", hue="specific_method",hue_order=hue_order)
        plt.savefig(f"{exp_dir}/{ref}_{task}_vs_avg_sim.png")
        plt.close()

    plt.figure()
    sns.lmplot(data=df, x="avg_sim", y=f"reasoning_harness", hue="specific_method")
    plt.savefig(f"{exp_dir}/reasoning_harness_vs_avg_sim.png")
    plt.close()


def ref_free_agg_methods_performance(dump_dir: str):
    exp_dir = f"{dump_dir.split('/')[-1]}/ref_free_agg_methods_performance"
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.read_csv(f"{dump_dir}/aggregated_results.csv")

    # prompt performance
    # df_prompt = df[(~df.ref_free_agg_method.isna()) & (df.method_category=="prompt")]
    # for ref in ["ref-free", "ref-dpo"]:
    #     for task in ["combined_pref", "reasoning"]:
    #         plt.figure()
    #         sns.barplot(df_prompt, x="ref_free_agg_method", y=f"{ref}_{task}", hue="specific_method")
    #         plt.savefig(f"{exp_dir}/bar_{ref}_{task}.png")
    #         plt.close()

    # multitask performance (more or less sanity check)
    df = df[df.method_category=="multitask_model"]
    df = df[~df.specific_method.str.contains("seed")]
    df.loc[df.ref_free_agg_method.isna(), "ref_free_agg_method"] = "avg"
    for ref in ["ref-free"]:
        for task in ["combined_pref", "reasoning", "safety"]:
            plt.figure()
            sns.barplot(df, x="ref_free_agg_method", y=f"{ref}_{task}", hue="specific_method")
            plt.savefig(f"{exp_dir}/bar_{ref}_{task}.png")
            plt.close()


def normalize_name(s: str):
    return s.replace(" ", "").lower()


def one_prefix_train_multi_prefix_eval_variance(dump_dir: str):
    """
    if you train single persona for this person during prefix, at test time,
    evaluate this person with
    - ID person with ID prefix (↑)
    - ID person with OOD prefix (↑)
    - OOD person (↑)
    - OOD person with wrong/shuffled prefix (↓)
        - shuffled can be broken down into shuffle in-axis or out-of-axis
    - ID person with wrong/shuffled ID prefix (↓)
    - ID person with wrong/shuffled OOD prefix (↓)
    """
    exp_dir = f"{dump_dir.split('/')[-1]}/one_prefix_train_multi_prefix_eval"
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.read_csv(f"{dump_dir}/aggregated_results.csv")
    df = df[~df.cv.isna()]
    df = df[~df.common_size.isna()]
    # TODO wait till no prefix jobs finish
    df_inference_no_prefix = df[(df.specific_method == "persona_xy_4s_gpt4") & (df.alt_prefix == "no_prefix") & (df.ref_free_agg_method=="avg") & (~df.common_size.isna()) & (df.seen_during_train==False)].drop_duplicates()
    df_no_prefix = df[(df.specific_method=="no_prefix")&(df.method_category=="multitask_model")&(df.seen_during_train==False) & (df.ref_free_agg_method=="avg")].drop_duplicates()
    hue_order = ["tag", "few-shot", "persona",  "persona gpt4", "name", "persona gold"]  #  "persona_xy_8s_gpt4",
    hue_order_old = ["tag","xyw_2s", "persona_xy_4s", "persona_xy_4s_gpt4", "name", "persona_gold_gpt4"]
    f = df[(~df.alt_prefix.isna()) | (df.specific_method.isin(hue_order_old))]
    # df = df[df.specific_method!= "no"]
    if "personas.json" not in dump_dir:
        df = df[~df.name.isin(["no_prefix", "timnitgebru", "yoshuabengio"])]
    else:
        df = df[~df.name.isin(["no_prefix"])]
    # df = df[df.seen_during_train == True]
    df.loc[df.alt_prefix.isna(), "alt_prefix"] = "" # convert nan back to empty string
    df = df[~df.specific_method.str.contains("_max")]

    df = df.drop_duplicates()
    df_no_prefix = df_no_prefix.drop_duplicates()

    def get_domain(r):
        return "Seen persona no prefix (↓)" if r.seen_during_train is True and r.alt_prefix=="no_prefix" \
            else "Unseen persona no prefix (↓)" if r.seen_during_train is False and r.alt_prefix=="no_prefix" \
            else "Unseen persona (↑)" if r.seen_during_train is False and not r.alt_prefix.startswith("shuffle") \
            else "Unseen persona wrong prefix (↓)" if r.seen_during_train is False and r.alt_prefix.startswith("shuffle") \
            else "Seen persona seen prefix (↑)" if r.alt_prefix == "" \
            else "Seen persona unseen prefix (↑)" if not r.alt_prefix.startswith("shuffle") \
            else "Seen persona wrong seen prefix (↓)" if "seed" not in r.alt_prefix \
            else "Seen persona wrong unseen prefix (↓)"
        # return "OOD person ID prefix" if r.seen_during_train is False and r.alt_prefix=="" \
        #     else "OOD person OOD prefix" if r.seen_during_train is False and not r.alt_prefix.startswith("shuffle") \
        #     else "OOD person wrong prefix" if r.seen_during_train is False and r.alt_prefix.startswith("shuffle") \
        #     else "ID person ID prefix" if r.alt_prefix == "" \
        #     else "ID person OOD prefix" if not r.alt_prefix.startswith("shuffle") \
        #     else "ID person wrong ID prefix" if "seed" not in r.alt_prefix \
        #     else "ID person wrong OOD prefix"

    df["domain"] = df.apply(lambda r: get_domain(r), 1)


    # df["prefix used in training"] = (df.alt_prefix == "")
    # df = df.rename(columns={"seen_during_train": "Have trained on this persona"})
    order = ["Seen persona seen prefix (↑)", "Seen persona unseen prefix (↑)", "Unseen persona (↑)", "Unseen persona wrong prefix (↓)",
             "Seen persona wrong seen prefix (↓)", "Seen persona wrong unseen prefix (↓)", "Seen persona no prefix (↓)", "Unseen persona no prefix (↓)"]
    # order = ["Seen person seen prefix (↑)", "Seen person unseen prefix (↑)", "Unseen person seen prefix (↑)", "OOD person OOD prefix", "OOD person wrong prefix",
    #          "ID person wrong ID prefix", "ID person wrong OOD prefix"]
    ref="ref-free"
    df, col_map, baseline_val_map = rename_df_for_plotting(df)
    df_no_prefix, _, _ = rename_df_for_plotting(df_no_prefix)
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(15, 5))
    for i, task in enumerate(["personal_prefs", "common_prefs"]): # "reasoning",
        agg_method = "avg" if "pref" in task else "sum"
        sub_df = df[df[col_map[f"{ref}_{task}"]]!= 0]
        sub_df = sub_df[(sub_df.inference_persona.isna())]
        sub_df = sub_df[sub_df.ref_free_agg_method == agg_method].drop_duplicates("eval_path")
        print(sub_df.Prefix.value_counts())
        sns.barplot(
            sub_df, x="domain", y=col_map[f"{ref}_{task}"], order=order,
            hue="Prefix", hue_order=hue_order, ax=axs[i]
        ) # hue="prefix used in training")
        for bar in axs[i].patches:
            if not isinstance(bar, str) and bar.xy[0] > 5.5:  # hacky way to make the bars with wrong prefix have hash
                try:
                    bar.set_hatch('x')
                except Exception as e:
                    print(f"weird bar cannot set hatch: {bar}. {e}")
            elif not isinstance(bar, str) and bar.xy[0] > 2.4:  # hacky way to make the bars with wrong prefix have hash
                try:
                    bar.set_hatch('//')
                except Exception as e:
                    print(f"weird bar cannot set hatch: {bar}. {e}")
        no_prefix_mt_baseline = df_no_prefix[df_no_prefix.seen_during_train == "Persona not trained"][col_map[f"{ref}_{task}"]].mean()
        axs[i].axhline(y=no_prefix_mt_baseline, linewidth=2, color='black', ls=':')

    # add the hash vs no hash
    handles, labels = axs[0].get_legend_handles_labels()
    handles.extend([
        matplotlib.patches.Patch(facecolor='white', edgecolor='black', hatch='', label='correct prefix'),
        matplotlib.patches.Patch(facecolor='white', edgecolor='black', hatch='//', label='wrong prefix'),
        matplotlib.patches.Patch(facecolor='white', edgecolor='black', hatch='x', label='no prefix')
    ])
    labels.extend(["correct prefix", "wrong prefix", "no prefix (inference time)"])
    axs[0].legend(handles=handles, labels=labels)

    sns.move_legend(
        axs[0], "lower center",
        bbox_to_anchor=(.5, 1), ncol=9, title="Prefix Types", frameon=False,
    )

    axs[0].set_ylim(0.5, 0.7)
    axs[1].set_ylim(0.5, 0.7)
    axs[1].set_xlabel("")
    axs[1].get_legend().remove()
    plt.xticks(rotation=-5)
    plt.tight_layout()
    plt.savefig(f"{exp_dir}/bar_prefix_sensitivity_personal_common.png")
    plt.close()

    ########################################
    # now let's look at OOD person with shuffled axis, we can look at in-axis vs out-of-axis difference
    # df_persona = pd.read_json("../data/personas.json/personas.json", lines=True, orient="records")
    # df_persona["name"] = df_persona.name.apply(lambda s: normalize_name(s))
    #
    # def if_same_axis(row, df_persona: pd.DataFrame):
    #     shuffle_name = df_persona[df_persona.name==row["name"]].shuffle_name.tolist()[0]
    #     shuffle_name_axis_list = df_persona[df_persona.name==normalize_name(shuffle_name)].axis.tolist()[0]
    #     shuffle_name_axis_list = set([a.split(":")[0].lower() for a in shuffle_name_axis_list])
    #     name_axis_list = df_persona[df_persona.name==row["name"]].axis.tolist()[0]
    #     name_axis_list = set([a.split(":")[0].lower() for a in name_axis_list])
    #     return len(shuffle_name_axis_list.intersection(name_axis_list)) > 0
    #
    # exp_df = df[df.domain=="OOD person wrong prefix"]
    # exp_df = exp_df[(exp_df.inference_persona.isna())]
    # exp_df["shuffled source"] = exp_df.apply(lambda row: if_same_axis(row, df_persona), 1)
    # for task in ["combined_pref", "personal_prefs", "common_prefs"]: # "reasoning",
    #     agg_method = "avg"
    #     plt.figure()
    #     sub_df = exp_df[exp_df[f"ref-free_{task}"]!= 0]
    #     sub_df = sub_df[sub_df.ref_free_agg_method == agg_method]
    #     plt.ylim(0.40, sub_df[f"ref-free_{task}"].max()-0.05)
    #     sns.barplot(sub_df, x="shuffled source", y=f"ref-free_{task}", hue="specific_method", hue_order=hue_order) # hue="prefix used in training")
    #     plt.xticks(rotation=-15)
    #     plt.savefig(f"{exp_dir}/OOD_shuffle_source_ref-free_{agg_method}_{task}.png")
    #     plt.close()

    ########################################
    # now let's look at whether different shots affect the persona quality in either
    # ID person OOD prefix or OOD person setting

    # exp_df = df[df.domain.isin(["OOD person ID prefix", "OOD person OOD prefix", "ID person OOD prefix"])]
    # exp_df.loc[exp_df.alt_prefix=="", "alt_prefix"] = "persona_xy_4s_gpt4"
    # exp_df = exp_df[exp_df.refresh.isna()]
    # exp_df["specific_method"] = exp_df.apply(
    #     lambda
    #         r: r.specific_method if r.inference_persona != r.inference_persona else f"{r.specific_method}_w_inf_{r.inference_persona}_{r.num_persona_inference_per_person}_{r.max_len}",
    #     1)
    # for m in ["persona_xy_4s_gpt4_w_inf_persona_xy_4s_gpt4_5.0_2048.0", "persona_xy_4s_gpt4"]:#, "persona_xy_4s_gpt4_w_inf_persona_xy_4s_gpt4_5.0_2048.0"]:
    #     sub_df = exp_df[exp_df.specific_method == m]
    #     for task in ["combined_pref", "personal_prefs", "common_prefs"]:
    #         agg_method = "avg"
    #         plt.figure()
    #         sub_df = sub_df[sub_df[f"ref-free_{task}"] != 0]
    #         sub_df = sub_df[sub_df.ref_free_agg_method == agg_method]
    #         print(sub_df.alt_prefix.value_counts())
    #         # plt.ylim(0.40, sub_df[f"ref-free_{task}"].max() - 0.05)
    #         sns.barplot(sub_df, x="seen_during_train", y=f"ref-free_{task}", hue="alt_prefix",
    #                     hue_order=[
    #                         "persona_xy_4s_gpt4",
    #                         "persona_xy_4s_gpt4_seed1",
    #                         "persona_xy_4s_gpt4_select_g_margin_yl",
    #                         "persona_xy_4s_gpt4_select_m_margin_yl",
    #                         "persona_xy_4s_gpt4_select_se_margin_yl",
    #                         "persona_xy_4s_gpt4_select_se_chosen",
    #                         "persona_xy_4s_gpt4_select_se_chosen_yl",
    #                         "persona_xy_4s_gpt4_select_se_margin_yl_reverse",
    #                         "persona_xy_4s_gpt4_select_se_margin_yl_sym",
    #                     ])  # hue="prefix used in training")
    #         plt.xticks(rotation=-15)
    #         plt.savefig(f"{exp_dir}/few_shot_importance{'_w_inf' if 'inf' in m else ''}_{agg_method}_{task}.png")
    #         plt.close()


def multiprefix_multitask_performance(dump_dir: str):
    exp_dir = f"{dump_dir.split('/')[-1]}/multiprefix_multitask_performance"
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.read_csv(f"{dump_dir}/aggregated_results.csv")
    df = df[~df.cv.isna()]
    df = df[~df.name.isin(["no_prefix", "timnitgebru", "yoshuabengio"])]
    df = df[df.alt_prefix.isna()]
    df = df[~df.multi_prefix.isna()]
    df["specific_method"] = df.apply(lambda r: r.specific_method + str(r.multi_prefix), 1)

    for ref in ["ref-free"]:  # "ref-dpo"
        for task in ["combined_pref", "reasoning", "personal_prefs", "common_prefs"]:
            agg_method = "avg" if "pref" in task else "sum"
            plt.figure()
            # plt.ylim(0.5, 0.7)
            sns.barplot(df[df.ref_free_agg_method==agg_method], x="seen_during_train", y=f"{ref}_{task}", hue="specific_method")
            plt.savefig(f"{exp_dir}/bar_{ref}_{task}.png")
            plt.close()

    # plt.figure()
    # plt.ylim(0.5, 0.7)
    # sns.barplot(data=df, x="seen_during_train", y=f"reasoning_harness", hue="specific_method")
    # plt.savefig(f"{exp_dir}/bar_reasoning_harness.png")
    # plt.close()


def add_majority_class_results(dump_dir:str, data_dir: str):
    """
    For each question, if we have seen the question before, use training answer distribution to find out
    what is most liked response. Otherwise, randomly pick amongst two options.

    turns out currently all common question prompts are not seen in training, so good for us, we don't
    have majority class votes at all.
    """
    df = pd.read_csv(f"{dump_dir}/aggregated_results.csv")
    if sum(df.method_category == "baseline") > 0:
        print("Random guess with majority class simulations found. No need to recompute.")
        return
    row_template = {
        "method_category": "baseline", "specific_method": "majority_class", "common_size": 50, "personal_size": 50, "cv": True
    }
    test_files = [f for f in os.listdir(data_dir) if f.startswith("test_") and f.count("_") > 2]
    common_test_dfs = {f.replace("test_", "").replace("_common_prefs.json",""): pd.read_json(f"{data_dir}/{f}", lines=True, orient="records") for f in test_files if "common" in f}
    personal_test_dfs = {f.replace("test_", "").replace("_personal_prefs.json", ""): pd.read_json(f"{data_dir}/{f}", lines=True, orient="records") for f in test_files if "personal" in f}

    all_names = list(set(common_test_dfs.keys()))
    baseline_rows = []
    for cv in range(5):
        train_df = pd.read_json(f"{data_dir}/cv{cv}/train_meta_prefs.json", lines=True, orient="records")
        train_df["name"] = train_df.name.apply(lambda n: n.replace(" ", "").lower())
        train_df["yw"] = train_df.chosen.apply(lambda x: x[1]["content"])
        train_pref_count = train_df.yw.value_counts()
        for name in all_names:
            row = row_template.copy()
            row["seen_during_train"] = name in train_df.name.unique()
            row["name"] = name
            def majority_guess(r):
                yw, yl = r.chosen[1]["content"], r.rejected[1]["content"]
                yw_count = train_pref_count[yw] if yw in train_pref_count else 0
                yl_count = train_pref_count[yl] if yl in train_pref_count else 0
                if yw_count == yl_count:
                    return np.random.choice(["yw", "yl"]) == "yw"
                else:
                    return yw_count > yl_count

            def random_guess(r):
                return np.random.choice(["yw", "yl"]) == "yw"

            row["ref-free_personal_prefs"] = (personal_test_dfs[name].apply(lambda r: random_guess(r), 1)).mean()
            row["ref-free_common_prefs"] = (personal_test_dfs[name].apply(lambda r: majority_guess(r), 1)).mean()
            row["ref-free_combined_pref"] = (row["ref-free_personal_prefs"]+row["ref-free_common_prefs"])/2
            baseline_rows.append(row)

    baseline_df = pd.DataFrame(baseline_rows)
    df = pd.concat([baseline_df, df])
    df.to_csv(f"{dump_dir}/aggregated_results.csv", index=False)


def compare_different_persona_inference_hyperparameters_1cv(dump_dir: str):
    exp_dir = f"{dump_dir.split('/')[-1]}/persona_inference_hyperparameters_cv0"
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.read_csv(f"{dump_dir}/aggregated_results.csv")
    df = df[~df.cv.isna()]
    df = df[df.cv == 0]
    df = df[~df.common_size.isna()]
    df = df[~df.name.isin(["no_prefix"])]#, "timnitgebru", "yoshuabengio"])]
    # df = df[df.alt_prefix.isna()]
    # df = df[df.multi_prefix.isna()]
    # df["specific_method"] = df.apply(lambda r: r.specific_method if r.multi_prefix!=r.multi_prefix else f"multi_prefix_{int(r.multi_prefix)}",1)
    df["specific_method"] = df.apply(
        lambda r: r.specific_method if r.inference_persona != r.inference_persona
        else f"{r.specific_method}_w_inf_{r.inference_persona}_{r.num_persona_inference_per_person}", 1)

    # 2 shot result seems to be best amongst all xyw
    # df = df[df.specific_method.str.startswith("xyw")]
    df = df[~df.specific_method.isin(["xyw_1s", "xyw_8s", "xyw_4s", "persona_xy_8s_gpt4"])]

    df = df[~df.specific_method.str.endswith("_only")]
    df = df[df.specific_method.str.startswith("persona_xy_4s")]

    df["inference_persona"] = df.inference_persona.fillna("")
    df = df.drop_duplicates().drop_duplicates("eval_path")

    # (self-training) Assuming no gpt4 persona, can we boostrap self performance?
    ref = "ref-free"
    # for task in ["personal_prefs", "common_prefs"]: # "combined_pref", "reasoning",  "safety"
    #     agg_method = "avg" if "pref" in task else "sum"
    #     plt.figure()
    #     # plt.ylim(0.5, 0.7)
    #     sub_df = df[df.ref_free_agg_method == agg_method]
    #     sub_df = sub_df[~sub_df.inference_persona.str.contains("gpt4")]
    #     sns.barplot(sub_df[sub_df[f"{ref}_{task}"]>0], x="seen_during_train", y=f"{ref}_{task}", hue="specific_method",
    #                 hue_order=[
    #                     "persona_xy_4s",
    #                     "persona_xy_4s_gpt4",
    #                     "persona_xy_4s_w_inf_persona_xy_4s_5.0",
    #                     "persona_xy_4s_w_inf_persona_xy_4s_10.0",
    #                     "persona_xy_4s_w_inf_persona_xy_4s_20.0",
    #                     # "persona_xy_4s_w_inf_persona_xy_4s_40.0",
    #                     # "persona_xy_4s_w_inf_persona_xy_4s_80.0",
    #                 ])
    #     plt.savefig(f"{exp_dir}/bar_{ref}_{agg_method}_{task}_bad_self_improve.png")
    #     plt.close()

    # (reasoning distillation) Assuming access to gpt4 persona, can we go beyond persona_xy_4s_gpt4?
    for task in ["personal_prefs", "common_prefs", "combined_pref"]:  # "combined_pref", "reasoning",  "safety"
        agg_method = "avg" if "pref" in task else "sum"
        plt.figure()
        # plt.ylim(0.5, 0.7)
        sub_df = df[df.ref_free_agg_method == agg_method]
        sub_df = sub_df[(~sub_df.specific_method.str.contains("4s_w_inf"))]
        sub_df = sub_df[sub_df[f"{ref}_{task}"] > 0]
        print(sub_df.specific_method.value_counts())
        sns.barplot(sub_df, x="seen_during_train", y=f"{ref}_{task}",
                    hue="specific_method",
                    hue_order=[
                        "persona_xy_4s",
                        "persona_xy_4s_gpt4",
                        "persona_xy_4s_gpt4_w_inf_persona_xy_4s_gpt4_0.1",
                        "persona_xy_4s_gpt4_w_inf_persona_xy_4s_gpt4_0.5",
                        "persona_xy_4s_gpt4_w_inf_persona_xy_4s_gpt4_1.0",
                        "persona_xy_4s_gpt4_w_inf_persona_xy_4s_gpt4_5.0",
                        "persona_xy_4s_gpt4_w_inf_persona_xy_4s_gpt4_10.0",
                        "persona_xy_4s_gpt4_w_inf_persona_xy_4s_gpt4_20.0",
                        "persona_xy_4s_gpt4_w_inf_persona_xy_4s_gpt4_40.0",
                    ]
                    )
        plt.savefig(f"{exp_dir}/bar_{ref}_{agg_method}_{task}_reasoning_distillation.png")
        plt.close()

    # (distillation) Assuming access to gpt4 persona, can we distil gpt4 persona while training preference on self-generated persona?
    # for task in ["personal_prefs", "common_prefs"]:  # "combined_pref", "reasoning",  "safety"
    #     agg_method = "avg" if "pref" in task else "sum"
    #     plt.figure()
    #     # plt.ylim(0.5, 0.7)
    #     sub_df = df[df.ref_free_agg_method == agg_method]
    #     sub_df = sub_df[(~sub_df.specific_method.str.contains("_w_inf")) | (sub_df.specific_method.str.contains("4s_w_inf_persona_xy_4s_gpt4"))]
    #     sub_df = sub_df[sub_df[f"{ref}_{task}"] > 0]
    #     sns.barplot(sub_df, x="seen_during_train", y=f"{ref}_{task}",
    #                 hue="specific_method",
    #                 hue_order=[
    #                     "persona_xy_4s",
    #                     "persona_xy_4s_gpt4",
    #                     "persona_xy_4s_w_inf_persona_xy_4s_gpt4_5.0",
    #                     "persona_xy_4s_w_inf_persona_xy_4s_gpt4_10.0",
    #                     "persona_xy_4s_w_inf_persona_xy_4s_gpt4_20.0",
    #                     "persona_xy_4s_w_inf_persona_xy_4s_gpt4_40.0",
    #                     "persona_xy_4s_w_inf_persona_xy_4s_gpt4_80.0",
    #                 ]
    #                 )
    #     plt.savefig(f"{exp_dir}/bar_{ref}_{agg_method}_{task}_distillation.png")
    #     plt.close()


def compare_different_persona_inference_hyperparameters(dump_dir: str):
    exp_dir = f"{dump_dir.split('/')[-1]}/persona_inference_hyperparameters"
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.read_csv(f"{dump_dir}/aggregated_results.csv")
    df = df[~df.cv.isna()]
    df = df[~df.name.isin(["no_prefix", "timnitgebru", "yoshuabengio"])]
    df = df[df.alt_prefix.isna()]
    # df = df[df.multi_prefix.isna()]
    df["specific_method"] = df.apply(lambda r: r.specific_method if r.multi_prefix!=r.multi_prefix else f"multi_prefix_{int(r.multi_prefix)}",1)
    df["specific_method"] = df.apply(
        lambda r: r.specific_method if r.inference_persona != r.inference_persona else f"{r.specific_method}_w_inf_{r.inference_persona}_{r.num_persona_inference_per_person}_{r.max_len}", 1)
    df = df[~df.specific_method.str.endswith("_only")]
    df["inference_persona"] = df.inference_persona.fillna("")
    df = df[df.inference_persona!="persona_xy_4s"]
    df = df[df.refresh.isna()]
    df =df.drop_duplicates().drop_duplicates(["eval_path","name","ref_free_agg_method"])
    # (reasoning distillation) Assuming access to gpt4 persona, can we go beyond persona_xy_4s_gpt4?
    ref = "ref-free"
    # for task in ["personal_prefs", "common_prefs"]:  # "combined_pref", "reasoning",  "safety"
    #     agg_method = "avg" if "pref" in task else "sum"
    #     plt.figure()
    #     # plt.ylim(0.5, 0.7)
    #     sub_df = df[df.ref_free_agg_method == agg_method]
    #     sub_df = sub_df[sub_df.specific_method.str.startswith("persona_xy_4s")]
    #     value_counts = sub_df["specific_method"].value_counts().to_dict()
    #     print(f"task={task}")
    #     pprint(value_counts)
    #     # sub_df["specific_method"] = sub_df.apply(lambda r: f"{r.specific_method}(cnt={value_counts[r.specific_method]})",1)
    #     sns.barplot(sub_df[sub_df[f"{ref}_{task}"] > 0], x="seen_during_train", y=f"{ref}_{task}",
    #                 hue="specific_method",
    #                 hue_order=[
    #                     "persona_xy_4s",
    #                     "persona_xy_4s_max2048",
    #                     "persona_xy_4s_max3072",
    #                     "persona_xy_4s_gpt4",
    #                     "persona_xy_4s_gpt4_max2048",
    #                     "persona_xy_4s_gpt4_max3072",
    #                     "persona_xy_4s_gpt4_w_inf_persona_xy_4s_gpt4_5.0_1024.0",
    #                     "persona_xy_4s_gpt4_w_inf_persona_xy_4s_gpt4_5.0_2048.0",
    #                     "persona_xy_4s_gpt4_w_inf_persona_xy_4s_gpt4_5.0_3072.0",
    #                     "persona_xy_4s_gpt4_w_inf_persona_xy_4s_gpt4_5.0_4096.0",
    #                 ]
    #                 )
    #     plt.savefig(f"{exp_dir}/bar_{ref}_{agg_method}_{task}_reasoning_distillation_persona.png")
    #     plt.close()

    for task in ["personal_prefs", "common_prefs"]:  # "combined_pref", "reasoning",  "safety"
        agg_method = "avg" if "pref" in task else "sum"
        plt.figure()
        # plt.ylim(0.5, 0.7)
        sub_df = df[df.ref_free_agg_method == agg_method]
        sub_df = sub_df[sub_df.specific_method.str.startswith("xyw")]
        value_counts = sub_df["specific_method"].value_counts().to_dict()
        print(f"task={task}")
        pprint(value_counts)
        # sub_df["specific_method"] = sub_df.apply(lambda r: f"{r.specific_method}(cnt={value_counts[r.specific_method]})",1)
        sns.barplot(sub_df[sub_df[f"{ref}_{task}"] > 0], x="seen_during_train", y=f"{ref}_{task}",
                    hue="specific_method",
                    hue_order=[
                        "xyw_1s",
                        "xyw_2s",
                        "xyw_2s_max2048",
                        "xyw_2s_max3072",
                        "xyw_4s",
                        "xyw_8s",
                    ]
                    )
        plt.savefig(f"{exp_dir}/bar_{ref}_{agg_method}_{task}_reasoning_distillation_fewshots.png")
        plt.close()


def oqa_prompt_performance_with_pp_finetuned(dump_dir: str):
    exp_dir = f"{dump_dir.split('/')[-1]}/best_prompt"
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.read_csv(f"{dump_dir}/aggregated_results.csv")
    df = df[df.method_category=="prompt"]
    df = df[df.specific_method.isin(["no_prefix", "xyw_2s", "persona_xy_16s", "persona_xy_16s_gpt4"])]
    df = df[df["ref-free_combined_pref"] != 0]
    df["finetuned"].fillna("no",inplace=True)
    ref="ref-free" #"ref-dpo"
    pdb.set_trace()
    for task in ["combined_pref"]: #"reasoning",
        plt.figure()
        ax = sns.barplot(df, x="specific_method", y=f"{ref}_{task}", hue="finetuned")
        plt.savefig(f"{exp_dir}/bar_{ref}_{task}.png")
        plt.close()


def load_featured_df(
    cv: bool = True,
    add_data_text: bool = False,
    add_persona_fields: Optional[List[str]] = None,
    ref_free: bool = True
):
    df = pd.read_csv(f"{dump_dir}/aggregated_results_detailed.csv")
    test_df = pd.read_json(
        "../data/pp-50-final/personas.json/test_50p_200d_total_50r_tem2.0_top0.8_cot_filtered20m4k_yl-random_cot_annotated.json", lines=True, orient="records")
    df["conversation_id"] = df.apply(lambda r: r["name"] + r.prompt, 1)

    # add in some features from test split
    if add_data_text:
        df = pd.merge(df, test_df[["name", "prompt", "yw", "yl"]], on=["name", "prompt"], how="left")

    # add in some features from personas df
    persona_df = pd.read_json("../data/pp-50-final/personas.json/personas.json", lines=True, orient="records")
    if cv:
        feature_cols = ["name", "cv"]
    else:
        feature_cols = ["name"]
    if add_persona_fields is not None:
        for add_persona_field in add_persona_fields:
            feature_cols.append(add_persona_field)
    if len(feature_cols) > 1:
        df = pd.merge(df, persona_df[feature_cols], on="name", how="left")

    if ref_free:
        df = df[df.ref_free == True]
    else:
        df = df[df.ref_free == False]
    return df


def analyze_no_prefix_signal_for_non_personalization():
    """
    Sometimes users may not want personalization, can we find a signal through same
    model just without the prefix?
    """
    exp_dir = f"{dump_dir.split('/')[-1]}/too_much_personalization_analysis"
    os.makedirs(exp_dir, exist_ok=True)
    all_df = load_featured_df(cv=True, add_data_text=True, ref_free=False)

    # method = "persona_xy_4s_gpt4"  # persona_xy_4s_gpt4_inf0.05, persona_xy_4s_gpt4, persona_xy_4s
    for method in ["persona_xy_4s_gpt4", "persona_xy_4s", "persona_gold_gpt4", "name", "xyw_2s"]:
        df = all_df[all_df.specific_method.isin([method])]
        # mean of this metric is 0.52 ish
        df["ref_margin"] = df["ref_chosen_score"] - df["ref_rejected_score"]
        df["ref_acc"] = df.ref_margin > 0
        df["pol_margin"] = df["pol_chosen_score"] - df["pol_rejected_score"]
        df["pol_acc"] = df.pol_margin > 0
        df["margin_diff"] = df.pol_margin - df.ref_margin
        # pick the model that has higher absolute margin? (usaully performance in the middle)
        df["comb_score_1"] = df.apply(
            lambda r: r.ref_margin if np.absolute(r.ref_margin) > np.absolute(r.pol_margin) else r.pol_margin, 1)
        df["comb_acc"] = df.comb_score_1 > 0
        # this wouldn't work, it assumes we know the direction
        # df["comb_score_2"] = df.apply(lambda r: max(r.ref_margin, r.pol_margin), 1)

        # how about reference scores (without prefix) alone?
        value_cols = ["ref_rejected_score", "ref_chosen_score", "pol_rejected_score", "pol_chosen_score"]
        melt_df = pd.melt(df, id_vars=['conversation_id'], value_vars=value_cols)
        plt.figure()
        sns.histplot(data=melt_df, x="value", hue="variable")
        plt.savefig(f"{exp_dir}/hist_different_logps.png")

        # get the ones where gpt4 got worse than the no prefix baseline, see if reference margin is any better
        no_prefix_saves = df[(df.seen_during_train==False) & (df.pol_acc==False)].ref_acc.mean()
        no_prefix_follows = df[(df.seen_during_train==False) & (df.pol_acc==True)].ref_acc.mean()
        print(f"For conversations where MT+{method} got right, \n"
              f"{no_prefix_follows* 100:.2f}% of examples are predicted correctly by MT with no prefix\n"
              f"For conversations where MT+{method} got wrong, \n"
              f"{no_prefix_saves * 100:.2f}% of examples are predicted correctly by MT with no prefix")

        res = df[(df.seen_during_train==False) & (df.pol_acc!=df.ref_acc)].pol_acc.mean()
        print(f"when MT with MT + {method} disagree, {res*100:.1f}% of time MT+{method} is correct")

        # if random_sample_seed is not None:
        #     random.seed(random_sample_seed)
        #     conv_ids = random.sample(all_conv_ids, n)
        # else:
        #     conv_ids = all_conv_ids[:n]
        #
        # for i, conv_id in enumerate(conv_ids):
        #     row = df[(df.conversation_id==conv_id) & (df.specific_method==method)]
        #     print(f"conversation {conv_id}, conversation type={row.conversation_type.tolist()[0]}")
        #     if method != "no_prefix":
        #         print("========================== prefix ==========================")
        #         print(row[method].tolist()[0])
        #     print("========================== prompt ==========================")
        #     print(row.prompt.tolist()[0])
        #     print(f"========================== yl (score={row.scores_rejected.tolist()[0]:.2f}) ==========================")
        #     print(row.yl.tolist()[0])
        #     print(f"========================== yw (score={row.scores_chosen.tolist()[0]:.2f}) ==========================")
        #     print(row.yw.tolist()[0])
        #     print("==========================   exp  ==========================")
        #     print(row.open_feedback.tolist()[0])
        #     print("")
        #     print("")


def mt_model_robustness_with_retrieval(dump_dir: str, cv="cv", model:Optional[str]="zephyr"):
    exp_dir = f"{dump_dir.split('/')[-1]}/ood_multitask_performance{'' if cv=='cv' else f'_{cv}'}{'' if model=='zephyr' else f'_{model}'}"
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.read_csv(f"{dump_dir}/aggregated_results.csv")
    df_multi_no_prefix = df[df.name == "no_prefix"]
    df_prompt = df[df.method_category=="prompt"]
    df_multi_no_prefix_pref = df[(df.alt_prefix=="no_prefix")&(df.ref_free_agg_method=="avg")&(~df.common_size.isna())].drop_duplicates()
    df = df[~df.cv.isna()]
    if cv == "cv":
        df = df[df.cv.apply(lambda c: not str(c).replace("_","").isalpha())]
    else:
        df = df[df.cv.apply(lambda c: str(c).replace("_","").isalpha())]
    if "model" in df.columns and model is not None:
        df = df[df.model.str.lower() == model.lower()]

    df = df[df.specific_method.isin(["persona_xy_4s", "persona_gold_gpt4"])] # "xyw_2s",
    df = df[(~df.personal_size.isna()) & (df['ref-free_common_prefs']!=0) &(df.alt_prefix!="no_prefix")]
    df.loc[df.alt_prefix.isna(), "alt_prefix"] = ""
    df = df[~df.alt_prefix.str.startswith("shuffle_")]
    df = df.drop_duplicates()

    df, col_map, _ = rename_df_for_plotting(df)
    df_prompt, _, _ = rename_df_for_plotting(df_prompt)
    df_multi_no_prefix, _, _ = rename_df_for_plotting(df_multi_no_prefix)
    df_multi_no_prefix["Inference Prefix"] = df_multi_no_prefix["name"] = "no prefix"
    df_multi_no_prefix_pref, _, _ = rename_df_for_plotting(df_multi_no_prefix_pref)
    df_multi_no_prefix_pref["seen_during_train"] = df_multi_no_prefix_pref.seen_during_train + " (no prefix)"
    print(df.Prefix.value_counts())
    sns.boxplot(df, x="Prefix", y="Total Accuracy", hue="alt_prefix")
    plt.show()
    return


def add_diagonal_line(ax):
    # Get the current axis limits to draw the x=y line across the entire plot
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    min_val = min(x_min, y_min)
    max_val = max(x_max, y_max)
    # Plot the x=y line
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='x=y line')


def performance_comparison_few_shot_vs_persona(dump_dir: str, cv="cv", model: Optional[str] = "zephyr"):
    exp_dir = f"{dump_dir.split('/')[-1]}/ood_multitask_performance{'' if cv == 'cv' else f'_{cv}'}{'' if model == 'zephyr' else f'_{model}'}"
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.read_csv(f"{dump_dir}/aggregated_results.csv")
    df = df[~df.cv.isna()]
    if cv == "cv":
        df = df[df.cv.apply(lambda c: not str(c).replace("_", "").isalpha())]
    else:
        df = df[df.cv.apply(lambda c: str(c).replace("_", "").isalpha())]
    if "model" in df.columns and model is not None:
        df = df[df.model.str.lower() == model.lower()]
    else:
        df["model"] = df.model.fillna("")
        df = df[~df.model.str.contains("_not_tuned")]
    if "inference_persona" in df.columns:
        df = df[df.inference_persona.isna()]
    df = df[~df.name.isin(["no_prefix"])]  # for 10 people
    if "multi_prefix" in df.columns:
        df = df[df.multi_prefix.isna()]

    df = df[df.specific_method.isin(["xyw_2s", "persona_xy_4s"])]
    if "refresh" in df.columns:
        df = df[(df.refresh.isna()) & (df.max_len.isna()) & (df.inference_persona.isna()) & (
            df.num_persona_inference_per_person.isna())]
    df = df[(~df.personal_size.isna()) & (df['ref-free_common_prefs'] != 0) & (df.alt_prefix != "no_prefix")]

    # if aggregating across model do this
    # df = df[df.alt_prefix.isna()]
    # if aggregating across seeds do this
    df.loc[df.alt_prefix.isna(),"alt_prefix"] = "original_prefix_seed0"
    df = df[(df.model.str.lower()=="zephyr") & (df.alt_prefix.str.contains("seed")) & (~df.alt_prefix.str.contains("shuffle"))]
    df["alt_prefix"] = df.alt_prefix.apply(lambda x: x if "original" in x else x[x.find("seed"):])

    df = df.drop_duplicates()

    df, col_map, _ = rename_df_for_plotting(df)
    df = df[df.seen_during_train=="Persona not trained"]
    print(df.Prefix.value_counts())
    print(df[["Prefix", "alt_prefix"]].value_counts())
    # merge on=model if multiple models, alt_prefix if multiple prefix
    rel_cols = ["name", "alt_prefix", "prefix", "model", "Total Accuracy"]
    df_join = pd.merge(df[df.Prefix=="few-shot"][rel_cols], df[df.Prefix=="persona"][rel_cols], on=["name", "model", "alt_prefix"], how="left",suffixes=[" (few-shot)", " (persona)"])
    df_join = pd.merge(df_join, df[df.Prefix=="persona gold"],on=["name", "model", "alt_prefix"],)
    df_join.rename(columns={"Total Accuracy": "Total Accuracy (persona gold)"}, inplace=True)
    sig_dict = {}
    for model in df_join.model.unique():
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df_join[df_join.model==model]["Total Accuracy (few-shot)"], df_join[df_join.model==model]["Total Accuracy (persona)"])
        sig_dict[model] = f"{model} (eqn={slope:.2g}x+{intercept:.2g}, std err={std_err:.2g})"
    df_join["model"] = df_join.model.map(sig_dict)

    # scatter plot of few-shot vs persona performance. a bit clustered
    # df_join_subset = df_join[df_join.name.isin(df_join.name.sample(10))]
    # df_persona_quality = persona_quality()
    # df_join = pd.merge(df_join, df_persona_quality, on=["name"])
    # ax = sns.scatterplot(df_join_subset, x="Total Accuracy (few-shot)", y="Total Accuracy (persona)", hue="model", size=2) #
    # add_diagonal_line(ax)
    # for index, row in df_join_subset.iterrows():
    #     ax.text(row["Total Accuracy (few-shot)"] + 0.002, row["Total Accuracy (persona)"] + 0.002, f"{row['name']}", fontsize=6, color='black')
    # plt.title("few-shot vs persona as prefix Performance")
    # plt.rcParams['figure.dpi'] = 600
    # plt.show()
    # plt.close()

    # maybe better way to visualize this is through sorted boxplot
    # this might be too pessimistic of a view on persona performance lol
    # let's aggregate over seeds
    df_join["Persona - Few-shot"] = df_join["Total Accuracy (persona)"] - df_join["Total Accuracy (few-shot)"]
    # plt.figure(figsize=(20, 10))
    # ax = sns.boxplot(df_join, x="name", y="Persona - Few-shot", order=df_join.groupby("name")["Persona - Few-shot"].median().sort_values().index)
    # plt.xticks(rotation=90)
    # ax.axhline(y=0, color='r', linestyle='--', label='no difference')
    # plt.tight_layout()
    # plt.show()

    # now let's see if we find any correlation between persona vs improvement
    df_persona_quality = persona_quality()
    df_persona_long = pd.melt(df_persona_quality, id_vars=["name"], var_name="alt_prefix", value_name="Persona Quality")
    # df_persona_long["alt_prefix"] = df_persona_long.alt_prefix.apply(lambda x: "original_prefix_seed0" if "seed" not in x else x.replace("_comb_score", ""))
    # df_join["metric"] = df_join["Persona - Few-shot"]
    # df_join["metric_type"] = "Persona - Few-shot"
    # df_persona_long["metric"] = df_persona_long["Persona Quality"]
    # df_persona_long["metric_type"] = "Persona Quality"
    # df_join_2 = pd.concat([df_join, df_persona_long])
    # plt.figure(figsize=(20, 10))
    # ax = sns.boxplot(df_join_2, x="name", y="metric", hue="metric_type",
    #                  order=df_join.groupby("name")["metric"].median().sort_values().index)
    # plt.xticks(rotation=90)
    # ax.axhline(y=0, color='r', linestyle='--', label='no difference')
    # plt.tight_layout()
    # plt.show()
    # not the best type of plot to show, given two metrics have different ranges
    # let's plot it as a scatter plot with error bars
    x_mean = df_persona_long.groupby("name")["Persona Quality"].mean()
    y_mean = df_join.groupby("name")["Persona - Few-shot"].mean()
    y2_mean = df_join.groupby("name")["Persona - Few-shot"].mean()
    ax = plt.scatter(x_mean, y_mean)
    x_std = df_persona_long.groupby("name")["Persona Quality"].std()
    y_std = df_join.groupby("name")["Persona - Few-shot"].std()
    y2_std = df_join.groupby("name")["Persona - Few-shot"].std()
    plt.errorbar(x_mean, y_mean, xerr=x_std, yerr=y_std, fmt="o", color="r")
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
        df_persona_long.sort_values(["name", "alt_prefix"])["Persona Quality"],
        df_join.sort_values(["name", "alt_prefix"])["Persona - Few-shot"])
    plt.axline((0, intercept), slope=slope, color='blue', linestyle=':', label=f'Regression line ({slope:.2f}x+{intercept:.2f})')
    plt.xlabel("Persona Quality")
    plt.ylabel("Persona - Few-shot Performance")
    plt.show()

    return


def significance_between_prefixes(dump_dir: str, cv="cv", model: Optional[str] = "zephyr"):
    exp_dir = f"{dump_dir.split('/')[-1]}/ood_multitask_performance{'' if cv == 'cv' else f'_{cv}'}{'' if model == 'zephyr' else f'_{model}'}"
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.read_csv(f"{dump_dir}/aggregated_results.csv")
    df = df[~df.cv.isna()]
    if cv == "cv":
        df = df[df.cv.apply(lambda c: not str(c).replace("_", "").isalpha())]
    else:
        df = df[df.cv.apply(lambda c: str(c).replace("_", "").isalpha())]
    df.loc[df.model.isna(), "model"] = "Zephyr"
    if "model" in df.columns and model is not None:
        df = df[df.model.str.lower() == model.lower()]
    # for 10 people
    df = df[~df.name.isin(["no_prefix"])]
    if "alt_prefix" in df.columns:
        df = df[df.alt_prefix.isna()]

    df = df[~df.specific_method.isin(["xyw_1s", "xyw_8s", "xyw_4s"])]

    df = df[(~df.specific_method.str.endswith("_only")) & (~df.specific_method.str.contains("_max"))]
    if "refresh" in df.columns:
        df = df[(df.refresh.isna()) & (df.max_len.isna()) & (df.inference_persona.isna()) & (
            df.num_persona_inference_per_person.isna())]
    df = df.drop_duplicates()

    df, col_map, _ = rename_df_for_plotting(df)
    df = df[(~df.personal_size.isna())  & (df["Personal Accuracy"] != 0)]  # (df.Prefix.isin(["no prefix", "persona gold"]))


    # for model in df.model.unique():
    #     if "tuned" in model:
    #         continue
    #     for seen in ["Persona not trained", "Persona trained"]:
    #
    #         # look at significance between prefixes
    #         # for pf1 in ["no prefix"]:
    #         #     for pf2 in df.Prefix.unique():
    #         #         if pf2 == pf1:
    #         #             continue
    #         #         df1 = df[(df.model.str.lower() == model) & (df.Prefix == pf1) & (df.seen_during_train == seen)].sort_values(["name", "cv"])
    #         #         df2 = df[(df.model.str.lower() == model) & (df.Prefix == pf2) & (df.seen_during_train == seen)].sort_values(["name", "cv"])
    #         #         res = scipy.stats.ttest_ind(df1["Total Accuracy"], df2["Total Accuracy"])
    #         #         print(f"{model=}, {seen=}, {pf1=}, {pf2=}, {res=}")
    #

    df = df[~df.model.str.contains("tuned")]
    df = df[df.Prefix.isin(["no prefix","tag", "few-shot","persona","persona gold"])]
    pd.pivot_table(df, values="Divergent Accuracy",index=["model", "seen_during_train"],columns=["Prefix"],aggfunc=["mean", "std"],)

    gb = df.groupby(["model", "seen_during_train", "Prefix"])["Personal Accuracy"].mean()
    gb.pivot(index=["model", "seen_during_train"], columns='Metric', values='Value')
    return


def improvement_across_demographics(dump_dir: str, cv="cv", model: Optional[str] = "zephyr"):
    exp_dir = f"{dump_dir.split('/')[-1]}/improvements_across_demographics"
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.read_csv(f"{dump_dir}/aggregated_results.csv")
    df_prompt = df[(df.method_category == "prompt") & (df.ref_free_agg_method=="avg") & (df["ref-free_common_prefs"]!=0.0)].drop_duplicates()
    df = df[~df.cv.isna()]
    if cv == "cv":
        df = df[df.cv.apply(lambda c: not str(c).replace("_", "").isalpha())]
    else:
        df = df[df.cv.apply(lambda c: str(c).replace("_", "").isalpha())]
    df.loc[df.model.isna(), "model"] = "Zephyr"
    df = df[~df.model.str.contains("not_tuned")]
    if "model" in df.columns and model is not None:
        df = df[df.model.str.lower() == model.lower()]
        df_prompt = df_prompt[df_prompt.model.str.lower() == model.lower()]
    df = df[~df.specific_method.isin(["xyw_1s", "xyw_8s", "xyw_4s"])]
    df = df[(~df.specific_method.str.endswith("_only")) & (~df.specific_method.str.contains("_max"))]
    df = df.drop_duplicates()

    df, col_map, _ = rename_df_for_plotting(df)
    df_prompt, _, _ = rename_df_for_plotting(df_prompt)
    df_prompt['Model'] = df_prompt.Model.map({"llama321b": "Llama1B", "llama323b": "Llama3B", "ministral8b": "Ministral", "Zephyr": "Zephyr"})

    df = df[(~df.personal_size.isna())  & (df["Personal Accuracy"] != 0) & (df.seen_during_train=="Persona not trained") & (df.alt_prefix.isna())]
    df = df[df.Prefix.isin(["persona", "few-shot", "persona gold", "persona gpt4"])] # no_prefix
    df_prompt = df_prompt[(df_prompt.Prefix.isin(["persona", "few-shot", "persona gold", "persona gpt4"]))]

    relevant_cols = ["model", "name", "Prefix"]
    df_join1 = pd.merge(df, df_prompt[["Total Accuracy", *relevant_cols]] , on=relevant_cols, how="inner", suffixes=(" mt", " prompt"))
    df_join1["MT - Prompt (Total Accuracy)"] = df_join1["Total Accuracy mt"] - df_join1["Total Accuracy prompt"]

    print(df_join1[["Prefix", "model"]].value_counts())
    # df_join2 = pd.merge(df[(df.Prefix=="persona gpt4")], df[(df.Prefix=="no prefix")], on=relevant_cols, how="inner", suffixes=(" persona gpt4", " no prefix"))
    # df_join2["diff"] = df_join2["Total Accuracy persona gpt4"] - df_join1["Total Accuracy no prefix"]

    df_persona = pd.read_json(f"../data/pp-50-final/personas.json", lines=True, orient="records")
    df_persona = add_demographics_info(df_persona)

    df_join1 = df_join1.merge(df_persona, on="name")
    rows = []
    prefix = "persona gold"
    for category in DEMOGRAPHIC_ATTRIBUTE_CATEGORIES:
        if category == "age":
            df_join1[f"{category} binned"] = pd.cut(df_join1[category], bins=5)
            diff_by_groups = df_join1[df_join1.Prefix == prefix].groupby(f"{category} binned")["MT - Prompt (Total Accuracy)"].apply(list)
        else:
            diff_by_groups = df_join1[df_join1.Prefix==prefix].groupby(category)["MT - Prompt (Total Accuracy)"].apply(list)
        f_statistic, p_value = scipy.stats.f_oneway(*diff_by_groups)
        rows.append({
            "category": category,
            "f_stat": f_statistic,
            "p_value": p_value,
        })
    df_stat = pd.DataFrame(rows).sort_values("p_value")

    # now plot it
    # df_long = pd.concat([df, df_prompt]).merge(df_persona, on="name")
    for category in DEMOGRAPHIC_ATTRIBUTE_CATEGORIES:
        print(f"plotting performance in {category=}")
        if category in ["age"]:  # continuous values
            df_join1[f"{category} binned"] = pd.cut(df_join1[category], bins=5)
            category = f"{category} binned"

        # explode data for multi-demographic values
        df_exp = []
        for i, row in df_join1.iterrows():
            if isinstance(row[category], str) and "/" in row[category]:
                for attr_split in row[category].split("/"):
                    row_split = row.copy()
                    row_split[category] = attr_split
                    df_exp.append(row_split.to_dict())
            else:
                df_exp.append(row.to_dict())
        df_exp = pd.DataFrame(df_exp)

        # now we can plot: sort values by frequency, plot using bins
        df_gb = df_exp.groupby(category)["MT - Prompt (Total Accuracy)"]
        attr_map = {idx: f"{idx} ({row.count() / len(df_exp) * 100:.0f}%)" for idx, row in df_gb}
        df_exp[category] = df_exp[category].map(attr_map)
        order = df_gb.count().sort_values().index
        plt.figure()
        g = sns.barplot(
            data=df_exp, x=category, y=f"MT - Prompt (Total Accuracy)",
            hue="Prefix", order=[attr_map[a] for a in order],
            palette="rocket"
        )  # , )
        # for index, row in df_gb:
        #     g.text(index, row.mean()+0.01, round(row.count()/len(df_exp), 2), color='black', ha='center')

        # min_val = df_join[[f"{m} (not trained no prefix)", f"{m} (multitask trained)"]].min().min()
        # plt.axline((min_val, min_val), slope=1, color='black', linestyle='--')
        if len(order) <= 9:
            plt.xticks(rotation=-15)
        else:
            plt.xticks(rotation=-90)
        plt.tight_layout()
        plt.savefig(f"{exp_dir}/performance_vs_{category.replace(' ','_')}.png", dpi=600)
        plt.close()

    return


def retrieval_train(dump_dir: str, cv="cv", model:Optional[str]="zephyr"):
    exp_dir = f"{dump_dir.split('/')[-1]}/ood_multitask_performance{'' if cv=='cv' else f'_{cv}'}{'' if model=='zephyr' else f'_{model}'}"
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.read_csv(f"{dump_dir}/aggregated_results.csv")
    df = df[~df.cv.isna()]
    if cv == "cv":
        df = df[df.cv.apply(lambda c: not str(c).replace("_","").isalpha())]
    else:
        df = df[df.cv.apply(lambda c: str(c).replace("_","").isalpha())]
    if "model" in df.columns and model is not None:
        df = df[df.model.str.lower() == model.lower()]

    df = df[df.specific_method.str.contains("bm25") | (df.specific_method.isin(["xyw_2s", "persona_xy_4s"]) & (df.alt_prefix.isna()))] # "xyw_2s",
    df = df[(~df.personal_size.isna()) & (df['ref-free_common_prefs']!=0) &(df.alt_prefix!="no_prefix")]
    df.loc[df.alt_prefix.isna(), "alt_prefix"] = df[df.alt_prefix.isna()].specific_method
    df = df[~df.alt_prefix.str.startswith("shuffle_")]
    df = df.drop_duplicates()

    df, col_map, _ = rename_df_for_plotting(df)
    print(df[["Prefix", "alt_prefix"]].value_counts())
    g = sns.barplot(df, x="Prefix", y="Total Accuracy", hue="alt_prefix",
                order=[
                    "few-shot", "few-shot (retrieval)", "few-shot (retrieval, reverse)",
                    "persona", "persona (retrieval)", "persona (retrieval, reverse)",
                ])
    plt.xticks(rotation=-30)
    plt.ylim(0.55)
    plt.show()
    # no interesting difference.

    return



if __name__ == "__main__":
    # dump_dir = "../dump/pp-dpo_70"
    # dump_dir = "../dump/pp-dpo"
    dump_dir = "../dump/pp-50-final"
    # dump_dir = "../dump/OpinionQAReward"
    aggregate(dump_dir)
    # add_majority_class_results(dump_dir, "../data/pp-200/10p_200d_total_50r_tem2.0_top0.8_cot_filtered20m4k_yl-random_cot_annotated_self-yl")
    # what_is_the_best_prompt(dump_dir)  # paper figure 1 (pp-dpo) and retrieval comparison
    # prompt_upperbounds(dump_dir)
    # personal_model_performance(dump_dir)  # paper figure 2 (pp-dpo)
    # personal_model_performance_vs_data_size(dump_dir)
    # overall_method_performance_comparison(dump_dir)  # paper appendix figure (personas.json
    # performance_across_political_axis(dump_dir)
    # performance_data_scaling(dump_dir) # updated appendix figure 12 (pp-dpo)
    # performance_data_scaling_extra(dump_dir)  #
    # ood_performance_multitask(dump_dir)  # paper figure 3,4
    # ood_performance_multitask(dump_dir, cv="cv", model="zephyr")  # paper figure 3,4,
    # ood_performance_multitask(dump_dir, cv="axis", model="zephyr") # leave one out
    # ood_performance_multitask(dump_dir, cv="cv", model=None) # different models
    # persona_similarity_heatmap(dump_dir)
    # ood_multitask_performance_vs_persona_similarity(dump_dir)
    # ref_free_agg_methods_performance(dump_dir)
    # one_prefix_train_multi_prefix_eval_variance(dump_dir)  # appendix figure
    # multiprefix_multitask_performance(dump_dir)
    # best_training_questions(dump_dir)
    # compare_different_persona_inference_hyperparameters_1cv(dump_dir)
    # compare_different_persona_inference_hyperparameters(dump_dir)
    # oqa_prompt_performance_with_pp_finetuned(dump_dir)
    # performance_in_each_demographics(dump_dir)
    # analyze_no_prefix_signal_for_non_personalization()
    # mt_model_robustness_with_retrieval(dump_dir)
    # significance_between_prefixes(dump_dir, model=None)
    improvement_across_demographics(dump_dir, cv="cv", model=None)
    # retrieval_train(dump_dir, cv="cv", model="zephyr")
