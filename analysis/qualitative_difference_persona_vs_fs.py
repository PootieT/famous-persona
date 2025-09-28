import glob
import os
import pdb
from pathlib import Path
from typing import Optional, Literal

import scipy

from tqdm import tqdm
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import evaluate

DUMP_DIR=str(Path(__file__).parents[1].joinpath('dump').joinpath("pp-50-final").joinpath("one_model_for_all_cv").resolve().absolute())
DATA_DIR=str(Path(__file__).parents[1].joinpath('data').joinpath("pp-50-final").resolve().absolute())
OUT_DIR=str(Path(__file__).parent.joinpath('attributions').resolve().absolute())
PLOT_DIR=str(Path(__file__).parent.joinpath('pp-50-final').joinpath("attribution_plots").resolve().absolute())

TEMPLATE_SOURCES={
'Cater the response to how they might like, agree with, or be interested in.',
'Respond to the following prompt from this person.',
'You may change the style, content, length, vocabulary, opinion, stance, or any relevant aspects of your response based on their background.',
'## Prompt:',
'### Preferred Response:',
'### Dispreferred Response:',
}


def read_full_outputs_for_prefix(prefix: str, ood: bool=True, model:str="zephyr", finetuned:bool=True) -> pd.DataFrame:
    if finetuned:
        cv_dirs = [f"{DUMP_DIR}{'_'+model if model!='zephyr' else ''}/overton_{prefix}_cv{cv}" for cv in range(5)]
        total_df = []
        for cv, cv_dir in tqdm(enumerate(cv_dirs), total=len(cv_dirs)):
            eval_cv_dir = f"{cv_dir}/eval" if not ood else f"{cv_dir}/eval_OOD"
            for name_dir in os.listdir(eval_cv_dir):
                if name_dir == "no_prefix":
                    continue

                # should be 40 if ID and 10 if OOD
                res_file_paths = [
                    f"{eval_cv_dir}/{name_dir}/all_results_ref_free_avg_test_{name_dir}_common_prefs_scores_dict.json",
                    f"{eval_cv_dir}/{name_dir}/all_results_ref_free_avg_test_{name_dir}_personal_prefs_scores_dict.json",
                ]
                for res_file_path, q_type in zip(res_file_paths, ["common", "personal"]):
                    if not os.path.exists(res_file_path):
                        continue
                    df  = pd.read_json(res_file_path, lines=True, orient="records")
                    df["question_type"] = q_type
                    df["cv"] = cv
                    df["qid"] = range(len(df))
                    total_df.append(df)
    else:
        name_dirs = glob.glob(f"{DUMP_DIR.replace('one_model_for_all_cv', 'prompt')}{'_' + model if model != 'zephyr' else ''}/{prefix}_*")
        total_df = []
        for name_dir in name_dirs:
            if "_bm25_" in name_dir or "_emb_" in name_dir:
                continue
            name = f"{name_dir.split('/')[-1].split('_')[-1]}"
            res_file_paths = [
                f"{name_dir}/all_results_ref_free_avg_test_{name}_common_prefs_scores_dict.json",
                f"{name_dir}/all_results_ref_free_avg_test_{name}_personal_prefs_scores_dict.json",
            ]
            for res_file_path, q_type in zip(res_file_paths, ["common", "personal"]):
                if not os.path.exists(res_file_path):
                    continue
                df = pd.read_json(res_file_path, lines=True, orient="records")
                df["question_type"] = q_type
                df["qid"] = range(len(df))
                total_df.append(df)
    return pd.concat(total_df)


def print_example_different_predictions(
    good_prefix: str="persona_xy_4s",
    bad_prefix: str="xyw_2s",
):
    # read in all predictions
    good_df = read_full_outputs_for_prefix(good_prefix)
    bad_df = read_full_outputs_for_prefix(bad_prefix)

    # read in dataset examples
    data_df, persona_df = load_data_and_prefices()

    # read in attribution results
    good_attr_df = load_attribution_dfs(prefix=good_prefix)
    bad_attr_df = load_attribution_dfs(prefix=bad_prefix)

    assert len(good_df) == len(bad_df)
    # filter so good_df["results"]==1 and bad_df["results"]==0
    good_f = good_df.results==1
    bad_f = bad_df.results==0
    f = good_f & bad_f
    print(f"{good_prefix} succeeds {sum(good_f)}/{len(good_f)} times")
    print(f"{bad_prefix} succeeds {len(bad_f) - sum(bad_f)}/{len(bad_f)} times")
    print(f"total of {sum(f)} points where {good_prefix} succeeded but {bad_prefix} failed")

    good_suc_df = good_df[f]
    bad_fail_df = bad_df[f]
    pdb.set_trace()

    random_indices = np.random.choice(len(good_suc_df), 5, replace=False)
    for i in random_indices:
        grow = good_suc_df.iloc[i]
        # brow = bad_fail_df.iloc[i]
        # pdb.set_trace()
        try:
            drow = data_df[(data_df.qid==grow.name) & (data_df.name==grow["name"]) & (data_df.question_type==grow["question_type"])].to_dict("records")[0]
            g_attr_row = good_attr_df[(good_attr_df.qid == grow.name) & (good_attr_df.name == grow["name"]) & (good_attr_df.question_type == grow["question_type"])].to_dict("records")[0]
            b_attr_row = bad_attr_df[(bad_attr_df.qid == grow.name) & (bad_attr_df.name == grow["name"]) & (bad_attr_df.question_type == grow["question_type"])].to_dict("records")[0]
        except IndexError:
            print(f'Somehow datapoint {i} for {grow["name"]}, {grow["question_type"]} cannot be found in one of the df')
            continue
        psrow = persona_df[persona_df['name']==grow["name"]].to_dict("records")[0]

        #### print some basic information
        # print(f"idx: {i}, name: {grow['name']}, question_type: {grow['question_type']}")
        # print(f"prompt:\n{drow['prompt']}")
        # print("="*40)
        # print(f"Good Prefix-{good_prefix}:")
        # print(f"{psrow[good_prefix]}" if good_prefix != "no_prefix" else "")
        # print("=" * 40)
        # print(f"Bad Prefix-{bad_prefix}:")
        # print(f"{psrow[bad_prefix]}" if bad_prefix != "no_prefix" else "")
        # print("=" * 40)
        # print(f"yw ({good_prefix}'s choice):\n{drow['yw']}")
        # print("=" * 40)
        # print(f"yl ({bad_prefix}'s choice):\n{drow['yl']}")
        # print("=" * 40)

        #### some attribution information




    return


def load_data_and_prefices(prefix:Optional[str]=None, load_persona:bool=True):
    # load data examples
    if prefix is None:
        data_df = pd.read_json(
            f"{DATA_DIR}/test_50p_200d_total_50r_tem2.0_top0.8_cot_filtered20m4k_yl-random_cot_annotated.json",
            lines=True, orient="records")
        data_df["qid"] = data_df.groupby(['name', "question_type"]).cumcount()
    else:
        data_prefix = "" if prefix == "no_prefix" else f"_{prefix}-name-prefixed"
        data_dir = f"{DATA_DIR}/50p_200d_total_50r_tem2.0_top0.8_cot_filtered20m4k_yl-random_cot_annotated_self-yl{data_prefix}"
        data_df = []
        for q_type in ["common", "personal"]:
            sub_df = pd.read_json(f"{data_dir}/test_{q_type}_prefs.json", lines=True, orient="records")
            sub_df["question_type"] = q_type
            data_df.append(sub_df)
        data_df = pd.concat(data_df)
        data_df["qid"] = data_df.groupby(['name', "question_type"]).cumcount()

    # load persona df
    if load_persona:
        persona_df = pd.read_json(f"{DATA_DIR}/personas.json", lines=True, orient="records")
    else:
        persona_df = None
    return data_df, persona_df




def aggregate_attribution_results(
    score_type: Literal["pos", "neg", "diff"]="diff",
    prefix: str="persona_xy_4s",
    use_finetuned_models: bool = True,
    attribution_source_type: Literal["sentence", "word"] = "sentence"
):
    persona_df = pd.read_json(f"{DATA_DIR}/personas.json", lines=True, orient="records")
    res_df = []
    rouge = evaluate.load('rouge')

    attr_df = load_attribution_dfs(attribution_source_type, prefix, use_finetuned_models)
    if len(attr_df) == 0:
        return

    for name in attr_df["name"].unique():
        name_df = attr_df[attr_df["name"] == name]
        if score_type == "pos":
            scores = np.stack(name_df["chosen_scores"])
        elif score_type == "neg":
            scores = - np.stack(name_df["rejected_scores"])
        else:
            scores = np.stack(name_df["diff_scores"])

        # source sentence wise mask
        triv_mask = np.abs(scores).sum(0) == 0  # 0 contribution across all datapoints
        pos_mask = (scores > 0).sum(0) > 0  # >0 contribution across any datapoint
        neg_mask = (scores < 0).sum(0) > 0  # <0 contribution across any datapoint
        # example wise mask
        non_triv_examples = (scores != 0).sum(1) > 0  # contains at least one contributing source sentence
        top_sources = scores.mean(0).argsort()[::-1]
        name_res = {
            "prefix": prefix,
            "name": name,
            "source_count": scores.shape[1],
            # across sentences in prefix, how many of them contribute at all
            "trivial_source_count": sum(triv_mask),
            "positive_source_count": sum(pos_mask),
            "negative_source_count": sum(neg_mask),
            # amongst sentences that are non-trivial, how often do they contribute
            "source_positive_frequency": (scores[:, pos_mask] > 0).mean(),
            "source_negative_frequency": (scores[:, neg_mask] < 0).mean(),
            # how much do they impact on average?
            "positive_source_positive_magnitude": scores[:, pos_mask][scores[:, pos_mask] > 0].mean(),
            "positive_source_negative_magnitude": scores[:, pos_mask][scores[:, pos_mask] < 0].mean(),
            "positive_source_avg_magnitute": scores[:, pos_mask].mean(),
            "negative_source_positive_magnitude": scores[:, neg_mask][scores[:, neg_mask] > 0].mean(),
            "negative_source_negative_magnitude": scores[:, neg_mask][scores[:, neg_mask] < 0].mean(),
            "negative_source_avg_magnitude": scores[:, neg_mask].mean(),
            "avg_magnitude": scores.mean(),
            # how many examples are non trivial?
            "trivial_data_count": len(non_triv_examples) - sum(non_triv_examples),
            # amongst non-trivial datapoints, how many source sentences attribute at once?
            "data_positive_frequency": (scores[non_triv_examples] > 0).mean(),
            "data_negative_frequency": (scores[non_triv_examples] < 0).mean(),
            # top scores statistics
            "top1_avg_score": scores.mean(0)[top_sources[:1]].mean(),
            "top3_avg_score": scores.mean(0)[top_sources[:3]].mean(),
            "top5_avg_score": scores.mean(0)[top_sources[:5]].mean(),
        }

        # concat top-k positive attributing sentences, calculate ROUGE score against persona gold
        topk = 5
        source_strs = name_df.sources.tolist()[0]
        agg_source_persona = "\n".join([source_strs[i] if source_strs[i] not in TEMPLATE_SOURCES else "[prompt template]" for i in sorted(top_sources[:topk])])
        ref_persona = persona_df[persona_df["name"] == name]["persona_gold_gpt4"].values[0]
        metrics = rouge.compute(predictions=[agg_source_persona], references=[ref_persona])
        name_res.update(metrics)
        name_res[f"top{topk}_sources_persona"] = agg_source_persona

        res_df.append(name_res)

    res_df = pd.DataFrame(res_df)
    return res_df


def load_attribution_dfs(
    attribution_source_type: Literal["sentence", "word"] = "sentence",
    prefix: str = "persona_xy_4s",
    use_finetuned_models: bool = True,
    alt_inference_prefix: Optional[str] = None,
    model_short: str="zephyr"
):
    alt_inference_prefix = None if alt_inference_prefix==prefix else alt_inference_prefix
    attr_paths = [
        (f"{OUT_DIR}/{model_short}/{prefix}{'_finetuned' if use_finetuned_models else ''}"
         f"{'_alt'+alt_inference_prefix if alt_inference_prefix else ''}"
         f"_{attribution_source_type}{'_cv' + str(cv) if cv is not None else ''}.json")
        for cv in range(5)]
    attr_paths = [p for p in attr_paths if os.path.exists(p)]
    attr_df = pd.concat([pd.read_json(p, lines=True, orient="records") for p in attr_paths])
    attr_df["qid"] = attr_df.groupby(['name', "question_type"]).cumcount()
    attr_df["diff_scores"] = attr_df.apply(lambda r: [r.chosen_scores[i] - r.rejected_scores[i] for i in range(len(r.chosen_scores))], axis=1)
    return attr_df


def attribution_difference():
    total_df = []
    for prefix in ["xyw_2s", "persona_xy_4s", "persona_xy_4s_gpt4", "persona_gold_gpt4"]:
        res_df = aggregate_attribution_results(prefix=prefix)
        total_df.append(res_df)
    total_df = pd.concat(total_df)

    cols = ["data_positive_frequency", "source_positive_frequency",
            "top1_avg_score", "top3_avg_score", "top5_avg_score", "avg_magnitude",
            "rouge1", "rouge2", "rougeL", "rougeLsum"]
    res = total_df.groupby("prefix")[cols].mean()

    return


def explode_vector_column(df, column_name):
    """
    Explodes a column containing NumPy vectors into a long table.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column containing NumPy vectors.

    Returns:
        pd.DataFrame: The transformed DataFrame in long format.
    """

    df_exploded = df.apply(lambda row: pd.DataFrame({
        'source_score': row[column_name],
        'source_index': np.arange(len(row[column_name])),
        'original_row_index': row.name  # Optional: Keep track of original row
    }), axis=1)

    df_long = pd.concat(df_exploded.tolist(), ignore_index=True)

    df_long = df_long.merge(df, left_on="original_row_index", right_index=True)
    return df_long


# Calculate and add significance annotations
def add_significance(df, group_col, hue_col, hue1, hue2, value_col, y_pos, ax):
    significance_cnt = 0
    for group in df.source_index.unique():

        data_hue1 = df[(df[group_col] == group) & (df[hue_col] == hue1)][value_col]
        data_hue2 = df[(df[group_col] == group) & (df[hue_col] == hue2)][value_col]
        t_stat, p_val = scipy.stats.ttest_ind(data_hue1, data_hue2)  # or other statistical test

        if p_val < 0.05:
            significance_cnt += 1
            if p_val < 0.01:
                symbol = '**'
            else:
                symbol = '*'
            x1, x2 = ax.get_xticks()[group] - 0.2, ax.get_xticks()[group] + 0.2

            ax.plot([x1, x2], [y_pos, y_pos], lw=1.5, color='black')
            # add the significance astrick
            ax.text((x1 + x2) * 0.5, y_pos + 0.2, symbol, ha='center', va='bottom', color='black')
            # add the actual source text
            source_text = df.sources.tolist()[0][group]
            # ax.text((x1 + x2) * 0.5, y_pos + 0.4, source_text, ha='center', va='bottom', color='black')
            plt.figtext(0.1, 0.32 - 0.025*significance_cnt, f"{group}: {source_text}", ha='left', va='bottom', color='black')


def plot_attribution_distribution_between_success_and_fail():
    """
    distribution plot, hue=success vs fail, subplot over different people/prefixes
    """
    pmap = {"xyw_2s":"few-shot", "persona_xy_4s": "persona", "persona_xy_4s_gpt4": "persona gpt4", "persona_gold_gpt4": "persona gold"}
    for prefix in ["xyw_2s", "persona_xy_4s", "persona_xy_4s_gpt4", "persona_gold_gpt4"]: #

        # get predictions
        pred_df = read_full_outputs_for_prefix(prefix)

        # get attributions
        attr_df = load_attribution_dfs(prefix=prefix)

        join_df = attr_df.merge(pred_df, on=["qid", "name", "question_type"], how="inner")

        for name in ["Chaz Bono", ]: # "Barack Obama",   "Elon Musk", "Donald Trump"
            name_df = join_df[join_df["name"] == name]
            long_df = explode_vector_column(name_df, "diff_scores")
            success_cnt = name_df["results"].sum()
            total_cnt = len(name_df)

            # plot boxplot
            long_df["results"] = long_df["results"].apply(lambda r: "success" if r == 1 else "fail")
            # plt.figure(figsize=(3+int(len(join_df["chosen_scores"].tolist()[0])*0.4), 8))
            plt.figure(figsize=(20,5))
            ax = sns.boxplot(x="source_index", y="source_score", data=long_df, hue="results", fill=False)
            add_significance(long_df, "source_index","results", "success", "fail", "source_score", long_df['source_score'].max() + 1, ax)

            plt.title(f"{pmap[prefix]} with {name}, {success_cnt}/{total_cnt} Reward Prediction is Accurate")
            plt.tight_layout()
            plt.xlabel("Prefix Sentence Index")
            plt.ylabel("Attribution Score")
            # plt.show()
            plt.savefig(f"{PLOT_DIR}/success_vs_fail_{prefix}_{name}.pdf", dpi=600)
            plt.close()


def calculate_significance_stats(join_df):
    agg_df = []
    for name in join_df.name.unique():
        name_df = join_df[join_df["name"] == name]
        long_df = explode_vector_column(name_df, "diff_scores")
        success_cnt = name_df["results"].sum()
        significance_cnt = 0
        for idx in long_df.source_index.unique():
            data_hue1 = long_df[(long_df.source_index == idx) & (long_df.results == 1)]["source_score"]
            data_hue2 = long_df[(long_df.source_index == idx) & (long_df.results == 0)]["source_score"]
            t_stat, p_val = scipy.stats.ttest_ind(data_hue1, data_hue2)  # or other statistical test
            if p_val < 0.05:
                significance_cnt += 1
        agg_df.append({
            "name": name,
            "success_cnt": success_cnt,
            "success_frac": success_cnt / len(name_df),
            "significance_cnt": significance_cnt,
            "significance_frac": significance_cnt / len(long_df.source_index.unique()),
        })
    return agg_df


def calculate_overall_success_fail_significance_counts(model_short:str="zephyr"):
    """
    For each prefix, calculate count and fraction of significant sentences.
    """
    total_df = []
    for finetuned in [False, True]:
        for prefix in ["xyw_2s", "persona_xy_4s", "persona_gold_gpt4"]: #"persona_xy_4s_gpt4", "persona_gold_gpt4" "xyw_2s", "persona_xy_4s",
            prefix = f"{prefix}_{model_short}" if (model_short != "zephyr" and prefix=="persona_xy_4s") else prefix
            # get predictions
            pred_df = read_full_outputs_for_prefix(prefix, model=model_short, finetuned=finetuned)
            # get attributions "xyw_2s_bm25", "xyw_2s_bm25_reverse" "persona_xy_4s_bm25", "persona_xy_4s_bm25_reverse"
            # alt_inference_prefices = [None, "xyw_2s_seed1"] if prefix == "xyw_2s" \
            #     else [None, "persona_xy_4s_seed1", "persona_xy_4s_gpt4", "persona_gold_gpt4", ] if prefix == "persona_xy_4s" \
            #     else [None]
            alt_inference_prefices = [None]
            for alt_inference_prefix in alt_inference_prefices:
                attr_df = load_attribution_dfs(prefix=prefix, alt_inference_prefix=alt_inference_prefix, model_short=model_short, use_finetuned_models=finetuned)
                join_df = attr_df.merge(pred_df, on=["qid", "name", "question_type"], how="inner")
                agg_df = calculate_significance_stats(join_df)
                agg_df = pd.DataFrame(agg_df)
                agg_df["prefix"]= prefix
                agg_df["alt_inference_prefix"]= alt_inference_prefix
                agg_df["finetuned"] = finetuned
                total_df.append(agg_df)


    total_df = pd.concat(total_df)
    total_df.loc[total_df.alt_inference_prefix.isna(),"alt_inference_prefix"] = ""
    print(total_df.groupby(["prefix", "alt_inference_prefix", "finetuned"])["significance_frac"].mean())
    print(total_df.groupby(["prefix", "alt_inference_prefix", "finetuned"])["significance_frac"].std())

    print(total_df.groupby(["prefix", "alt_inference_prefix", "finetuned"])["significance_cnt"].mean())
    print(total_df.groupby(["prefix", "alt_inference_prefix", "finetuned"])["significance_cnt"].std())

    # see if there are relations between ones they are good at
    persona_field = f"persona_xy_4s{'_'+model_short if model_short!='zephyr' else ''}"
    pivot_df = total_df.pivot(index="name", columns="prefix", values="significance_frac")
    pivot_df["diff"] = pivot_df[persona_field] - pivot_df["xyw_2s"]
    print(pivot_df["diff"].describe())
    plt.close()
    sns.scatterplot(pivot_df, x="xyw_2s", y=persona_field)
    plt.axline((0, 0), slope=1, ls="--", c=".3", label="x=y line")  # Starting point (0,0) and slope 1
    plt.figaspect(1)
    plt.show()
    return


def calculate_high_attribution_count_for_retrievals():
    """
    For retrieval prefices (where every prompt is inferenced with a different prefix),
    we calculate the count and fraction of sentences that have high importance (in attribution score)
    """
    total_df = []
    for prefix in ["xyw_2s_bm25"]: # "xyw_2s_bm25", persona_xy_4s_bm25
        # get predictions
        # pred_df = read_full_outputs_for_prefix(prefix)
        # get attributions
        alt_inference_prefices = [None, "xyw_2s", "xyw_2s_bm25", "xyw_2s_bm25_reverse"] if prefix.startswith("xyw_2s") \
            else [None, "persona_xy_4s", "persona_xy_4s_bm25", "persona_xy_4s_bm25_reverse"] if prefix.startswith("persona_xy_4s") \
            else [None]
        for alt_inference_prefix in alt_inference_prefices:
            attr_df = load_attribution_dfs(prefix=prefix, alt_inference_prefix=alt_inference_prefix)
            # join_df = attr_df.merge(pred_df, on=["qid", "name", "question_type"], how="inner")
            attr_df["pos_source_cnt"]= attr_df.diff_scores.apply(lambda s: sum(np.array(s)>0))
            attr_df["pos_source_frac"]=attr_df.diff_scores.apply(lambda s: sum(np.array(s)>0)/len(s))
            attr_df["prefix"]= prefix
            attr_df["alt_inference_prefix"]= alt_inference_prefix
            total_df.append(attr_df)

    total_df = pd.concat(total_df)
    total_df.loc[total_df.alt_inference_prefix.isna(),"alt_inference_prefix"] = ""
    print(total_df.groupby(["prefix", "alt_inference_prefix"])["pos_source_frac"].mean())
    print(total_df.groupby(["prefix", "alt_inference_prefix"])["pos_source_frac"].std())
    return

if __name__ == "__main__":
    print_example_different_predictions(
        good_prefix="persona_xy_4s",
        bad_prefix="xyw_2s",
    )
    # aggregate_attribution_results(prefix="persona_xy_4s")
    # attribution_difference()
    # plot_attribution_distribution_between_success_and_fail()
    # calculate_overall_success_fail_significance_counts("llama323b")
    calculate_high_attribution_count_for_retrievals()