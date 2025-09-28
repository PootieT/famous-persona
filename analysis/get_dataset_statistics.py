import os
import pdb
from functools import partial
from typing import List, Dict, Union, Optional, Any
import json

import evaluate
import torch
import random
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from tqdm import tqdm
import spacy
import pandas as pd
import seaborn as sns
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    print("better not be using pycharm!")
import matplotlib.pyplot as plt
import plotly.express as px

sns.set_theme(style="ticks")
PALETTE = "vlag"
EXP_DIR = "pp-50-final/dataset_statistics"
ALL_AXES = ["age", "religion", "politics", "gender", "profession", "diet", "education level", "family marriage status", "AI professors", "geographical location", "sports"]
DEMOGRAPHIC_ATTRIBUTE_CATEGORIES = "age,race,gender,sexual preference,ethnicity,education level,family marriage status,birth state,birth country,current state,current country,religion,political affiliation,profession,economic status".split(",")

def get_string_len_statistics(train_path:str, eval_path: str):
    df_train = pd.read_json(train_path, lines=True, orient="records")
    df_eval = pd.read_json(eval_path, lines=True, orient="records")
    names = df_train.name.unique()
    stats = {}
    agg_stats = {}
    # metrics = ["num_chars", "num_words", "unique_words"]
    metrics = ["num_words"]
    fields = ["prompt", "yw", "yl"]
    for split, df in zip(["train", "eval"], [df_train, df_eval]):
        for name in names:
            for field in fields:
                sub_df = df[(df.name==name)]
                df.loc[df.name==name, f"{field}_num_chars"] = sub_df[field].apply(lambda s: len(s))
                df.loc[df.name == name, f"{field}_num_words"] = sub_df[field].apply(lambda s: len(s.split()))
                df.loc[df.name == name, f"{field}_unique_words"] = sub_df[field].apply(lambda s: len(set(s.split())))
        #         stats[f"{split}_num_chars_{field}_{name}"] = sub_df[field].apply(lambda s: len(s))
        #         stats[f"{split}_num_words_{field}_{name}"] = sub_df[field].apply(lambda s: len(s.split()))
        #         stats[f"{split}_unique_words_{field}_{name}"] = sub_df[field].apply(lambda s: len(set(s.split())))
        # for field in ["prompt", "yw", "yl"]:
        #     agg_stats[f"{split}_num_chars_{field}_agg"] = np.mean([stats[f"{split}_num_chars_{field}_{n}"] for n in names])
        #     agg_stats[f"{split}_num_words_{field}_agg"] = np.mean([stats[f"{split}_num_words_{field}_{n}"] for n in names])
        #     agg_stats[f"{split}_unique_words_{field}_agg"] = np.mean([stats[f"{split}_unique_words_{field}_{n}"] for n in names])

    os.makedirs(EXP_DIR, exist_ok=True)
    df_train["split"], df_eval["split"] = "train", "eval"
    df = pd.concat([df_train, df_eval])
    # for field in fields:
    #     for metric in metrics:
    #         plt.figure()
    #         sns.barplot(data=df, x="name", y=f"{field}_{metric}", hue="split")
    #         plt.xticks(rotation=30)
    #         plt.tight_layout()
    #         plt.savefig(f"{EXP_DIR}/bar_{field}_{metric}_train_vs_eval.png")
    #         plt.close()
    #
    # all fields in one
    for metric in metrics:
        # per person
        # df_long = pd.melt(df, id_vars=['name', "prompt", "yw"], value_vars=[f"{f}_{metric}" for f in fields])
        # plt.figure()
        # sns.barplot(data=df_long, x="name", y="value", hue="variable")
        # plt.xticks(rotation=30)
        # plt.tight_layout()
        # plt.savefig(f"{EXP_DIR}/bar_{metric}_different_fields.png")
        # plt.close()

        # per axis
        df_long = pd.melt(df, id_vars=['name', "prompt", "yw", "question_type", "axis"],
                          value_vars=[f"{f}_{metric}" for f in fields])
        df_long.rename(columns={"question_type": "Question type", "value": "Length"}, inplace=True)
        df_long["variable"] = df_long["variable"].map({
            "prompt_num_words": "prompt",
            "yw_num_words": "chosen (yw)",
            "yl_num_words": "rejected (yl)",
        })
        plt.figure()
        df_long["axis"] = df_long["axis"].map({
            "geographical location": "location",
            "education level": "education",
            "age": "age",
            "family marriage status": "family",
            "diet": "diet",
            "religion": "religion",
            "sports": "sports",
            "profession": "profession",
             "politics": "politics",
            "gender": "gender",
            "AI professors": "AI professors"
        })
        sns.boxplot(data=df_long[df_long.variable!="prompt"], x="axis", y="Length", hue="variable",
                    order=["location", "education", "age","family",  "diet", "religion", "sports", "profession", "politics", "gender", "AI professors"],
                    palette=PALETTE)
        plt.xticks(rotation=30)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{EXP_DIR}/box_{metric}_per_axis.png")
        plt.close()

        # # aggregate all
        # plt.figure()
        # sns.boxplot(data=df_long, x="variable", y="Length", hue="Question type", palette=PALETTE)
        # # plt.xticks(rotation=30)
        # plt.xlabel("")
        # plt.tight_layout()
        # plt.savefig(f"{EXP_DIR}/box_{metric}_stat.png")
        # plt.close()
        #
        # # yw yl diff
        # plt.figure()
        # df.rename(columns={"question_type": "Question type"}, inplace=True)
        # df["Length of chosen (yw) - rejected (yl)"] = df["yw_num_words"] - df["yl_num_words"]
        # sns.boxplot(data=df, x="Question type", y="Length of chosen (yw) - rejected (yl)",palette=PALETTE)
        # # plt.xticks(rotation=30)
        # plt.xlabel("")
        # plt.tight_layout()
        # plt.savefig(f"{EXP_DIR}/box_{metric}_diff_stat.png")
        # plt.close()

    # print to latex table
    for q_type in ["common", "personal"]:
        print("%"*20+ f" {q_type} " + "%"*20 )
        res_df = df[df.question_type==q_type].describe()
        print(res_df.to_latex(float_format="{:.0f}".format))

    return df


def persona_statistics():
    df_persona = pd.read_json("../data/pp-50-final/personas.json", lines=True, orient="records")
    fields = ["xyw_2s", "persona_xy_4s", "persona_xy_4s","persona_xy_4s_gpt4", "persona_gold_gpt4"]
    df_persona = df_persona[~df_persona.persona_xy_4s.isna()]
    for field in fields:
        df_persona[f"{field}_len"] = df_persona[field].apply(lambda s: len(s.split()))
    print("%" * 20 + f" persona statistics " + "%" * 20)
    print(df_persona[[f"{f}_len" for f in fields]].describe().to_latex(float_format="{:.0f}".format))

    # plot box-whisker plot
    plt.figure()
    df_long = pd.melt(df_persona, id_vars=['name'], value_vars=[f"{f}_len" for f in fields])
    df_long["variable"] = df_long.variable.map({
        "xyw_2s_len": "few-shot (n=2)",
        "persona_xy_4s_len": "persona",
        "persona_xy_4s_gpt4_len": "persona gpt4",
        "persona_gold_gpt4_len": "persona gold"
    })
    df_long.rename(columns={"value": "Prefix Length"}, inplace=True)
    sns.boxplot(data=df_long, y="Prefix Length", x="variable", palette=PALETTE)
    # plt.xticks(rotation=30)
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(f"{EXP_DIR}/box_prefix.png")
    plt.close()
    return df_persona


def persona_quality(exp_cols:Optional[List[str]]=None):
    df_persona = pd.read_json("../data/pp-50-final/personas.json", lines=True, orient="records")
    if exp_cols is None:
        # exp_cols = ["persona_xy_4s", "persona_xy_4s_seed1", "persona_xy_4s_seed2", "persona_xy_4s_seed3", "persona_xy_4s_seed4"]  #$persona_xy_4s_llama321b", "persona_xy_4s_llama323b", "persona_xy_4s_ministral8b"
        exp_cols=["persona_xy_4s","persona_xy_4s_llama321b", "persona_xy_4s_llama323b", "persona_xy_4s_ministral8b", "persona_xy_4s_gpt4"]
        # exp_cols = ["xyw_2s"]
    ref_cols = ["persona_gold_gpt4"]  # "persona_xy_4s_gpt4",

    rouge = evaluate.load('rouge')
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    ref_emb = model.encode(df_persona[ref_cols[0]], batch_size=4, show_progress_bar=True, normalize_embeddings=True)

    for exp_col in tqdm(exp_cols):
        metric_list = df_persona.apply(lambda row: rouge.compute(predictions=[row[exp_col]], references=[[row[c] for c in ref_cols]]), axis=1)
        for k, v in metric_list[0].items():
            df_persona[f"{exp_col}_{k}"] = [m[k] for m in metric_list]

            print(f"Mean {exp_col} {k}: {np.mean(df_persona[f'{exp_col}_{k}'])}")
            print(f"std {exp_col} {k}: {np.std(df_persona[f'{exp_col}_{k}'])}")
            df_persona[f"{exp_col}_{k}_rank"] = np.argsort(df_persona[f"{exp_col}_{k}"])

        exp_embs = model.encode(df_persona[exp_col], batch_size=4, show_progress_bar=True, normalize_embeddings=True)
        df_persona[f"{exp_col}_emb_sim"] = np.diag(exp_embs @ ref_emb.T)
        print(f"Mean {exp_col} emb sim: {np.mean(df_persona[f'{exp_col}_emb_sim'])}")
        print(f"std {exp_col} emb sim: {np.std(df_persona[f'{exp_col}_emb_sim'])}")
        df_persona[f"{exp_col}_emb_sim_rank"] = np.argsort(df_persona[f"{exp_col}_emb_sim"])

        df_persona[f"{exp_col}_len"] = df_persona[exp_col].apply(lambda s: len(s.split()))
        print(f"Mean {exp_col} length: {np.mean(df_persona[f'{exp_col}_len']):.3f}")
        print(f"Std {exp_col} length: {np.std(df_persona[f'{exp_col}_len']):.3f}")
    # aggregate metrics, calculate ranks, and see if anyone especially bad
    def sum_norm_col(c_list: List[str]):
        return np.mean([(df_persona[c] - df_persona[c].mean())/ df_persona[c].std() for c in c_list], axis=0)
    for exp_col in exp_cols:
        metric_cols = [f"{exp_col}_rouge1", f"{exp_col}_emb_sim"]
        df_persona[f"{exp_col}_comb_score"] = sum_norm_col(metric_cols)
        df_persona[f"{exp_col}_comb_rank"] = np.argsort(df_persona[f"{exp_col}_comb_score"])

    score_cols = [c for c in df_persona.columns if "_comb_score" in c]
    df_persona["avg_score"] = df_persona[score_cols].mean(axis=1)

    rank_cols = [c for c in df_persona.columns if "_comb_rank" in c]
    df_persona["avg_rank"] = df_persona[rank_cols].mean(axis=1)

    best_persona_names = df_persona.sort_values("avg_rank", ascending=True).name[:10]
    worst_persona_names = df_persona.sort_values("avg_rank", ascending=False).name[:10]
    # box plots, since names don't really tell a lot other than good at sports and bad at AI profs
    for attribute_category in []:
        sns.boxplot(df_persona, x=attribute_category, y="avg_rank")
    return df_persona[["name", *score_cols]]



def print_top_attributes_and_distribution(df: pd.DataFrame, plot=False):

    # print("%"*40)
    # print(df.describe().to_latex(float_format="{:.1f}".format))
    # print("%" * 40)
    # print()
    # print("%" * 40)

    def top_counts(col, text=True):
        col_flattened = []
        [col_flattened.extend(r.split("/")) for r in col]
        if text:
            return pd.Series.value_counts(pd.Series(col_flattened), normalize=True).reset_index().apply(lambda r: f"{r['index']}: {r.proportion*100:.0f}%", 1)
        else:
            return pd.Series.value_counts(pd.Series(col_flattened))

    agg_df = df[DEMOGRAPHIC_ATTRIBUTE_CATEGORIES].apply(partial(top_counts, text=True))

    cnt=0
    for attr, val_pct in agg_df.iloc[0].items():
        val, pct = val_pct.split(": ")
        pct = float(pct.replace("%", ""))
        if pct > 50 and val!= "NA":
            print(f"Strong correlation attribute: {attr=} {val_pct=}")
            cnt += 1
    print(f"total of {cnt} biasing attributes")

    if not plot:
        return
    # okay for small amount of attributes, but not able to format to multiple rows subplot
    # df[attr_cols].apply(partial(top_counts, text=False)).plot.pie(subplots=True, legend=False, figsize=(12,15), autopct='%.0f%%')

    n_rows, n_cols = 3,5
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(30,15))
    for i in range(n_rows*n_cols):
        row = i // n_cols
        col = i % n_cols
        if i < len(DEMOGRAPHIC_ATTRIBUTE_CATEGORIES):
            attr = DEMOGRAPHIC_ATTRIBUTE_CATEGORIES[i]
            series = agg_df[~agg_df[attr].isna()][attr].tolist()
            keys = [i.split(": ")[0] if idx < 8 else "" for idx, i in enumerate(series)]
            data = [float(i.split(": ")[1].replace("%", "")) for i in series]
            palette_color = sns.color_palette(PALETTE)
            axs[row, col].pie(data, labels=keys, colors=palette_color, autopct='%.0f%%')
            axs[row, col].set_title(attr, fontsize=24, fontweight='bold')
        else:
            # fig.delaxes(axs[row][col])
            # sns.displot(df, x="age", ax=axs[row, col])
            sns.histplot(df, x="age", ax=axs[row, col], palette=PALETTE)
            axs[row, col].set_title("age", fontsize=24, fontweight='bold')
            axs[row, col].spines['top'].set_visible(False)
            axs[row, col].spines['right'].set_visible(False)
            axs[row, col].set_xlabel('')

    plt.tight_layout()
    plt.savefig(f"{EXP_DIR}/demographics_pie.png")


def demographics_statistics():
    os.makedirs(EXP_DIR, exist_ok=True)
    df_persona = pd.read_json("../data/pp-50-final/personas.json", lines=True, orient="records")
    df = add_demographics_info(df_persona)
    # print("========= for all personas =========")
    # print_top_attributes_and_distribution(df, plot=True)

    for axis in ALL_AXES:
        print(f"========= for personas in {axis} =========")
        print_top_attributes_and_distribution(df[df.axis.apply(lambda a: axis in a)])


def add_demographics_info(df_persona):
    df = pd.read_csv("../data/pp-50-final/demographics.csv", index_col=False)
    df = df.merge(df_persona[["name", "axis"]], how="left", on="name")
    df["axis"] = df.axis.apply(lambda a: [s for s in a if s.split(": ")[0] in ALL_AXES])
    df["category"] = df.axis.apply(lambda a: [s.split(": ")[-1] for s in a])
    df["axis"] = df.axis.apply(lambda a: [s.split(": ")[0] for s in a])
    df["age"] = 2025 - df["birth year"]
    del df["birth year"]
    df.fillna("NA", inplace=True)
    return df


def common_axis_conflicting_preferences():
    df = pd.read_json("../data/pp-50-final/test_50p_200d_total_50r_tem2.0_top0.8_cot_filtered20m4k_yl-random_cot_annotated.json", lines=True, orient="records")
    df = df[df.question_type=="common"]
    df_avg = df.groupby("axis")["yw"].value_counts().groupby("axis").mean().T
    df_std = df.groupby("axis")["yw"].value_counts().groupby("axis").std().T
    # df_avg.rename(columns={0:"mean"}, inplace=True)
    # df_std.rename(columns={0: "std"}, inplace=True)
    df_res = pd.DataFrame({"mean":df_avg, "std":df_std}).T
    print(df_res.to_latex(float_format="{:.2f}".format))


def extract_root_verb_and_object(sentence, nlp):
    doc = nlp(sentence.strip())

    # Find the root verb and its associated noun object
    root_verb = None
    noun_object = None

    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ in ["VERB"]: #"AUX", "PRON"
            root_verb = token
            # Look for noun objects associated with the root verb
            for child in token.children:
                if child.dep_ in ["dobj", "obj", "nsubjpass"]: #expl
                    noun_object = child
                    break
            break

    return root_verb.text if root_verb else doc[0].text if len(doc)>0 else "null", \
           noun_object.text if noun_object else doc[1].text if len(doc)>1 else "null"


def prompt_lexical_diversity():
    df = pd.read_json("../data/pp-50-final/train_50p_200d_total_50r_tem2.0_top0.8_cot_filtered20m4k_yl-random_cot_annotated.json", lines=True, orient="records")
    df = df.drop_duplicates("prompt")
    nlp = spacy.load('en_core_web_sm')
    for i, row in tqdm(df.iterrows(), total=len(df)):
        root_verb, noun_object = extract_root_verb_and_object(row.prompt.strip(), nlp)
        df.loc[i, "verb"] = root_verb.lower()
        df.loc[i, "noun"] = noun_object.lower()
    # df = df[(df.verb!="null") & ()]
    df = df[df.verb.isin(df.verb.value_counts().head(20).index.tolist())]
    df = df[df.noun.isin(df.noun.value_counts().head(5).index.tolist())]
    fig = px.sunburst(df, path=['verb', 'noun'],)
    fig.show()

    return


def visualize_question_distribution():
    cache_path = "prompt_emb_distribution.csv"
    if not os.path.isfile(cache_path):
        data_path = f"../data/pp-50-final/train_50p_200d_total_50r_tem2.0_top0.8_cot_filtered20m4k_yl-random_cot_annotated.json"
        df = pd.read_json(data_path, lines=True, orient="records")
        df_persona = pd.read_json(f"../data/pp-50-final/personas.json", lines=True, orient="records")
        def get_axis(r):
            if r.question_type == "common":
                return r.axis
            else:
                axes_list = df_persona[df_persona.name == r["name"]].axis.tolist()[0]
                axes_left = [a.split(": ")[0] for a in axes_list if a.split(": ")[0] in ALL_AXES]
                return random.choice(axes_left)
        df["axis"] = df.apply(lambda r: get_axis(r),1)
        sent_model = SentenceTransformer("sentence-transformers/sentence-t5-xxl")
        embs = sent_model.encode(df.prompt, convert_to_tensor=True, show_progress_bar=True)
        embs = (embs / torch.sqrt((embs ** 2).sum(axis=1))[:, None]).cpu().numpy()
        tsne = TSNE(n_components=2, perplexity=30, verbose=1, random_state=10, n_iter=300)
        # Fit the t-SNE model to the data
        X_tsne = tsne.fit_transform(embs)
        df["tsne dim1"], df["tsne dim2"] = X_tsne[:, 0], X_tsne[:, 1]
        df.to_json(cache_path, lines=True, orient="records")
    else:
        df = pd.read_json(cache_path, lines=True, orient="records")

    df["question_type"] = df.question_type.map({"common": "divergent", "personal": "personal"})

    # Create a scatter plot of the data, colored by group
    plt.figure(figsize=(16, 10))
    # rndperm = np.random.permutation(df.shape[0])
    sns.scatterplot(
        x="tsne dim1", y="tsne dim2",
        # hue="name",
        # palette=sns.color_palette("hls", len(df.name.unique())),
        hue="axis",
        palette=sns.color_palette("hls", len(df.axis.unique())),
        data=df,  # df.loc[rndperm, :],
        style="question_type",
        legend="full",
        alpha=0.3
    )

    for i, row in df.sample(n=5, random_state=40).iterrows():
        # plt.text(row["tsne dim1"], row["tsne dim2"], f"{row.name}: {row.prompt}", ha='center', va='center')
        plt.annotate(f"{row.question_type} question from {row['name']} ({row.axis}):\n{row.prompt}", xy=(row["tsne dim1"], row["tsne dim2"]), xytext=(-20, 20),
                    textcoords='offset points', ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',
                                    color='red'))

    # Show the plot
    # plt.show()
    plt.axis('off')
    os.makedirs(EXP_DIR, exist_ok=True)
    plt.savefig(f"{EXP_DIR}/prompt_emb_tsne.png")


# def prompt_topics():
#     from bertopic import BERTopic
#     topic_model = BERTopic.load("MaartenGr/BERTopic_Wikipedia")
#     topic, prob = topic_model.transform("This is an incredible movie!")

def prompt_overlap():
    cache_path = "prompt_overlap_rouge.json"
    if not os.path.isfile(cache_path):
        train_path = f"../data/pp-50-final/train_50p_200d_total_50r_tem2.0_top0.8_cot_filtered20m4k_yl-random_cot_annotated.json"
        train_df = pd.read_json(train_path, lines=True, orient="records")
        eval_path = f"../data/pp-50-final/test_50p_200d_total_50r_tem2.0_top0.8_cot_filtered20m4k_yl-random_cot_annotated.json"
        eval_df = pd.read_json(eval_path, lines=True, orient="records")
        rouge = evaluate.load('rouge')
        # this will work as well, just don't know how much speed up there will be
        # train_df["eval_prompts"] = train_df.name.apply(lambda n: eval_df[eval_df.name==n].prompt.tolist())
        # metric = rouge.compute(predictions=train_df.prompt, references=train_df.eval_prompts)
        for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
            eval_prompts = eval_df[eval_df.name==row["name"]].prompt
            # aggregation over multiple reference here is max
            metric = rouge.compute(predictions=[row.prompt], references=[eval_prompts.tolist()])
            for k, v in metric.items():
                train_df.loc[i, f"{k}"] = v
        df_melt = pd.melt(train_df, id_vars=["name", "prompt"], value_vars=["rouge1", "rouge2", "rougeL", "rougeLsum"])
        df_melt.rename(columns={"variable": "Metric"}, inplace=True)
        df_melt.to_json(cache_path, lines=True, orient="records")

    else:
        df_melt = pd.read_json(cache_path, lines=True, orient="records")

    # plot aggregate len
    plt.figure()
    sns.histplot(df_melt, x="value", hue="Metric")
    plt.tight_layout()
    plt.savefig(f"{EXP_DIR}/train_test_prompt_similarity.png")
    plt.close()

    # for each person, plot boxplot
    plt.figure()
    df_melt = df_melt[df_melt.Metric=="rougeL"]
    df_melt.rename(columns={"value": "rougeL"}, inplace=True)
    sns.boxplot(df_melt, x="name", y="rougeL")
    plt.tight_layout()
    plt.savefig(f"{EXP_DIR}/train_test_prompt_similarity_per_person.png")
    plt.close()
    # pdb.set_trace()

if __name__ == "__main__":

    # get_string_len_statistics(
    #     # "../data/pp-200/train_10p_200d_total_50r_tem2.0_top0.8_cot_filtered20m4k_yl-random_cot_annotated.json",
    #     # "../data/pp-200/test_10p_200d_total_50r_tem2.0_top0.8_cot_filtered20m4k_yl-random_cot_annotated.json"
    #     "../data/pp-50-final/train_50p_200d_total_50r_tem2.0_top0.8_cot_filtered20m4k_yl-random_cot_annotated.json",
    #     "../data/pp-50-final/test_50p_200d_total_50r_tem2.0_top0.8_cot_filtered20m4k_yl-random_cot_annotated.json"
    # )
    # persona_statistics()
    persona_quality()
    # demographics_statistics()
    # prompt_lexical_diversity()  # not really good
    # common_axis_conflicting_preferences()
    # visualize_question_distribution()
    # prompt_overlap()