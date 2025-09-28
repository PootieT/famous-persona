import glob
import json
import os
import pdb
from typing import List, Dict, Optional
import random

from sklearn.metrics import roc_curve, auc, roc_auc_score
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import scipy as sp
from sklearn.preprocessing import normalize
import matplotlib

try:
    matplotlib.use('TkAgg')
except:
    print("better not be using pycharm!")
import matplotlib.pyplot as plt


def aggregate():
    all_res = []
    test_df = pd.read_json(test_data_path, lines=True, orient="records")
    for eval_path in tqdm(glob.glob(f"{dump_dir}/*/*/*scores_dict.json")):
        if "one_model_for_all_cv_" not in eval_path or "_axis" in eval_path:
            continue
        res_df = pd.read_json(eval_path, lines=True, orient="records")
        assert len(test_df) == len(res_df)
        if "name" in res_df:
            del res_df["name"]
        res_df["conversation_id"] = test_df["conversation_id"]
        res_df.rename(columns={"results": "accuracy"}, inplace=True)
        if "ref_sft" in eval_path:
            res_df.drop(columns=["subset"], inplace=True)
            res_df["ref_free"] = False
        else:
            res_df.drop(columns=["subset", "pol_chosen_score", "pol_rejected_score", "ref_chosen_score", "ref_rejected_score"], inplace=True)
            res_df["ref_free"] = True
        group_dir, exp_dir = eval_path.split("/")[-3], eval_path.split("/")[-2]
        res_df["cv"] = int(exp_dir[exp_dir.find("_cv")+3:])
        res_df["specific_method"] = "_".join(exp_dir.split("_")[1:-1])
        res_df["group"] = group_dir
        all_res.append(res_df)
    all_res_df = pd.concat(all_res).reset_index().drop(columns=["index"])
    all_res_df.to_csv(f"{dump_dir}/aggregated_results_single_eval_files.csv", index=False)

    pd.pivot_table(all_res_df, values=col_map[f"{ref}_{task}"], index=['model', 'seen_during_train'], columns=['Prefix'], aggfunc="mean")
    return all_res_df


if __name__ == "__main__":
    dump_dir = "../dump/pp-50-final"
    aggregate()