import argparse
import pdb
from pathlib import Path
import numpy as np
import pandas as pd

from generate_personal_preference_dataset import format_few_shot_prompt


def arg_parser():
    """
    Collect sampled generations into single dataset file in the same format as original dataset
    """
    args = argparse.ArgumentParser()
    args.add_argument("--name", type=str, required=True,
                      help="the name of the persona to get")
    args.add_argument("--field", type=str, default=None,
                      help="Path to augmented dataset.json from which the samples are generated")
    args.add_argument("--train_path", type=str, default=Path(__file__).parent.joinpath("train_10p_200d_total_50r_tem2.0_top0.8_cot_filtered20m4k_yl-random_cot_annotated.json"))

    return args


def normalize(s):
    return s.lower().replace(" ","").replace("_","")


def get_prefix(args):
    if args.name.lower == "no_prefix" or args.field.lower == "no_prefix":
        print("")
    else:
        df = pd.read_json(Path(__file__).parent.joinpath("personas.json"), lines=True, orient="records")
        # if args.field.startswith("shuffle_"):
        #     args.field = args.field.replace("shuffle_", "")
        #     # note this shuffle is only for inference perturbation purpose, if model is trained
        #     # on shuffled data, then eval should be done on that particular shuffled order
        #     df.loc[~df[args.field].isna(), "name"] = np.random.permutation(df.loc[~df[args.field].isna(), "name"])
        if args.field in df.columns:
            prefix = df[df.name.apply(lambda x: normalize(x)) == normalize(args.name)][args.field].tolist()[0]
            print(prefix)
        else:
            assert "xyw" in args.field and "_idx" in args.field
            train_df = pd.read_json(args.train_path, lines=True, orient="records")
            data_index = int(args.field[args.field.find("_idx")+4:])
            train_name_df = train_df[train_df.name.apply(lambda x: normalize(x)) == normalize(args.name)]
            prefix_name, prompt = format_few_shot_prompt(train_name_df, 1, include_yl="yl" in args.field, data_indices=[data_index])
            assert prefix_name == args.field
            print(prompt)


if __name__ == "__main__":
    np.random.seed(42)
    args = arg_parser().parse_args()
    get_prefix(args)
