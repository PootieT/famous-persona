import os
import pdb
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    print("better not be using pycharm!")
import matplotlib.pyplot as plt


def cache_persona_embeddings(model_name="all-mpnet-base-v2"):
    df = pd.read_json(PERSONA_DF_PATH, orient="records", lines=True)
    model = SentenceTransformer(model_name)
    base_columns = ["name", "shuffle_name", "axis", "description"]
    for c in tqdm(df.columns, total=len(df.columns)):
        if c in base_columns:
            continue

        indices = df.loc[~df[c].isna()].index
        embs = model.encode(df.loc[indices, c].tolist())
        df.loc[indices, c] = [list(row) for row in embs]
    df.to_pickle(Path(__file__).parent.joinpath(f"{DATA}_personas_{model_name}.pkl"))


def visualize_persona_emb_across_methods(cache_path: str):
    exp_dir = f"{DATA}/persona_embs"
    os.makedirs(exp_dir, exist_ok=True)
    df = pd.read_pickle(cache_path)
    methods = ["persona_gold_gpt4", "xyw_2s", "persona_xy_4s", "persona_xy_4s_gpt4", "persona_xy_8s_gpt4"]
    df = df[~df.name.isin(["Yoshua Bengio", "Timnit Gebru"])]
    for method in methods:
        sub_df = df[~df[method].isna()]
        embs = sub_df[method].tolist()
        names = sub_df["name"].tolist()
        if f"{method}_seed1" in sub_df.columns:
            sub_df = df[~df[f"{method}_seed1"].isna()]
            embs.extend(sub_df[f"{method}_seed1"])
            names.extend(sub_df["name"])

        # calculate intra-person similarity and inter-person similarity
        embs_norm = embs / np.linalg.norm(embs, axis=1).reshape(-1, 1)
        embs_rows = embs_norm.reshape((~df[method].isna()).sum(), 2, -1)
        sim = np.dot(embs_rows[:,0], embs_rows[:, 1].T)
        intra_sim = np.diagonal(sim).mean()
        inter_sim = (sim.sum() - np.diagonal(sim).sum()) / (len(sim)**2 - len(sim))
        print(f"{method=}, intra_sim={intra_sim:.2f}, inter_sim={inter_sim:.2f}, gap={intra_sim-inter_sim:.2f}")

        # PCA and plot
        pca = PCA(n_components=2)
        pc = pca.fit_transform(embs)
        pc_df = pd.DataFrame({
            "PC1": pc[:, 0],
            "PC2": pc[:, 1],
            "name": names
        })
        plt.figure()
        sns.scatterplot(pc_df, x="PC1", y="PC2", hue="name")
        plt.title(f"Prefix = {method}, intra_sim={intra_sim:.2f}, inter_sim={inter_sim:.2f}, gap={intra_sim-inter_sim:.2f}")
        plt.savefig(f"{exp_dir}/pca_{method}.png")
        plt.close()


if __name__ == "__main__":
    DATA = "pp-200"
    PERSONA_DF_PATH = Path(__file__).parents[1].joinpath("data").joinpath(DATA).joinpath("personas.json")
    # cache_persona_embeddings()
    visualize_persona_emb_across_methods("pp-200_personas_all-mpnet-base-v2.pkl")

