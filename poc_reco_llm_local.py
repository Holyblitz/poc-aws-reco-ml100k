#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# =============================
# Charger le dataset ML-100k
# =============================
def load_ml100k(base_path="/home/romain/port_folio/port_folio_ds/poc_aws/data/movielens100k/ml-100k"):
    ratings = pd.read_csv(
        os.path.join(base_path, "u.data"),
        sep="\t", names=["user_id","item_id","rating","timestamp"]
    )
    items = pd.read_csv(
        os.path.join(base_path, "u.item"),
        sep="|", encoding="latin-1",
        names=["item_id","title","release_date","video_release_date",
               "imdb_url","unknown","Action","Adventure","Animation",
               "Children","Comedy","Crime","Documentary","Drama","Fantasy",
               "Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi",
               "Thriller","War","Western"],
        usecols=["item_id","title"]
    )
    return ratings, items

# =============================
# Fonctions de reco
# =============================
def build_user_item_matrix(ratings: pd.DataFrame):
    users = np.sort(ratings["user_id"].unique())
    items = np.sort(ratings["item_id"].unique())
    mat = pd.DataFrame(0.0, index=users, columns=items)
    for _, row in ratings.iterrows():
        mat.loc[row["user_id"], row["item_id"]] = row["rating"]
    return mat

def recommend(user_id: int, ratings: pd.DataFrame, items: pd.DataFrame, topk=5, alpha=0.5):
    mat = build_user_item_matrix(ratings)
    if user_id not in mat.index:
        raise ValueError(f"Utilisateur {user_id} inconnu")

    sims = cosine_similarity([mat.loc[user_id]], mat)[0]
    sim_users = pd.Series(sims, index=mat.index)

    collab_scores = mat.T.dot(sim_users)
    popularity_scores = ratings.groupby("item_id")["rating"].mean()

    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-8)

    collab_scores = norm(collab_scores)
    popularity_scores = norm(popularity_scores)

    hybrid = alpha * collab_scores + (1 - alpha) * popularity_scores
    seen = ratings.loc[ratings["user_id"] == user_id, "item_id"].unique()
    hybrid = hybrid.drop(labels=seen, errors="ignore")

    top_items = hybrid.sort_values(ascending=False).head(topk)

    recos = (pd.DataFrame({"item_id": top_items.index, "score": top_items.values})
             .merge(items, on="item_id", how="left")
             .sort_values("score", ascending=False)
             .reset_index(drop=True))
    return recos

# =============================
# CLI
# =============================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, required=True, help="ID utilisateur (1–943)")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    ratings, items = load_ml100k()
    recos = recommend(args.user, ratings, items, args.topk, args.alpha)

    print(f"=== Recommandations pour l’utilisateur {args.user} ===")
    for _, row in recos.iterrows():
        print(f"- {row['title']} — score {row['score']:.3f}")

    os.makedirs("outputs", exist_ok=True)
    out_path = f"outputs/ml100k_recos_user_{args.user}.csv"
    recos.to_csv(out_path, index=False)
    print(f"\nCSV écrit: {out_path}")


