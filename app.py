# app.py
import os, json, boto3
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

S3_BUCKET = os.environ.get("DATA_BUCKET")
S3_PREFIX = os.environ.get("DATA_PREFIX", "movielens-100k/ml-100k")  # dossier avec u.data/u.item/u.genre
s3 = boto3.client("s3")

# ====== Chargement au cold start ======
def _read_s3_csv(key, **read_csv_kwargs):
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_csv(obj["Body"], **read_csv_kwargs)

def load_ml100k():
    ratings = _read_s3_csv(f"{S3_PREFIX}/u.data",
                           sep="\t", header=None, names=["user_id","item_id","rating","timestamp"])
    # u.item contient les 19 flags genre; on récupère au moins le titre
    items = _read_s3_csv(f"{S3_PREFIX}/u.item",
                         sep="|", header=None, encoding="latin-1",
                         names=["item_id","title","release_date","video_release_date","imdb_url",
                                "unknown","Action","Adventure","Animation","Children","Comedy","Crime",
                                "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
                                "Mystery","Romance","Sci-Fi","Thriller","War","Western"],
                         usecols=["item_id","title"])
    return ratings, items

def build_user_item_matrix(ratings: pd.DataFrame):
    return ratings.pivot(index="user_id", columns="item_id", values="rating").fillna(0.0)

def recommend(user_id: int, ratings: pd.DataFrame, items: pd.DataFrame, topk=10, alpha=0.5):
    mat = build_user_item_matrix(ratings)
    if user_id not in mat.index:
        raise ValueError(f"user_id {user_id} not found (range {mat.index.min()}–{mat.index.max()})")

    sims = cosine_similarity([mat.loc[user_id]], mat)[0]
    sim_users = pd.Series(sims, index=mat.index)

    collab_scores = mat.T.dot(sim_users)                     # item_id index
    popularity_scores = ratings.groupby("item_id")["rating"].mean()

    def norm(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-8)

    collab_scores = norm(collab_scores).fillna(0.0)
    popularity_scores = norm(popularity_scores).reindex(collab_scores.index).fillna(0.0)

    hybrid = alpha * collab_scores + (1 - alpha) * popularity_scores

    seen = ratings.loc[ratings["user_id"] == user_id, "item_id"].unique()
    hybrid = hybrid.drop(labels=seen, errors="ignore")

    top = hybrid.sort_values(ascending=False).head(topk)
    recos = (pd.DataFrame({"item_id": top.index, "score": top.values})
             .merge(items, on="item_id", how="left")
             .sort_values("score", ascending=False)
             .reset_index(drop=True))
    return recos

# Cold start cache
_RATINGS, _ITEMS = load_ml100k()

def handler(event, context):
    try:
        params = event.get("queryStringParameters") or {}
        user = int(params.get("user", "42"))
        topk = int(params.get("topk", "10"))
        alpha = float(params.get("alpha", "0.5"))

        recos = recommend(user, _RATINGS, _ITEMS, topk=topk, alpha=alpha)
        body = recos.to_dict(orient="records")

        return {
            "statusCode": 200,
            "headers": {"content-type": "application/json"},
            "body": json.dumps(body)
        }
    except Exception as e:
        return {
            "statusCode": 400,
            "headers": {"content-type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }

