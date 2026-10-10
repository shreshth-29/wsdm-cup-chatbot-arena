"""Train the XGBoost / LightGBM / CatBoost ensemble on TF-IDF and text features.

Steps:
  1. Load the competition train set, add the Arena-Hard-Auto pairs (ties
     dropped), dedupe, shuffle.
  2. Stratified 85/15 split. The train half is doubled by swapping
     response_a and response_b and flipping the label, which removes
     position bias from the tree models and doubles the data.
  3. Fit TF-IDF on train text, build features for train, validation,
     test and swapped test.
  4. Fit the three GBDT models with early stopping on the validation split.
  5. Average the three models, then average the (A, B) and (B, A) orderings
     of the test set (test-time augmentation). Saves ``prob_gbdt.npy`` with
     columns [p(model_a), p(model_b)].
"""
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from features import fit_tfidf, transform_features

warnings.simplefilter("ignore")

DATA = "/kaggle/input/wsdm-cup-multilingual-chatbot-arena"
ARENA_HARD = "/kaggle/input/arena-hard-auto-dataset/gpt_4.11_eval_model_arena-hard-auto.parquet"


def load_data():
    train = pd.read_parquet(f"{DATA}/train.parquet")
    test = pd.read_parquet(f"{DATA}/test.parquet")

    train = train[["prompt", "response_a", "response_b", "winner"]]
    extra = pd.read_parquet(ARENA_HARD)
    extra = extra[extra["winner"] != "tie"]
    del extra["question_id"]
    train = pd.concat([train, extra])
    train = train.drop_duplicates()
    train = train.dropna()
    train = train.sample(frac=1).reset_index(drop=True)
    return train, test


def swap(df: pd.DataFrame, flip_label: bool) -> pd.DataFrame:
    out = df.copy()
    out["response_a"], out["response_b"] = out["response_b"], out["response_a"]
    if flip_label:
        out["winner"] = out["winner"].apply(lambda x: "model_a" if "b" in x else "model_b")
    return out


def main():
    train, test = load_data()
    train, valid = train_test_split(train, test_size=0.15, stratify=train["winner"], random_state=42)

    train_inv = swap(train, flip_label=True)
    test_inv = swap(test, flip_label=False)

    tfidf = fit_tfidf(train)

    train = pd.concat([train, train_inv])
    train_features = transform_features(train, tfidf)
    val_features = transform_features(valid, tfidf)
    test_features = transform_features(test, tfidf)
    test_inv_features = transform_features(test_inv, tfidf)
    y_train = (train["winner"] == "model_a").astype("int")
    y_val = (valid["winner"] == "model_a").astype("int")

    xgb_params = {
        "tree_method": "auto",
        "device": "cuda",
        "n_estimators": 2000,
        "early_stopping_rounds": 100,
        "eval_metric": "logloss",
        "objective": "binary:logistic",
        "lambda": 6.4786798596508,
        "alpha": 0.030943844008921396,
        "colsample_bytree": 0.5,
        "subsample": 0.8,
        "learning_rate": 0.018,
        "max_depth": 9,
        "random_state": 2024,
        "min_child_weight": 207,
    }
    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(train_features, y_train, eval_set=[(val_features, y_val)], verbose=False)

    light_model = LGBMClassifier(n_estimators=1000, learning_rate=0.1, early_stopping_rounds=15)
    light_model.fit(train_features, y_train, eval_set=[(val_features, y_val)], eval_metric="binary_logloss")

    catboost_params = {"depth": 8, "iterations": 500, "l2_leaf_reg": 3, "learning_rate": 0.1}
    catboost_model = CatBoostClassifier(**catboost_params, random_state=42, task_type="GPU")
    catboost_model.fit(train_features, y_train, eval_set=[(val_features, y_val)], early_stopping_rounds=50, verbose=0)

    models = [xgb_model, light_model, catboost_model]
    a = sum(m.predict_proba(test_features) for m in models) / 3
    a_inv = sum(m.predict_proba(test_inv_features) for m in models) / 3

    # TTA over the two orderings: column 1 is p(model_a) for the original
    # order, column 0 is p(model_a) for the swapped order.
    prob_a = (a[:, 1] + a_inv[:, 0]) / 2
    prob_b = (a[:, 0] + a_inv[:, 1]) / 2
    np.save("prob_gbdt.npy", np.stack([prob_a, prob_b], axis=1))


if __name__ == "__main__":
    main()
