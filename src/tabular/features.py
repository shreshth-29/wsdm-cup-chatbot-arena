"""Hand-crafted text features and TF-IDF features for the GBDT models.

Two kinds of features:
  * ``compute_feats`` counts characters and patterns in the prompt and both
    responses (length, punctuation, brackets, formatting characters, digit
    and case ratios, CJK ratio, prompt/response word overlap, and so on).
  * ``transform_features`` stacks TF-IDF vectors (300 features, shared
    vocabulary for prompt and responses), the absolute TF-IDF difference
    between the two responses, length ratios, and the counting features.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf(train: pd.DataFrame, max_features: int = 300) -> TfidfVectorizer:
    """Fit one vectorizer on prompts and both responses together."""
    tfidf = TfidfVectorizer(max_features=max_features)
    all_text = pd.concat([train["prompt"], train["response_a"], train["response_b"]])
    tfidf.fit(all_text)
    return tfidf


def compute_feats(dataf: pd.DataFrame) -> pd.DataFrame:
    df = dataf.copy()
    feats = []
    for col in ["response_a", "response_b", "prompt"]:
        # response length is a key factor when choosing between two responses
        df[f"{col}_len"] = df[f"{col}"].str.len()

        # character counting features
        df[f"{col}_spaces"] = df[f"{col}"].str.count(r"\s")
        df[f"{col}_punct"] = df[f"{col}"].str.count(r",|\.|!")
        df[f"{col}_question_mark"] = df[f"{col}"].str.count(r"\?")
        df[f"{col}_quot"] = df[f"{col}"].str.count("'|\"")
        df[f"{col}_formatting_chars"] = df[f"{col}"].str.count(r"\*|\_")
        df[f"{col}_math_chars"] = df[f"{col}"].str.count(r"\-|\+|\=")
        df[f"{col}_curly_open"] = df[f"{col}"].str.count(r"\{")
        df[f"{col}_curly_close"] = df[f"{col}"].str.count("}")
        df[f"{col}_round_open"] = df[f"{col}"].str.count(r"\(")
        df[f"{col}_round_close"] = df[f"{col}"].str.count(r"\)")
        df[f"{col}_accent_chars"] = df[f"{col}"].str.count("è|ò|à|ù|é|ì")
        df[f"{col}_special_chars"] = df[f"{col}"].str.count(r"\W")
        df[f"{col}_digits"] = df[f"{col}"].str.count(r"\d") / df[f"{col}_len"]
        df[f"{col}_lower"] = df[f"{col}"].str.count("[a-z]").astype("float32") / df[f"{col}_len"]
        df[f"{col}_upper"] = df[f"{col}"].str.count("[A-Z]").astype("float32") / df[f"{col}_len"]
        df[f"{col}_chinese"] = df[f"{col}"].str.count(r"[一-鿿]+").astype("float32") / df[f"{col}_len"]
        df[f"{col}_tild"] = df[f"{col}"].str.count("~") > 0
        df[f"{col}_letters"] = df[f"{col}"].str.count("[a-zA-Z]")

        # how balanced curly and round brackets are (unbalanced often means truncated code)
        df[f"{col}_round_balance"] = df[f"{col}_round_open"] - df[f"{col}_round_close"]
        df[f"{col}_curly_balance"] = df[f"{col}_curly_open"] - df[f"{col}_curly_close"]

        # whether the text mentions json / yaml (structured output requests)
        df[f"{col}_json"] = df[f"{col}"].str.lower().str.count("json")
        df[f"{col}_yaml"] = df[f"{col}"].str.lower().str.count("yaml")

        feats = feats + [
            f"{col}_len", f"{col}_spaces", f"{col}_punct", f"{col}_question_mark", f"{col}_quot",
            f"{col}_formatting_chars", f"{col}_math_chars", f"{col}_curly_open", f"{col}_curly_close", f"{col}_round_open",
            f"{col}_round_close", f"{col}_accent_chars", f"{col}_special_chars", f"{col}_digits", f"{col}_lower",
            f"{col}_upper", f"{col}_chinese", f"{col}_tild", f"{col}_letters",
        ]

    def _overlap(prompt: str, response: str) -> float:
        p = set(prompt.split())
        return len(p.intersection(set(response.split()))) / len(p) if len(p) > 0 else 0

    df["response_a_prompt_overlap"] = df.apply(lambda x: _overlap(x["prompt"], x["response_a"]), axis=1)
    df["response_b_prompt_overlap"] = df.apply(lambda x: _overlap(x["prompt"], x["response_b"]), axis=1)
    feats = feats + ["response_a_prompt_overlap", "response_b_prompt_overlap"]
    return df[feats]


def transform_features(df: pd.DataFrame, vectorizer: TfidfVectorizer) -> np.ndarray:
    """TF-IDF blocks, TF-IDF difference, length ratios and counting features, stacked."""
    prompt_features = vectorizer.transform(df["prompt"])
    response_a_features = vectorizer.transform(df["response_a"])
    response_b_features = vectorizer.transform(df["response_b"])
    df["prompt_len"] = df["prompt"].apply(len)
    df["response_a_len"] = df["response_a"].apply(len)
    df["response_b_len"] = df["response_b"].apply(len)
    df["response_a_len/prompt_len"] = df["response_a_len"] / df["prompt_len"]
    df["response_b_len/prompt_len"] = df["response_b_len"] / df["prompt_len"]
    df["response_b_len/response_a_len"] = df["response_b_len"] / df["response_a_len"]

    return np.hstack([
        prompt_features.toarray(),
        response_a_features.toarray(),
        response_b_features.toarray(),
        np.abs(response_a_features.toarray() - response_b_features.toarray()),
        df[["prompt_len", "response_a_len", "response_b_len", "response_a_len/prompt_len",
            "response_b_len/prompt_len", "response_b_len/response_a_len"]].to_numpy(),
        compute_feats(df).to_numpy(),
    ])
