"""Blend the Gemma-2 judge with the GBDT ensemble and write submission.csv.

Gemma probabilities are 3-way (a, b, tie). The tie column is dropped and the
remaining two are renormalised, then blended 80/20 with the GBDT ensemble.
"""
import numpy as np
import pandas as pd

LLM_WEIGHT = 0.8
GBDT_WEIGHT = 0.2

df = pd.read_parquet("test.parquet")

preds = np.load("prob_m0.npy")
preds = preds[:, 0:2]
preds = preds / preds.sum(axis=1).reshape(-1, 1)

gbdt = np.load("prob_gbdt.npy")
prob_a, prob_b = gbdt[:, 0], gbdt[:, 1]

winners = []
for i in range(len(preds)):
    score_a = LLM_WEIGHT * preds[i, 0] + GBDT_WEIGHT * prob_a[i]
    score_b = LLM_WEIGHT * preds[i, 1] + GBDT_WEIGHT * prob_b[i]
    winners.append("model_a" if score_a > score_b else "model_b")

sub = pd.DataFrame({"id": df["id"], "winner": winners})
sub.to_csv("submission.csv", index=False)
print(sub.head())
