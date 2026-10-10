"""Add dummy label columns to the competition test set.

The LMSYSDataset loader expects the three label columns to exist even at
inference time, so we fill them with a placeholder (model_a = 1).
"""
import pandas as pd

INPUT = "/kaggle/input/wsdm-cup-multilingual-chatbot-arena/test.parquet"

df = pd.read_parquet(INPUT)
df["winner_model_a"] = 1
df["winner_model_b"] = 0
df["winner_tie"] = 0
df.to_parquet("test.parquet", index=False)

# Swapped copy, used when running the Llama judge on the (B, A) ordering.
# df["response_a"], df["response_b"] = df["response_b"], df["response_a"]
# df.to_parquet("test_swap.parquet", index=False)
