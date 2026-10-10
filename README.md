# repo for analysis (WSDM Cup, Nov 2024 to Feb 2025)

Archive of our team's final submission to the [WSDM Cup: Multilingual Chatbot Arena](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena) on Kaggle. The competition ran from 18 Nov 2024 to 3 Feb 2025. The task: given a user prompt and two chatbot responses, predict which response the human voter preferred, across many languages. Metric is accuracy.

This repository was created in October 2026 to keep the code in one place. The code itself was written during the competition (Jan to Feb 2025) and lived in a single Kaggle notebook. The original notebook is kept untouched in `notebooks/`, and the same code is split into readable modules under `src/`. Nothing in `src/` changes the logic of the submission.

## Approach

The submission blends two very different models.

**1. Fine-tuned LLM judge (Gemma-2 9B).** A Gemma-2 9B sequence-classification model fine-tuned for pairwise preference on LMSYS Chatbot Arena data. The prompt and both responses are packed into a single judge prompt ("Act as an impartial judge...") that asks the model to ignore length and position, and the model outputs three logits: A wins, B wins, tie. Inference runs on two Kaggle T4 GPUs with layer-wise pipeline parallelism (first 21 layers on GPU 0, last 21 on GPU 1), fp16, with variable-length packed batches so no padding is wasted. The judge prompt is our own; the modeling code and checkpoint come from a public solution to the earlier LMSYS Chatbot Arena competition (see Dependencies).

**2. GBDT ensemble on text statistics.** XGBoost, LightGBM and CatBoost trained on 1,200 TF-IDF features (prompt, response A, response B, and the absolute A minus B difference) plus about 65 hand-made features: lengths and length ratios, punctuation and bracket counts, bracket balance, formatting characters, digit, case and CJK ratios, prompt-response word overlap, and mentions of json or yaml. The training set is the competition data plus Arena-Hard-Auto pairs, doubled by swapping the two responses and flipping the label so the trees cannot learn position bias. Test-time augmentation averages predictions over both orderings.

**Blend.** The tie column of the LLM output is dropped and the remaining two probabilities renormalised, then combined 80/20 with the GBDT ensemble. Argmax gives the winner.

A Llama-3 8B judge was also run on the swapped ordering during experiments; it is in the repo but was not part of the final blend.

## Layout

```
notebooks/baseline-v3-chatbot-arena.ipynb   original Kaggle notebook, as submitted
src/
  prepare_test_file.py                      adds dummy label columns the dataset loader expects
  human_pref/data/processors.py             our judge prompt (overrides the file from lmsys-modules-0805)
  llm_inference/predict_gemma2_9b.py        Gemma-2 9B two-GPU pipelined inference -> prob_m0.npy
  llm_inference/predict_llama3_8b.py        Llama-3 8B counterpart (experimental, not in final blend)
  tabular/features.py                       TF-IDF and hand-made text features
  tabular/train_gbdt.py                     XGB + LGBM + CatBoost with swap augmentation -> prob_gbdt.npy
  make_submission.py                        80/20 blend -> submission.csv
```

## Running it

The pipeline was built to run inside a Kaggle notebook with two T4s and the input datasets mounted under `/kaggle/input/`. Paths in the scripts reflect that. Order:

```
python src/prepare_test_file.py
python src/llm_inference/predict_gemma2_9b.py      # needs human_pref package and checkpoint, see below
python src/tabular/train_gbdt.py
python src/make_submission.py
```

The LLM step takes about 6 minutes for the public test set on two T4s. The GBDT step takes under a minute on GPU.

## Dependencies

Python packages are in `requirements.txt`. The LLM inference additionally needs three things that are not in this repo because they are large and not ours:

* `human_pref`: the modeling and data-loading package (custom Gemma-2 and Llama classes with `forward_part1` / `forward_part2` for pipeline parallelism, varlen collators, dataset class). Kaggle dataset `lmsys-modules-0805`. Copy it next to `src/` and drop our `processors.py` on top of `human_pref/data/processors.py`.
* `lmsys-checkpoints-0-0805` (Gemma-2 9B) and `lmsys-checkpoints-3-0805` (Llama-3 8B): the fine-tuned weights, also Kaggle datasets.
* `lmsys-packages`: prebuilt wheels for `triton==2.2.0` and `xformers` matching the Kaggle Python 3.10 / CUDA image at the time.

Both the package and the checkpoints come from a publicly shared solution to the LMSYS Chatbot Arena Human Preference competition (Kaggle, 2024). Credit for that code and those weights goes to its authors.

## Data

* Competition data: `train.parquet`, `test.parquet` from the Kaggle competition page.
* Extra training pairs for the GBDT models: Arena-Hard-Auto GPT-4 evaluations (`arena-hard-auto-dataset` on Kaggle), ties removed.

## License

MIT for the code in `src/` and `notebooks/`. Third-party code and weights listed under Dependencies keep their own licenses.
