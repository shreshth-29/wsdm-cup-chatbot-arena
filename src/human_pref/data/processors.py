"""Prompt builder for the pairwise LLM judge.

This file overrides ``human_pref/data/processors.py`` from the public
``lmsys-modules-0805`` Kaggle dataset. The only change from the original is
the judge prompt (``PROMPT_PREFIX``): we rewrote it to ask the model to weigh
helpfulness, relevance, accuracy, depth, creativity and topic coverage, and to
explicitly ignore response length and assistant names.

The notebook copies the ``human_pref`` package from the Kaggle dataset and
then writes this file on top of it (``%%writefile human_pref/data/processors.py``).
"""
import json

import torch


class ProcessorPAB:
    # Earlier version of the judge prompt, kept for reference.
    # PROMPT_PREFIX = """You will be given two responses from two Assistants to the same Prompt.
    # Your task is to decide which response is better based on its relevance, helpfulness, and level of detail.
    # Begin your evaluation by comparing the two responses and provide a short explanation. Do not favor certain names of
    # the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly
    # following this format with no tie : "[[A]]" if assistant A is better, "[[B]]" if assistant B is better."""

    PROMPT_PREFIX = """Act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that
    follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, topic coverage
    and level of detail of their responses. Begin your evaluation by comparing the two responses and provide explanation and observations about each responses then provide reason why would you prefer certain
    response over other response. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation.
    Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation and observation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better."""

    # Another variant we tried, kept for reference.
    #     PROMPT_PREFIX = """
    #     Please act as an impartial judge and evaluate the quality of the responses provided by two
    #     AI assistants to the user question displayed below. You should choose the assistant that
    #     follows the user’s instructions and answers the user’s question better.
    #     Carefully analyze response from Assistant A and Assistant B in relation to the User's Query. Determine which response is more likely to be selected by a user based on the following criteria:
    #     - Completeness in addressing the query
    #     - Accuracy of information
    #     - Clarity and coherence
    #     - Conciseness vs appropriate detail
    #     - Helpful examples or explanations when needed
    #     - Professional yet engaging tone
    #     - Sound reasoning and logic
    #     - Format and presentation
    #     Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision.
    #     Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format:
    #     "[[A]]" if assistant A is better,
    #     "[[B]]" if assistant B is better.
    # """

    PROMPT_SUFFIX = "verdict is: [["

    LABEL_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]

    def __init__(self, tokenizer, max_length, support_system_role):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.support_system_role = support_system_role

    def build_conversation(self, prompts, responses_a, responses_b):
        head = "<|The Start of Conversation between a User and two Assistants|>"
        tail = "<|The End of Conversation between a User and two Assistants|>\n"
        parts = []
        for prompt, response_a, response_b in zip(prompts, responses_a, responses_b):
            if prompt is None:
                prompt = "null"
            if response_a is None:
                response_a = "null"
            if response_b is None:
                response_b = "null"
            parts.append(
                f"\n### User:\n{prompt}\n\n### Assistant A:\n{response_a}\n\n### Assistant B:\n{response_b}\n"
            )
        text = "".join(parts)
        input_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        ).input_ids

        truncated_text = self.tokenizer.decode(input_ids)
        return head + truncated_text + tail

    def build_input(self, data):
        conversation = self.build_conversation(
            [data["prompt"]],
            [data["response_a"]],
            [data["response_b"]],
        )
        if self.support_system_role:
            messages = [
                {"role": "system", "content": self.PROMPT_PREFIX},
                {"role": "user", "content": conversation},
            ]
        else:
            messages = [
                {"role": "user", "content": f"{self.PROMPT_PREFIX}\n{conversation}"},
            ]
        input_text = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            + self.PROMPT_SUFFIX
        )
        input_ids = self.tokenizer(
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]
        label = torch.tensor([data[col] for col in self.LABEL_COLS]).float()
        return dict(
            input_ids=input_ids,
            input_text=input_text,
            label=label,
        )
