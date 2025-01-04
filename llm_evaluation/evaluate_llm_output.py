from asyncio import streams
from dataclasses import dataclass, field
from lzma import MODE_FAST
from math import e
import os
import json
import pickle
import random
import time
import typing as t
from typing import Any, Dict, List, Optional

import groq
from groq import Groq
import numpy as np
import pandas as pd
from tqdm import tqdm

from datasets import Dataset, load_dataset
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase


os.environ["GROQ_API_KEY"] = ""
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
os.environ["ERROR_REPORTING"] = "NO"

N_ITEMS = 20
CRITIC_MODEL = "llama3-70b-8192"

np.random.seed(42)


def save_list(items, file_path):
    try:
        with open(file_path, "wb") as f:
            pickle.dump(items, f)
    except Exception as e:
        print(f"An error occured while saving the list of objects to file {e}")

def load_list(file_path):
    try:
        with open(file_path, "rb") as f:
            items = pickle.load(f)
        return items
    except Exception as e:
        print(f"An error occured while loading list of objects from file {e}")
        return None


dataset = load_dataset("virattt/financial-qa-10k")

sample = dataset["train"].shuffle().select(range(N_ITEMS))

client = Groq(api_key = os.environ.get("GROQ_API_KEY"))

def predict(prompt: str, model: str, client: Groq = client):
    chat_completion = client.chat_completions.create(
        messages = [
            {
                "role": "user", "content": prompt,
            }
        ],
        model = model,
    )
    return chat_completion.choices[0].message.content


def format_prompt(question: str, context: str):
    return f"""
        Use the following context:
        ```
        {context}
        ```
        to answer the question:
        ```
        {question}
        ```

        Your answer must be accurate!
        Answer:
    """

@dataclass
class QuestionAnswer:
    question: str
    answer: str
    true_answer: str
    context: str

def extract_predictions(dataset: Dataset, model:str) -> List[QuestionAnswer]:
    return [
        QuestionAnswer(
            question=item["question"],
            answer=predict(format_prompt(item["question"], item["context"], model)),
            true_answer=item["answer"],
            context=item["context"]
        )
        for item in tqdm(dataset)
    ]

models = ["gemma-7b-it", "llama3-8b-8192", "mixtral-8x7b-32768"]

predictions = {}
for model in models:
    predictions[model] = extract_predictions(sample, model=model)

save_list(predictions, "predictions.pkl")

## Simple Evaluation
eval_prompt = """
Consider the question: {question}
add answer: {answer}
based on the context: {context}
compare with true answer: {true_answer}

Score how correct the response is on a scale from 0 to 10.
Respond with the integer number only.
"""

prediction = predictions["gemma-7b-it"][0]
prediction.__dict__

predict(eval_prompt.format(**prediction), model=CRITIC_MODEL)

scores = {}
for model,preds in predictions.items():
    model_scores = []
    for pred in preds:
        score = predict(eval_prompt.format(**pred), model=CRITIC_MODEL)
        model_scores.append(int(score))

        # add a delay to avoid rate limit
        time.sleep(1)
    scores[model] = model_scores

rows = []
for model, model_scores in scores.items():
    rows.append({"model": model, "score": np.mean(model_scores)})

df = pd.DataFrame(rows)
print(df)

# using DeepEvalLLM

class GroqCriticModel(DeepEvalBaseLLM):
    def __init__(self, model: str):
        self.model = model

    def load_model(self):
        pass

    def generate(self, prompt: str) -> str:
        return predict(prompt, model=self.model)
    
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)
    
    def get_model_name(self) -> str:
        return self.model
    
critic_model= GroqCriticModel(model=CRITIC_MODEL)

test_case = LLMTestCase(
    input=prediction.question,
    actual_output=prediction.answer,
    context=[prediction.context],
    retrieval_context=[prediction.context],
)

#Answer Relevancy
relevancy_metric = AnswerRelevancyMetric(model=critic_model, threshold=0.7, include_reason=True)
relevancy_metric.measure(test_case)
print(relevancy_metric.score, relevancy_metric.reason)

#Faithfulness
faithfulness_metric = FaithfulnessMetric(model=critic_model, threshold=0.7, include_reason=True)
faithfulness_metric.measure(test_case)
print(faithfulness_metric.score, faithfulness_metric.reason)

#Hallucination
hallucination_metric = HallucinationMetric(model=critic_model, threshold=0.5, include_reason=True)
hallucination_metric.measure(test_case)
print(hallucination_metric.score, hallucination_metric.reason)

@dataclass
class ModelEvaluation:
    model: str
    relevancy: List[float] = field(default_factory=list)
    faithfulness: List[float] = field(default_factory=list)
    hallucination: List[float] = field(default_factory=list)

    no_relevancy_reasons: List[str] = field(default_factory=list)
    no_faithfulness_reasons: List[str] = field(default_factory=list)
    hallucination_reasons: List[str] = field(default_factory=list)

evaluations = []
for model, model_predictions in predictions.items():
    evaluation = ModelEvaluation(model=model)
    for prediction in model_predictions:
        test_case = LLMTestCase(
            input=prediction.question,
            actual_output=prediction.answer,
            context=[prediction.context],
            retrieval_context=[prediction.context],
        )
        try:
            relevancy_metric.measure(test_case)
            faithfulness_metric.measure(test_case)
            hallucination_metric.measure(test_case)

            evaluation.relevancy.append(relevancy_metric.score)
            evaluation.faithfulness.append(faithfulness_metric.score)
            evaluation.hallucination.append(hallucination_metric.score)

            if relevancy_metric.score < 0.5:
                evaluation.no_relevancy_reasons.append(relevancy_metric.reason)
            if faithfulness_metric.score < 0.5:
                evaluation.no_faithfulness_reasons.append(faithfulness_metric.reason)
            if hallucination_metric.score > 0.5:
                evaluation.hallucination_reasons.append(hallucination_metric.reason)
        except:
            continue
        
        # add a delay to avoid rate limit
        time.sleep(1)
        evaluations.append(evaluation)

save_list(evaluations, "evaluations.pkl")

# generate report
rows = []
for evaluation in evaluations:
    rows.append(
        {
            "model": evaluation.model,
            "relevancy": np.mean(evaluation.relevancy),
            "faithfulness": np.mean(evaluation.faithfulness),
            "hallucination": np.mean(evaluation.hallucination),
        }
    )

df = pd.DataFrame(rows)
print(df)