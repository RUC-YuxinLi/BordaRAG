import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import json
from tqdm import * 
import re
from vllm import LLM, SamplingParams
import torch.nn.functional as F
from argparse import ArgumentParser
import os
import math

system_prompt = """
You are an expert fact-checking assistant. Your task is to determine whether the given answer is factually supported by the provided context.
You must answer "True" only if the answer is directly supported by the context. If the answer is not explicitly stated in the context you must answer "False".
Do not include any explanation. Output only one word: "True" or "False".
"""
user_prompt = """
Question: 
{question}

Context:
{context}

Answer: 
{answer}
"""

class Preference_Calculator:
    def __init__(self, data_path, model_path, save_results_path):
        self.query, self.context, self.answer, self.candidates = self.read_data(data_path)
        self.model = LLM(model=model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.sampling_params = SamplingParams(temperature=0, top_p=0.8, max_tokens=8192)
        self.save_results_path = save_results_path

    def generate_prompt(self):
        prompts = []
        for q, ctext, c in tqdm(zip(self.query, self.context, self.candidates), total=len(self.query), desc='Generating prompts'):
            query_prompts = []
            for ci in c:
                candidate_prompts = []
                for ctexti in ctext:
                    cp = f"(Title: {ctexti['title']}) {ctexti['snippet']}"
                    message = [
                        {"role":"system","content":system_prompt},
                        {"role":"user", "content":user_prompt.format(question=q,context=cp,answer=ci)}
                    ]
                    prompt = self.tokenizer.apply_chat_template(
                        message,
                        add_generation_prompt=True,
                        tokenize=False
                    )
                    candidate_prompts.append(prompt)
                query_prompts.append(candidate_prompts)
            prompts.append(query_prompts)
        return prompts
    
    def read_data(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        query = [i['query'] for i in data]
        context = [i['context'] for i in data]
        answer = [i['answer'] for i in data]
        candidates = [list(set(i['candidates'])) for i in data]
        return query, context, answer, candidates

    def calculate_probs(self, prompts):
        results = []
        True_token = self.tokenizer.encode('True', add_special_tokens=False)[0]
        False_token = self.tokenizer.encode('False', add_special_tokens=False)[0]
        for row in tqdm(prompts, desc='Generating text'):
            row_results = []
            for text in row:
                outputs = self.model.generate(text, self.sampling_params, use_tqdm=False)
                c_results = []
                for output in outputs:
                    logprobs = output.outputs[0].logprobs[0]
                    probs_True = 0
                    probs_False = 0
                    if True_token in logprobs.keys():
                        probs_True = math.exp(logprobs[True_token].logprob)
                    if False_token in logprobs.keys():
                        probs_False = math.exp(logprobs[False_token].logprob)
                    c_results.append(probs_True / (probs_True + probs_False))
                row_results.append(c_results)
            results.append(row_results)
        return results

    def run(self):
        prompts = self.generate_prompt()
        results = self.calculate_probs(prompts)
        os.makedirs(self.save_results_path, exist_ok=True)
        save_results = []
        for q, c, a, ca, r in zip(self.query, self.context, self.answer, self.candidates, results):
            save_results.append({
                'query': q,
                'context': c,
                'answer': a,
                'candidates': ca,
                'scores': r,
            })
        with open(os.path.join(self.save_results_path, 'preference_scores.json'), 'w') as f:
            json.dump(save_results, f, indent=4)