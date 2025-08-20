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

base_system_prompt = """
Answer the question based on the given context. Only give me the answer and do not output any other words. The following are given context.

Context:{context}
"""

base_user_prompt = """
Question: {question}
"""

class Candidate_Generator:
    def __init__(self, data_path, model_path, save_prompt_path, save_results_path):
        self.query, self.context, self.answer = self.read_dataset(data_path)
        self.model = LLM(model=model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.sampling_params = SamplingParams(temperature=0, top_p=0.8, max_tokens=8192)
        self.save_prompt_path = save_prompt_path
        self.save_results_path = save_results_path

    def generate_prompts(self):
        prompts = []
        for q, c in tqdm(zip(self.query, self.context), total=len(self.query), desc='Generating prompts'):
            query_prompts = []
            for p in c:
                context_passages = f"(Title:{p['title']}) {p['snippet']}"
                message = [
                    {"role":"system","content":base_system_prompt.format(context=context_passages)},
                    {"role":"user","content":base_user_prompt.format(question=q)}
                ]
                prompt = self.tokenizer.apply_chat_template(
                    message,
                    add_generation_prompt=True,
                    tokenize=False
                )
                query_prompts.append(prompt)
            prompts.append(query_prompts)

    def read_dataset(self, data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
        query = [i['query'] for i in data]
        context = [i['context'] for i in data]
        answer = [i['answer'] for i in data]
        return query, context, answer
    
    def generate_answers(self, prompts):
        results = []
        for row in tqdm(prompts, desc='Generating text'):
            row_results = []
            for text in row:
                outputs = self.model.generate(text, self.sampling_params, use_tqdm=False)
                for output in outputs:
                    row_results.append(output.outputs[0].text)
            results.append(row_results)
        return results
    
    def run(self):
        prompts = self.generate_prompts()
        if self.save_prompt_path:
            os.makedirs(self.save_prompt_path, exist_ok=True)
            with open(os.path.join(self.save_prompt_path, 'prompts.json'), 'w') as f:
                json.dump(prompts, f, indent=4)
        results = self.generate_answers(prompts)
        os.makedirs(self.save_results_path, exist_ok=True)
        save_results = []
        for q, c, a, r in zip(self.query, self.context, self.answer, results):
            save_results.append({
                'query': q,
                'context': c,
                'answer': a,
                'candidates': r
            })
        with open(os.path.join(self.save_results_path, 'candidates.json'), 'w') as f:
            json.dump(save_results, f, indent=4)

