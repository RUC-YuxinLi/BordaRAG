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
import numpy as np
from candidate_generator import Candidate_Generator
from preference_calculator import Preference_Calculator

def parse():
    parser = ArgumentParser()
    parser.add_argument('-model_path', type=str, default='')
    parser.add_argument('-candidate_datapath', type=str, default='')
    parser.add_argument('-candidate_prompt_path', type=str, default='')
    parser.add_argument('-candidate_result_path', type=str, default='')
    parser.add_argument('-preference_datapath', type=str, default='')
    parser.add_argument('-preference_result_path', type=str, default='')
    parser.add_argument('-qa_datapath', type=str, default='')
    parser.add_argument('-qa_prompt_path', type=str, default='')
    parser.add_argument('-qa_result_path', type=str, default='')
    parser.add_argument('-topk', type=int, default=5)
    args = parser.parse_args()
    return args

base_system_prompt = """
You are a knowledgeable assistant tasked with answering questions based on the provided context. Follow these rules:
Answer Structure:
- If the context directly relates to the question, provide a concise, fact-based answer using only the context.
- If the context is irrelevant to the question, answer using your general knowledge.
Output Requirements:
- Ensure answers are consistent, coherent, and contain one complete entity (e.g., names, dates, concepts).
- Avoid speculative or uncertain language (e.g., "maybe," "possibly").

Context:{context}
"""

base_user_prompt = """
Question: {question}
"""

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class Voter:
    def __init__(self, model_path, datapath, topk, save_prompt_path, save_results_path):
        self.model = LLM(model=model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.sampling_params = SamplingParams(temperature=0, top_p=0.8, max_tokens=8192)
        self.topk = topk
        with open(datapath, 'r') as f:
            self.data = json.load(f)
        self.query = [i['query'] for i in self.data]
        self.save_prompt_path = save_prompt_path
        self.save_results_path = save_results_path

    def vote(self):
        filter_context = []
        for p in self.data:
            candidates = p['candidates']
            context = p['context']
            scores = p['scores']
            scores = np.array(scores, dtype=float)
            scores = np.array([softmax(s) for s in scores.T]).T
            chosen_candidate = self.borda_voting(candidates, scores)
            voter_ranks = []
            for s in scores.T:
                voter_ranks.append(int(np.where(np.argsort(s)[::-1]==candidates.index(chosen_candidate))[0])+1)
            chosen_voters = np.argsort(voter_ranks)[:self.topk]
            filter_context.append([context[i] for i in chosen_voters])
        return filter_context
    
    def generate_prompts(self, filter_context):
        prompts = []
        for q, c in tqdm(zip(self.query, filter_context), total=len(self.query), desc='Generating prompts'):
            context = '\n'.join([f"Doc {idx+1} (Title:{ci['title']}) {ci['snippet']}" for idx, ci in enumerate(c)])
            message = [
                {"role":"system","content":base_system_prompt.format(context=context)},
                {"role":"user","content":base_user_prompt.format(question=q)}
            ]
            prompt = self.tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=False
            )
            prompts.append(prompt)
        return prompts
    
    def generate_answers(self, prompts):
        results = []
        for text in tqdm(prompts, desc='Generating text'):
            outputs = self.model.generate(text, self.sampling_params, use_tqdm=False)
            for output in outputs:
                results.append(output.outputs[0].text)
        return results

    def borda_voting(self, candidates, scores):
        select_score = [0 for _ in range(len(candidates))]
        for s in scores.T:
            rank = np.argsort(s)[::-1]
            for idx, r in enumerate(rank):
                select_score[r] += len(candidates) - idx
        return candidates[np.argmax(select_score)]

    def run(self):
        filter_context = self.vote()
        prompts = self.generate_prompts(filter_context)
        if self.save_prompt_path:
            os.makedirs(self.save_prompt_path, exist_ok=True)
            with open(os.path.join(self.save_prompt_path, 'prompts.json'), 'w') as f:
                json.dump(prompts, f, indent=4)
        results = self.generate_answers(prompts)
        save_results = []
        for p, fc, r in zip(self.data, filter_context, results):
            save_results.append({
                "query": p['query'],
                "context": fc,
                "answer": p['answer'],
                "predict":r,
            })
        os.makedirs(self.save_results_path, exist_ok=True)
        with open(os.path.join(self.save_results_path, 'vote_results.json'), 'w') as f:
            json.dump(save_results, f, indent=4)

if __name__ == '__main__':
    args = parse()
    # generate candidates
    candidate_generator = Candidate_Generator(
        model_path=args.model_path,
        data_path=args.candidate_datapath,
        save_prompt_path=args.candidate_prompt_path,
        save_results_path=args.candidate_result_path,
    )
    candidate_generator.run()
    # calculate preference
    preference_calculator = Preference_Calculator(
        model_path=args.model_path,
        data_path=args.preference_datapath,
        save_results_path=args.preference_result_path,
    )
    preference_calculator.run()

    # vote
    voter = Voter(
        model_path=args.model_path,
        datapath=args.qa_datapath,
        topk=args.topk,
        save_prompt_path=args.qa_prompt_path,
        save_results_path=args.qa_result_path,
    )
    voter.run()
