import warnings
warnings.filterwarnings("ignore")
import re
import os
import csv
import json
import time
import types
import random
import textwrap
from tqdm import tqdm
from datetime import datetime
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from vllm import LLM, SamplingParams

def main():
    # parser = argparse.ArgumentParser(description="Parse GPU index and step parameters")
    # parser.add_argument('--stage', type=int, required=True)
    # parser.add_argument('--step', type=int, required=True)
    # args = parser.parse_args()

    # # model_path = f"/mnt/petrelfs/renqingnan/xietian/Logic-RL/checkpoints_2_4/GRPO_logic_KK/Qwen-7B-1M-3to7ppl_cold5/actor/global_step_225"
    # # model_path = f"/volume/ailab4sci/ztgao/Logic-RL/checkpoints/GRPO_logic_KK/RF++-xppl-stage3-len8192-step1800-t0_7-001/actor/global_step_{args.step}"
    # # model_path = f"/volume/ailab4sci/ztgao/Logic-RL/checkpoints/GRPO_logic_KK/RF++-Qwen-7B-1M-xppl-test-01/actor/global_step_{args.step}"
    
    # if args.stage == 0:
    #     model_path = "/volume/ailab4sci/models/Qwen2.5-7B-Instruct-1M"
    # elif args.stage == 1:
    #     model_path = f"/volume/ailab4sci/ztgao/Logic-RL/checkpoints/GRPO_logic_KK/RF++-Qwen-7B-1M-xppl-002/actor/global_step_{args.step}"
    # elif args.stage == 2:
    #     model_path = f"/volume/ailab4sci/ztgao/Logic-RL/checkpoints/GRPO_logic_KK/RF++-xppl-step1320-t0_7-001/actor/global_step_{args.step}"
        
    model_path = "/root/autodl-fs/checkpoints/6ppl/global_step_60"
    # model_path = "/volume/ailab4sci/models/CodeR1-Zero-Qwen2.5-7B-12k-832"
    # model_path = "/volume/ailab4sci/models/CodeR1-Zero-Qwen2.5-7B-LC2k-1088"
    
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        max_num_seqs=4
    )
    
    sampling_params = SamplingParams(
        max_tokens=10000,
        temperature=0.8,
        top_p=0.95,
    )

    with open("amc.jsonl", encoding="utf-8") as file:
        data = [json.loads(line) for line in file.readlines() if line]
    
    cnt = 0
    total_time = 0
    results = []

    for d in tqdm(data):
        prompt = d["problem"]
        messages = [
            {"role": "system", "content": "You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a math problem. After thinking, when you finally reach a conclusion, clearly state the answer within <answer> </answer> tags. i.e., <answer> (\\boxed{}\\) </answer>."},
            {"role": "user", "content": prompt}
        ]
        
        tokenizer = llm.get_tokenizer()
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        expected_answer = d['answer']
        start_time = time.time()
        outputs = llm.generate([text], sampling_params)
        time_taken = time.time() - start_time
        response = outputs[0].outputs[0].text.strip()

        if '<answer>' in response:
            result = re.split(r'<answer>', response)[1]
        else:
            result = response[len(response) - 30:]
        
        correct = str(int(expected_answer)) in result
        
        result = {
            "question": d['problem'],
            "generated_output": response,
            "expected_expected_answer": expected_answer,
            "correct": correct,
            "time_taken": time_taken
        }

        results.append(result)

        if correct:
            cnt += 1

        total_time += time_taken
    
    acc = cnt / len(data)
    print(f"ACC: {acc}")
    output_json = f"amc_output_6ppl.json"
    with open(output_json, 'w') as outfile:
        json.dump(results, outfile, indent=4)

if __name__ == "__main__":
    main()