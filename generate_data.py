import sys
import gc
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import torch
from torch import nn
from transformers import LlamaTokenizer, AutoModelForCausalLM, pipeline
from types import SimpleNamespace


paths = {
    'base': str(
        Path.cwd()),
    'test': '/kaggle/input/kaggle-llm-science-exam/test.csv',
    'sample_sub': '/kaggle/input/kaggle-llm-science-exam/sample_submission.csv',
    'model': '/kaggle/input/Llama-2-7b-chat-hf',
    'prompts': '/scripts/llama2-prompts.txt'}

for key in ['test', 'sample_sub', 'model', 'prompts']:
    paths[key] = Path(paths['base'] + paths[key])
paths = SimpleNamespace(**paths)


if __name__ == '__main__':
    df = pd.read_csv(paths.test)
    sub = pd.read_csv(paths.sample_sub)

    # Load prompts text file
    with open(paths.prompts, 'r') as f:
        lines = f.readlines()
    f.close()
    num_of_prompts = int(lines[0].split('Prompts:')[-1])
    # Parse each prompt

    df_prompts = None
    for prompt_num in range(1, num_of_prompts + 1, 1):
        info = {}
        info['prompt_num'] = [prompt_num]
        for name in ['INSTRUCT', 'CONTEXT']:
            start = lines.index(f'{name}_START-{prompt_num}\n')
            end = lines.index(f'{name}_END-{prompt_num}\n')
            info[name.lower()] = [' '.join(lines[start + 1:end]).strip('\n')]
        if df_prompts is None:
            df_prompts = pd.DataFrame(info)
        else:
            df_prompts = pd.concat([df_prompts, pd.DataFrame(info)], axis=0)

    model_name = paths.model

    tokenizer = LlamaTokenizer.from_pretrained(paths.model)
    pipeline = pipeline(
        "text-generation",
        model=paths.model,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )
    #https://huggingface.co/blog/llama2#how-to-prompt-llama-2
    for _ in range(3):
        for row in df_prompts.iterrows():
            row = row[1]
            system_prompt = row.instruct
            user_message = row.context
            context = 'The original name of the city was Whynot now it has been updated to Seagrove.'
            SYS = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>"
            CONVO = f"\n{user_message} [/INST]"
            SYS = "<s>" + SYS
            prompt = SYS + CONVO
            prompt = prompt.replace('{context}', context)
            sequences = pipeline(
                prompt,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=1028,
            )
            llama_reply = sequences[0]['generated_text'].split('[/INST]')[-1]
            print(f'{llama_reply}\n')
    print('checkpoint')