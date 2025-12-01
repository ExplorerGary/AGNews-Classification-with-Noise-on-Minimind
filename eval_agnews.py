'''
HOW TO USE:
1. make sure the environment is set.
2. rename the model you want to eval (which is stored in out_agnews folder) to "full_sft_768.pth" # this is vital as otherwise the script can't identify the checkpoint.
3. run the following command:

python eval_agnews.py \
  --jsonl your_test_dataset.jsonl \
  --out_dir out_agnews \
  --hidden_size 768 \
  --num_hidden_layers 16 \
  --device cuda \
  --model_mode 1 \

BE AWARE: the only args you need to change is the jsonl.
simply input the name (like clean_valid.jsonl) alone, and the script will automatically run the eval pipeline

the available .jsonl includes:
clean_train clean_valid clean_test
noisy_train noisy_valid noisy_test

++++++++++++++++++++++++++++++++++++++++++++++++++++
Here is an example running eval on clean_test.jsonl:
++++++++++++++++++++++++++++++++++++++++++++++++++++
python eval_agnews.py \
  --jsonl clean_test.jsonl \
  --out_dir out_agnews \
  --hidden_size 768 \
  --num_hidden_layers 16 \
  --device cuda \
  --model_mode 1 \

'''


import argparse
import json
import torch
import random
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForCausalLM

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained('./model/')

    if args.load == 0:
        moe_path = '_moe' if args.use_moe else ''
        modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason', 4: 'grpo'}
        ckp = f'./{args.out_dir}/{modes[args.model_mode]}_{args.hidden_size}{moe_path}.pth'

        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=args.use_moe
        ))

        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)

        if args.lora_name != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.out_dir}/lora/{args.lora_name}_{args.hidden_size}.pth')
    else:
        transformers_model_path = './MiniMind2'
        tokenizer = AutoTokenizer.from_pretrained(transformers_model_path)
        model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)

    print(f'Model has: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params')
    return model.eval().to(args.device), tokenizer

def load_jsonl(path):
    datas = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            datas.append(json.loads(line))
    return datas

def extract_xy(item):
    user = item["conversations"][0]["content"].strip()
    label = item["conversations"][1]["content"].strip()
    return user, int(label)

def extract_pred(text):
    # 只提取 0/1/2/3
    for ch in text:
        if ch in ["0", "1", "2", "3"]:
            return int(ch)
    return -1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--out_dir", default="out_agnews")
    parser.add_argument("--lora_name", default="None")

    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--use_moe", default=False, type=bool)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--max_new_tokens", default=32, type=int)

    parser.add_argument("--model_mode", default=1, type=int)
    parser.add_argument("--load", default=0, type=int)

    args = parser.parse_args()
    setup_seed(2025)
    jsonl_file = args.jsonl
    jsonl_path = os.path.join("dataset",jsonl_file)
    model, tokenizer = init_model(args)
    data = load_jsonl(jsonl_path)

    y_true, y_pred = [], []
    results = []

    for item in tqdm(data):
        prompt, label = extract_xy(item)

        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(args.device)

        with torch.no_grad():
            out_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        output_text = tokenizer.decode(
            out_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        pred = extract_pred(output_text)

        y_true.append(label)
        y_pred.append(pred)

        results.append({
            "input": prompt,
            "label": label,
            "prediction": pred,
            "raw_output": output_text
        })

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    print("\n==========  Evaluation  ==========")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {p:.4f}")
    print(f"Recall    : {r:.4f}")
    print(f"F1-score  : {f1:.4f}")

    save_path = f"{args.out_dir}/eval_results.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f1,
            "details": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n results stored to: {save_path}")

if __name__ == "__main__":
    main()
