import os, re, random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# -------- hyperparams --------
MODEL_PATH      = "Qwen/Qwen2.5-Math-1.5B"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE           = torch.bfloat16            # use torch.float16 if needed

TOTAL_STEPS     = 50
PRINT_EVERY     = 1
EVAL_EVERY      = 5
SAVE_EVERY      = 25

BATCH_PROMPTS   = 2
CANDIDATES_PER_Q= 8
TOPK_ACCEPT     = 1

PHASE_SWITCH_STEP = 10
GEN_TEMP          = 0.9
MAX_NEW_TOKENS    = 256
LR                = 3e-5

GSM_SUBSET      = 500
SAVE_DIR        = "raft_ckpt_quick"

# -------- templates --------
SYSTEM_PROMPT = (
    "You are a helpful assistant. The user asks a math question. "
    "Wrap your chain-of-thought between <think></think> and final answer "
    "between <answer></answer>."
)

def build_prompt(q: str, tok, include_system: bool) -> str:
    msgs = []
    if include_system:
        msgs.append({"role": "system", "content": SYSTEM_PROMPT})
    msgs.append({"role": "user", "content": q})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

# -------- reward helpers --------
_tag_pat   = re.compile(r"<think>.*?</think>.*?<answer>.*?</answer>", re.DOTALL)
_num_pat   = re.compile(r"\d+(?:/\d+)?(?:\.\d+)?")
def _last_number(txt: str):
    nums = _num_pat.findall(txt)
    return nums[-1] if nums else None
def make_reward(gt: str, gen: str) -> float:
    correct = _last_number(gen) == _last_number(gt)
    formatted = bool(_tag_pat.search(gen))
    return 1.0 if (correct or formatted) else -1.0

# -------- load model & data --------
print("Loading model …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=DTYPE, _attn_implementation="sdpa"
).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
model.gradient_checkpointing_enable()
model.train()

print("Loading datasets …")
gsm = load_dataset("openai/gsm8k", "main", split="train")
if GSM_SUBSET:
    gsm = gsm.select(range(GSM_SUBSET))
corpus = list(zip(gsm["question"], [a.split("####")[-1].strip() for a in gsm["answer"]]))
random.shuffle(corpus)

aime = load_dataset("Maxwell-Jia/AIME_2024", split="train")
aime_q = aime["Problem"]
aime_a = [str(x) for x in aime["Answer"]]

# -------- sampling util --------
@torch.no_grad()
def sample(prompt_ids: torch.Tensor, n: int):
    pl = prompt_ids.size(1)
    outs = model.generate(
        prompt_ids.repeat(n,1).to(DEVICE),
        max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=GEN_TEMP,
        pad_token_id=tokenizer.eos_token_id,
    )
    ans_toks = [outs[i][pl:].cpu() for i in range(n)]
    texts    = tokenizer.batch_decode(ans_toks, skip_special_tokens=True)
    return texts, ans_toks

# -------- eval util --------
@torch.no_grad()
def eval_aime() -> float:
    model.eval()
    hits = 0
    for q, gt in zip(aime_q, aime_a):
        p = build_prompt(q, tokenizer, include_system=False)
        pids = tokenizer(p, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
        out  = model.generate(
            pids, max_new_tokens=50, do_sample=False, temperature=0.0,
            pad_token_id=tokenizer.eos_token_id
        )[0][pids.size(1):]
        if _last_number(tokenizer.decode(out, skip_special_tokens=True)) == gt:
            hits += 1
    model.train()
    return hits / len(aime_q)

# -------- training loop --------
print("Starting training …")
accum_loss = 0.0
opt_step   = 0
bar = tqdm(range(1, TOTAL_STEPS+1), position=0, desc="RAFT-quick")

for gstep in bar:
    include_system_gen = gstep <= PHASE_SWITCH_STEP

    batch = random.sample(corpus, BATCH_PROMPTS)
    accepted = []
    prompt_lens = []
    for q, gt in batch:
        gen_prompt = build_prompt(q, tokenizer, include_system_gen)
        train_prompt = build_prompt(q, tokenizer, False)
        gen_ids   = tokenizer(gen_prompt, return_tensors="pt", add_special_tokens=False).input_ids
        train_ids = tokenizer(train_prompt, return_tensors="pt", add_special_tokens=False).input_ids

        texts, tok_ids = sample(gen_ids, CANDIDATES_PER_Q)
        rewards = [make_reward(gt, t) for t in texts]
        # take up to TOPK_ACCEPT best positives
        pos = [pair for pair in sorted(zip(rewards, tok_ids), key=lambda x: -x[0]) if pair[0] > 0][:TOPK_ACCEPT]
        for _, ans_ids in pos:
            ans_ids = ans_ids.cpu()
            merged = torch.cat([train_ids[0], ans_ids])
            accepted.append(merged)
            prompt_lens.append(train_ids.size(1))

    if not accepted:
        bar.set_postfix_str("no accept")
        continue

    pad = pad_sequence(accepted, batch_first=True, padding_value=tokenizer.pad_token_id).to(DEVICE)
    labels = pad.clone()
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100

    loss = model(input_ids=pad, labels=labels).loss
    loss.backward()

    if (gstep % 1) == 0:
        optimizer.step(); optimizer.zero_grad()
        opt_step += 1
        bar.set_postfix(loss=f"{loss.item():.3f}")

        if opt_step % EVAL_EVERY == 0:
            acc = eval_aime()
            bar.set_postfix(loss=f"{loss.item():.3f}", AIME=f"{acc:.2%}")
            print(f"\n[AIME 2024] step {opt_step} – acc {acc:.2%}\n")

        if opt_step % SAVE_EVERY == 0:
            ckpt = Path(SAVE_DIR)/f"step_{opt_step}"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt); tokenizer.save_pretrained(ckpt)

print("Done.")
