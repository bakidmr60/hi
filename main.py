# RAFT fine‑tuning on GSM‑8K – **two‑phase prompt strategy** + periodic **AIME 2024** evaluation
# ============================================================================
# **Phases**
#   • *Steps 1‑100* – Generation keeps a verbose system instruction, but the
#     gradients see only the user question (no system text) to teach the model
#     to rely on its own reasoning rather than memorising that single prompt.
#   • *Step >100*   – Both generation **and** training drop the system prompt
#     entirely. The model must now solve problems from a plain question.
#
# **New in this revision**
#   • Every `EVAL_EVERY` optimiser steps we run an accuracy sweep on the
#     **Maxwell‑Jia/AIME_2024** dataset (American Invitational Math Exam).
#   • The eval prompt **never** includes the system instruction – we measure
#     true generalisation.
#   • Accuracy is reported to the TQDM bar and logged; checkpoints carry the
#     metric in their directory name for quick inspection.
# ----------------------------------------------------------------------------

import os, re, random, time, json
from pathlib import Path
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# -------------------- hyper‑parameters --------------------
MODEL_PATH   = "Qwen/Qwen2.5-Math-1.5B"   # <‑‑ change to your local checkpoint
SAVE_DIR     = "raft_ckpt"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE        = torch.bfloat16            # use fp16 if your GPU lacks bf16

BATCH_PROMPTS      = 8                  # #different problems per optimiser step
CANDIDATES_PER_Q   = 6                  # N in RAFT
GEN_TEMPERATURE    = 0.9
MAX_NEW_TOKENS     = 700

LR                 = 1e-6
GRAD_ACC_STEPS     = 4                  # effective batch == BATCH_PROMPTS×ACC
TOTAL_STEPS        = 1_000
PRINT_EVERY        = 50
SAVE_EVERY         = 200

PHASE_SWITCH_STEP  = 100               # after this, drop system prompt fully
ACCEPT_THRESHOLD   = 0.0               # reward > this ⇒ keep trace

EVAL_EVERY         = 200               # optimiser steps between AIME evals
EVAL_MAX_PROBLEMS  = None              # None ⇒ full set; else first N items

# -------------------- prompt builders --------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant. A conversation between User and Assistant. "
    "The user asks a question, and the Assistant solves it. "
    "The Assistant first thinks about the reasoning process in the mind and "
    "then provides the user with the answer. The reasoning process and answer "
    "are enclosed within <think></think> and <answer></answer> tags, "
    "respectively, i.e., <think> reasoning process here </think><answer> answer "
    "here </answer>."
)

def build_prompt(question: str, tokenizer, include_system: bool) -> str:
    """Return a chat‑formatted string with or without the system role."""
    messages = []
    if include_system:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# -------------------- utility: numeric extraction --------------------

def _last_number(text: str):
    nums = re.findall(r"\d+\.\d+|\d+/\d+|\d+", text)
    return nums[-1] if nums else None

# -------------------- reward functions (for GSM‑8K) --------------------

def reward_correct(gt_answer: str, generation: str) -> float:
    gt   = _last_number(gt_answer)
    pred = _last_number(generation)
    return 1.0 if (gt is not None and pred == gt) else -1.0

def reward_format(generation: str) -> float:
    pattern = r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$"
    proper  = re.match(pattern, generation, re.DOTALL) is not None
    tags_ok = generation.count("<think>")  + generation.count("</think>")   == 2 and \
              generation.count("<answer>") + generation.count("</answer>") == 2
    return 1.25 if (proper and tags_ok) else -1.0

# -------------------- model & datasets --------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    _attn_implementation="sdpa",
).to(DEVICE)
model.gradient_checkpointing_enable()

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# GSM‑8K train split (~7.5 k items)
gsm8k = load_dataset("openai/gsm8k", "main", split="train")
corpus = list(zip(gsm8k["question"], [a.split("####")[-1].strip() for a in gsm8k["answer"]]))
random.shuffle(corpus)

# AIME 2024 eval dataset (≈15 problems)
aime_ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
aime_questions = aime_ds["Problem"]
aime_answers   = [str(x) for x in aime_ds["Answer"]]
if EVAL_MAX_PROBLEMS is not None:
    aime_questions = aime_questions[:EVAL_MAX_PROBLEMS]
    aime_answers   = aime_answers[:EVAL_MAX_PROBLEMS]

# -------------------- helpers --------------------

def sample_candidates(prompt_ids: torch.Tensor, n: int) -> Tuple[List[str], List[torch.Tensor]]:
    """Generate *n* candidate completions; return decoded text & token‑ids."""
    prompt_len = prompt_ids.size(1)
    batch_ids  = prompt_ids.repeat(n, 1)
    with torch.no_grad():
        gen_out = model.generate(
            inputs=batch_ids.to(DEVICE),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=GEN_TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
        )
    answers_token_ids = [gen_out[i][prompt_len:] for i in range(n)]
    answers_text      = tokenizer.batch_decode(answers_token_ids, skip_special_tokens=True)
    return answers_text, answers_token_ids

# deterministic generation for evaluation
@torch.no_grad()
def eval_aime() -> float:
    model.eval()
    correct = 0
    for q, gt in zip(aime_questions, aime_answers):
        prompt = build_prompt(q, tokenizer, include_system=False)
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
        out_ids = model.generate(
            inputs=prompt_ids,
            max_new_tokens=100,
            do_sample=False,  # greedy / deterministic
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )[0][prompt_ids.size(1):]
        pred_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        if _last_number(pred_text) == str(gt):
            correct += 1
    acc = correct / len(aime_questions)
    model.train()
    return acc

# -------------------- training loop --------------------

accum_loss = 0.0
opt_step   = 0
progress   = tqdm(range(1, TOTAL_STEPS + 1), desc="RAFT‑2phase")

for global_step in progress:
    # 0) decide which prompts keep the system instruction -------------------
    include_system_for_gen   = global_step <= PHASE_SWITCH_STEP
    include_system_for_train = False  # spec: never include during training

    # 1) sample GSM‑8K batch -------------------------------------------------
    batch = random.sample(corpus, BATCH_PROMPTS)
    accepted_inputs = []  # token‑id sequences retained for training

    for q_text, gt_ans in batch:
        # ---- build prompts --------------------------------------------------
        gen_prompt   = build_prompt(q_text, tokenizer, include_system_for_gen)
        train_prompt = build_prompt(q_text, tokenizer, include_system_for_train)
        gen_ids      = tokenizer(gen_prompt,   return_tensors="pt", add_special_tokens=False).input_ids
        train_ids    = tokenizer(train_prompt, return_tensors="pt", add_special_tokens=False).input_ids

        # ---- sample candidates ---------------------------------------------
        cand_texts, cand_token_ids = sample_candidates(gen_ids, CANDIDATES_PER_Q)
        rewards = [reward_correct(gt_ans, t) + reward_format(t) for t in cand_texts]

        # ---- accept positives ----------------------------------------------
        for rwd, ans_ids in zip(rewards, cand_token_ids):
            if rwd > ACCEPT_THRESHOLD:
                merged = torch.cat([train_ids[0], ans_ids])  # NOTE: training prompt ids
                accepted_inputs.append(merged)

    if not accepted_inputs:
        if global_step % PRINT_EVERY == 0:
            progress.set_postfix_str("no accepted traces – skipping update")
        continue

    # 2) pad & create labels --------------------------------------------------
    input_pad = pad_sequence(accepted_inputs, batch_first=True, padding_value=tokenizer.pad_token_id).to(DEVICE)
    labels    = input_pad.clone()
    for i, seq in enumerate(accepted_inputs):
        prompt_len = len(seq) - len(seq[seq != tokenizer.pad_token_id])  # seq has no pad; prompt==train_ids
        labels[i, :prompt_len] = -100

    # 3) forward / backward ---------------------------------------------------
    outputs = model(input_ids=input_pad, labels=labels)
    loss    = outputs.loss / GRAD_ACC_STEPS
    loss.backward()
    accum_loss += loss.item()

    if (global_step % GRAD_ACC_STEPS) == 0:
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        opt_step += 1
        if opt_step % PRINT_EVERY == 0:
            progress.set_postfix(loss=f"{accum_loss / PRINT_EVERY:.4f}")
            accum_loss = 0.0

        # 4) scheduled evaluation -------------------------------------------
        if opt_step % EVAL_EVERY == 0:
            acc = eval_aime()
            progress.set_postfix(loss=f"{accum_loss:.4f}", AIME_acc=f"{acc:.3f}")
            print(f"\n[AIME 2024] step {opt_step} – accuracy {acc:.3%}\n")

            # save quick‑look metric into checkpoint dir name
            ckpt_dir = Path(SAVE_DIR) / f"step_{opt_step}_AIME{int(acc*1000):03d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"[✓] checkpoint saved → {ckpt_dir}")

    # 5) periodic save without eval -----------------------------------------
    if opt_step and (opt_step % SAVE_EVERY == 0 and opt_step % EVAL_EVERY != 0):
        ckpt_dir = Path(SAVE_DIR) / f"step_{opt_step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"[✓] checkpoint saved → {ckpt_dir}")

print("\nTraining complete.")
