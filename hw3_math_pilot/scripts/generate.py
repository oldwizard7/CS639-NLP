import json, os, re, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA_PATH = "data/math_problems.jsonl"
OUT_PATH = "logs/generations.jsonl"

MODELS = {
    "base": "Qwen/Qwen2.5-Math-1.5B",
    "sft":  "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "drpo": "sail/Qwen2.5-Math-1.5B-Oat-Zero",
}

K = 5
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_NEW_TOKENS = 512
SEED = 0

PROMPT_TMPL = """Solve the following problem. Show numbered steps, then end with:
Final Answer: <your answer>

Problem:
{problem}
"""

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_final_answer(text):
    # 1) Final Answer:
    m = re.search(r"Final Answer:\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip().splitlines()[0].strip()

    # 2) boxed answer (common for SFT/DRPO)
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()

    # 3) natural language pattern (common for Base)
    m = re.search(r"final answer is\s*\$?(.+?)\$?[\.\n]", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 4) fallback: last number-like token
    m = re.findall(r"(-?\d+\.?\d*)", text)
    if m:
        return m[-1].strip()

    return None

def main():
    os.makedirs("logs", exist_ok=True)
    problems = load_jsonl(DATA_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print(f"Loaded {len(problems)} problems from {DATA_PATH}")

    # overwrite each run
    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)

    for tag, name in MODELS.items():
        print(f"\n=== Loading {tag}: {name} ===")
        tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            model.to(device)
        model.eval()

        with open(OUT_PATH, "a", encoding="utf-8") as out_f:
            for p in problems:
                prompt = PROMPT_TMPL.format(problem=p["problem"])
                inputs = tok(prompt, return_tensors="pt").to(device)

                for sample_id in range(K):
                    set_seed(SEED + sample_id)
                    t0 = time.time()

                    with torch.no_grad():
                        out = model.generate(
                            **inputs,
                            do_sample=True,
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            max_new_tokens=MAX_NEW_TOKENS,
                            pad_token_id=tok.eos_token_id,
                        )

                    full = tok.decode(out[0], skip_special_tokens=True)
                    gen = full[len(prompt):].strip() if full.startswith(prompt) else full.strip()

                    row = {
                        "problem_id": p["problem_id"],
                        "model_tag": tag,
                        "model_name": name,
                        "sample_id": sample_id,
                        "decoding": {
                            "temperature": TEMPERATURE,
                            "top_p": TOP_P,
                            "max_new_tokens": MAX_NEW_TOKENS,
                            "seed": SEED + sample_id,
                        },
                        "prompt": prompt,
                        "output_text": gen,
                        "decoded_final_answer": extract_final_answer(gen),
                        "runtime_sec": round(time.time() - t0, 3),
                    }
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_f.flush()

        del model
        torch.cuda.empty_cache()

    print(f"\nDone. Wrote generations to {OUT_PATH}")

if __name__ == "__main__":
    main()
