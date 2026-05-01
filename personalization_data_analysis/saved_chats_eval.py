# saved_chats_eval.py
from pathlib import Path
import math
import numpy as np
from llama_cpp import Llama

# --- Config ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CHAT_DIR = ROOT / "Chat_history"
MODEL_PATH = ROOT.parent / "models" / "qwen2.5-1.5b-instruct-q5_k_m.gguf"

CONVERSATIONS = ["c1", "c2", "c3"]
TOPIC_LABELS = {
    "c1": "Joins",
    "c2": "Logging",
    "c3": "Query Opt.",
}

BASELINE_PATTERN = "{c}_response_no_history.txt"
WITH_PATTERN     = "{c}_response_yes_history.txt"

CONTEXT_SIZE  = 4096
THRESHOLD_PCT = 25.0   


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    print(f"Loading model: {MODEL_PATH.name}")
    return Llama(
        model_path=str(MODEL_PATH),
        n_ctx=CONTEXT_SIZE,
        logits_all=True,
        verbose=False,
    )


def perplexity(llm: Llama, text: str) -> float:
    """
    Unconditional perplexity of `text`.

    Tokenizes the text, runs a single forward pass, and averages the negative
    log-likelihood across all tokens (skipping the first, since there's no
    prior context to predict it from).
    """
    token_ids = llm.tokenize(text.encode("utf-8"), add_bos=True)

    if len(token_ids) > CONTEXT_SIZE:
        # truncate from the right; we want a representative sample of the response
        token_ids = token_ids[:CONTEXT_SIZE]
        print(f"  (truncated to {CONTEXT_SIZE} tokens)")

    if len(token_ids) < 2:
        return float("nan")  

    llm.reset()
    llm.eval(token_ids)

    scores = np.array(llm.scores[: len(token_ids)])

    target_ids = token_ids[1:]
    target_logits = scores[:-1]   

    max_logits = target_logits.max(axis=-1, keepdims=True)
    log_probs = target_logits - max_logits - np.log(
        np.exp(target_logits - max_logits).sum(axis=-1, keepdims=True)
    )

    token_logprobs = log_probs[np.arange(len(target_ids)), target_ids]
    avg_neg_logprob = -token_logprobs.mean()
    return float(math.exp(avg_neg_logprob))


def read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8")


def main():
    llm = load_model()

    header = f"{'Scenario':<12} {'No SC':>8} {'With SC':>9} {'% Diff':>9}  Significant?"
    print("\n" + header)
    print("-" * len(header))

    for c in CONVERSATIONS:
        baseline_text = read(CHAT_DIR / BASELINE_PATTERN.format(c=c))
        with_text     = read(CHAT_DIR / WITH_PATTERN.format(c=c))

        ppl_baseline = perplexity(llm, baseline_text)
        ppl_with     = perplexity(llm, with_text)

        pct = ((ppl_with - ppl_baseline) / ppl_baseline * 100) if ppl_baseline else 0.0
        flag = "yes" if pct >= THRESHOLD_PCT else "no"

        label = TOPIC_LABELS.get(c, c)
        print(f"{label:<12} {ppl_baseline:>8.2f} {ppl_with:>9.2f} {pct:>8.1f}%  {flag}")


if __name__ == "__main__":
    main()