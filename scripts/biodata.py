#!/usr/bin/env python3
"""
Generate biodata.md from biodata_raw.txt using a local LLM.
"""

import os
import re
import sys
from pathlib import Path




def load_llama_model(model_path: str):
    try:
        from llama_cpp import Llama
    except ImportError:
        print("ERROR: llama_cpp is not installed. Run: pip install llama-cpp-python")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    try:
        return Llama(model_path=model_path, n_ctx=4096, verbose=False, n_gpu_layers=-1)
    except Exception as e:
        print(f"GPU load failed ({e}), retrying CPU-only...")
        return Llama(model_path=model_path, n_ctx=4096, verbose=False)



ANSWER_START = "<<<ANSWER>>>"
ANSWER_END   = "<<<END>>>"

QUESTIONS = [
    ("OCCUPATION",   "What is the student's occupation?"),
    ("SKILLS",       "What are the student's current skills and proficiencies? If unknown, state unknown.         Format of the skills are one per line 'skill : level'. Level is a scale between 1-10 of proficiency, or 'unknown' if not enough information."),
    ("CLASSES",      "What relevant classes has the student taken and what information would be covered in each class?"),
    ("TEST_SCORES",  "If the student has mentioned any recent test scores, note the score and any relevant notes. If none, state none."),
    ("INTERESTS",    "What are the basic personal information and interests of the student?"),
]

SYSTEM_PROMPT = f"""You are a student profile extractor.
You will be given raw notes about a student and a single question.
Answer using ONLY the information present in the notes.
If the answer is not present, clearly say so. NEVER create information not present in the notes.
Begin your answer with {ANSWER_START} and end it with {ANSWER_END}.
Do not include anything outside those markers."""


def build_prompt(biodata_raw: str, question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Student notes:\n{biodata_raw}\n\n"
        f"Question: {question}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{ANSWER_START}"
    )


def extract_answer(raw: str) -> str:
    """Pull out text between ANSWER_START and ANSWER_END."""
    # The model continues after ANSWER_START, so raw starts just after it
    # but we also handle cases where the model echoes the marker
    text = raw
    if ANSWER_START in text:
        text = text.split(ANSWER_START, 1)[1]
    if ANSWER_END in text:
        text = text.split(ANSWER_END, 1)[0]
    return text.strip()


def ask(model, biodata_raw: str, question: str, max_tokens: int = 512) -> str:
    prompt = build_prompt(biodata_raw, question)
    result = model.create_completion(
        prompt,
        max_tokens=max_tokens,
        temperature=0.1,
        stop=[ANSWER_END],
    )
    raw = result["choices"][0]["text"]
    return extract_answer(raw)



def main():
    project_root = Path(__file__).resolve().parent.parent

    biodata_raw_path = project_root / "biodata_raw.txt"
    if not biodata_raw_path.exists():
        print(f"ERROR: biodata_raw.txt not found at {biodata_raw_path}")
        sys.exit(1)
    biodata_md_path = project_root / "biodata.md"
    biodata_raw = biodata_raw_path.read_text(encoding="utf-8").strip()
    if not biodata_raw:
        print("Biodata_raw.txt is empty — nothing to process.")
        with open(biodata_md_path, "w", encoding="utf-8") as f:
            f.write("# Student Biodata\n\n")
            for label, _ in QUESTIONS:
                f.write(f"## {label}\n\n")
                f.write("")
                f.write("\n\n")
        print(f"\nbiodata.md written to: {biodata_md_path}")
        return

    print(f"Loaded biodata_raw.txt ({len(biodata_raw)} chars)")

    model_path = str(project_root / "models/qwen2.5-1.5b-instruct-q5_k_m.gguf")

    model = load_llama_model(model_path)

    results = {}
    for label, question in QUESTIONS:
        print(f"  Asking: {label}...")
        results[label] = ask(model, biodata_raw, question)

    
    with open(biodata_md_path, "w", encoding="utf-8") as f:
        f.write("# Student Biodata\n\n")
        for label, _ in QUESTIONS:
            f.write(f"## {label}\n\n")
            f.write(results[label])
            f.write("\n\n")

    print(f"\nbiodata.md written to: {biodata_md_path}")


if __name__ == "__main__":
    main()