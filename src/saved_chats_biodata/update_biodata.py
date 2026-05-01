import json
import re
import textwrap
from pathlib import Path

from src.generator import run_llama_cpp, text_cleaning, ANSWER_END



def load_biodata_section(biodata_path: str, section: str) -> str:
    """Extract a specific ## section from biodata.md."""
    path = Path(biodata_path)
    if not path.exists():
        return ""
    
    content = path.read_text(encoding="utf-8")
    in_section = False
    lines = []
    
    for line in content.splitlines():
        if line.startswith("## "):
            if in_section:
                break  # hit the next section, stop
            if line[3:].strip().upper() == section.upper():
                in_section = True
        elif in_section:
            lines.append(line)
    
    return "\n".join(lines).strip()

def _build_transcript(chat: list) -> str:
    """Convert chat list into a plain readable transcript."""
    return "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}"
        for turn in chat
    )

def replace_biodata_section(biodata_path: str, section: str, new_content: str):
    """Replace a specific ## section's content in biodata.md in-place."""
    path = Path(biodata_path)
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    
    result = []
    in_target = False
    skipping = False

    for line in lines:
        if line.startswith("## "):
            if skipping:
                result.append(f"## {section}")
                result.append("")
                result.append(new_content)
                result.append("")
                skipping = False
                in_target = False
            if line[3:].strip().upper() == section.upper():
                in_target = True
                skipping = True
                continue  
        
        if skipping:
            continue  
        
        result.append(line)

    if skipping:
        result.extend([f"## {section}", "", new_content, ""])
    elif not any(line.strip().upper() == f"## {section.upper()}" for line in lines):
        if result and result[-1] != "":
            result.append("")
        result.extend([f"## {section}", "", new_content, ""])


    path.write_text("\n".join(result), encoding="utf-8")


def _dedupe_skills(text: str) -> str:
    """
    Parse the LLM's skill list, deduplicate by skill name (case-insensitive,
    keeping the first occurrence), and return one skill per entry separated
    by blank lines. Robust to the model dumping everything on a single line.
    """
    seen = {}
    candidates = re.split(r'(?=- )', text)
    for chunk in candidates:
        chunk = chunk.strip()
        if not chunk.startswith("- ") or ":" not in chunk:
            continue
        line = chunk.split(" - ")[0].strip()
        name = line.split(":", 1)[0].strip().lower()
        if name not in seen:
            seen[name] = line
    return "\n\n".join(seen.values())


def update_biodata_skills(
    model_path: str,
    new_chat: list,
    biodata_path: str = "biodata.md",
    max_tokens: int = 512,
    max_transcript_chars: int = 8000,
    **llm_kwargs,
):
    """
    Updates the SKILLS section of biodata.md based on patterns inferred
    from the provided chat/session summaries.

    new_chat should be a list of chat summary entries compatible with
    _format_chat_summaries().
    """
    
    current_skills = load_biodata_section(biodata_path, "SKILLS")


    if not new_chat:
        print("No chat data provided — skipping skill update.")
        return

    transcript = _build_transcript(new_chat)
 
 
    if len(transcript) > max_transcript_chars:
        head_budget = max_transcript_chars // 4
        tail_budget = max_transcript_chars - head_budget - 50  # 50 chars for the marker
        transcript = (
            transcript[:head_budget]
            + "\n\n[... middle of conversation omitted ...]\n\n"
            + transcript[-tail_budget:]
        )

    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a student profile updater for an educational database systems assistant.

        Your task is to update the student's SKILLS section based on evidence from
        a single conversation transcript between the student and the assistant.
        Read the transcript carefully and infer skill signals from how the student
        asks questions, follows up, and reacts to explanations.

        Skill signals to look for:
        - Confident engagement: precise terminology, correct follow-up questions,
          extending the topic on their own.
        - Struggle: repeated re-asking of the same idea, basic clarifying questions
          on a topic the SKILLS list claims they know, explicit confusion.
        - New topic exposure: a concept appears that isn't in the current SKILLS list.

        Rules:
        - Copy any skill not addressed in the transcript EXACTLY as it appears in
          the current SKILLS section. Do not rephrase preserved skills.
        - Raise a level by at most 1 only when confident engagement is clear.
        - Lower a level by at most 1 when struggle is clear.
        - Add a new skill only if it is explicitly discussed in the transcript.
          New skills start at level 1-3.
        - Never invent skills not evidenced in the transcript.
        - Output ONLY the updated SKILLS list — no headers, no explanation, no preamble.
        - After the last skill line, output {ANSWER_END} and stop. Do not repeat the list.

        Format: one skill per line, exactly: "- <skill_name> : <level>"
        where <level> is an integer 1-10 or the literal "unknown".

        Example:
        Current SKILLS:
        - sql-joins : 4
        - normalization : 5

        Transcript shows confident inner/outer join usage and confusion about 3NF.

        Updated SKILLS:
        - sql-joins : 5
        - normalization : 4
        {ANSWER_END}
        <|im_end|>
        <|im_start|>user
        Current SKILLS section:
        {current_skills}

        Conversation Transcript:
        {transcript}

        Output the updated SKILLS list only:
        <|im_end|>
        <|im_start|>assistant
        """)

    prompt = text_cleaning(prompt)

    output = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=max_tokens,
        temperature=0.1,
        **llm_kwargs,
    )
    
    print("Raw LLM Output for Biodata SKILLS Update:")
    print(output)

    updated_skills = output["choices"][0]["text"].strip()

    updated_skills = _dedupe_skills(updated_skills)

    if not updated_skills or len(updated_skills) < 10:
        print("LLM returned empty or unparseable update — biodata.md unchanged.")
        return

    replace_biodata_section(biodata_path, "SKILLS", updated_skills)
    print("biodata.md SKILLS section updated from provided chat history.")