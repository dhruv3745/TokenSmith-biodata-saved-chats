import json
import textwrap
from pathlib import Path
from src.query_enhancement import load_recent_chat_summaries, _format_chat_summaries
from src.generator import run_llama_cpp, text_cleaning


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
        result.append(f"## {section}")
        result.append("")
        result.append(new_content)
        result.append("")

    path.write_text("\n".join(result), encoding="utf-8")


def update_biodata_skills(
    model_path: str,
    new_chat: list,
    biodata_path: str = "biodata.md",
    max_tokens: int = 512,
    **llm_kwargs,
):
    """
    Updates the SKILLS section of biodata.md based on patterns inferred
    from the provided chat/session summaries.

    new_chat should be a list of chat summary entries compatible with
    _format_chat_summaries().
    """
    current_skills = load_biodata_section(biodata_path, "SKILLS")
    # if not current_skills:
    #     print("No existing SKILLS section found in biodata.md — skipping update.")
    #     return

    if not new_chat:
        print("No chat data provided — skipping skill update.")
        return

    history_block = _format_chat_summaries(new_chat)

    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a student profile updater for an educational database systems assistant.

        Your task is to update the student's SKILLS section based on evidence from
        their recent chat sessions. The chat sessions show what concepts they've
        discussed, their learning progress, and any struggles or breakthroughs.

        Rules:
        - Only upgrade a skill if the session shows confident engagement with that concept or a significant focus on it.
        - Any new skills added should have a low proficiency level (1-4).
        - Note weaknesses or gaps if learning_progress shows repeated confusion.
        - Do NOT invent skills that aren't evidenced in the session history.
        - Output ONLY the updated SKILLS section content — no headers, no explanation.
        - Preserve existing skills that are not contradicted by the chat history.
        - Format as a clear, readable bullet list.
        
        Format of the skills are one per line "skill : level". Level is a scale between 1-10 of proficiency, or "unknown" if not enough information.
        <|im_end|>
        <|im_start|>user
        Current SKILLS section:
        {current_skills}

        Recent Chat Sessions:
        {history_block}

        Output the updated SKILLS section content only:
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

    updated_skills = output["choices"][0]["text"].strip()

    if not updated_skills or len(updated_skills) < 10:
        print("LLM returned empty update — biodata.md unchanged.")
        return

    replace_biodata_section(biodata_path, "SKILLS", updated_skills)
    print("biodata.md SKILLS section updated from provided chat history.")