import textwrap
import json
import uuid
import re
from datetime import datetime
from src.generator import (
    text_cleaning, get_llama_model, run_llama_cpp,
    answer, format_prompt, ANSWER_START, ANSWER_END
)
from .update_biodata import update_biodata_skills

SAVED_CHATS_FILE = 'saved_chats.json'

TIER_SHORT_MAX  = 3   
TIER_MEDIUM_MAX = 8   


def _count_exchanges(chat: list) -> int:
    """Return the number of user/assistant pairs in the chat."""
    return sum(1 for turn in chat if turn.get("role") == "user")


def _build_transcript(chat: list) -> str:
    """Convert chat list into a plain readable transcript."""
    return "\n".join(
        f"{turn['role'].capitalize()}: {turn['content']}"
        for turn in chat
    )


def _build_metadata(chat: list) -> dict:
    """
    Build a metadata dict to attach to every saved chat entry.

    Fields
    ------
    chat_id        : unique identifier for this conversation
    timestamp      : ISO-8601 UTC timestamp of when it was saved
    message_count  : total number of turns (user + assistant)
    exchange_count : number of user/assistant pairs
    tier           : summarization tier chosen (short/medium/detailed)
    """
    exchange_count = _count_exchanges(chat)

    if exchange_count <= TIER_SHORT_MAX:
        tier = "short"
    elif exchange_count <= TIER_MEDIUM_MAX:
        tier = "medium"
    else:
        tier = "detailed"

    return {
        "chat_id":        str(uuid.uuid4()),
        "timestamp":      datetime.utcnow().isoformat() + "Z",
        "message_count":  len(chat),
        "exchange_count": exchange_count,
        "tier":           tier,
    }


def _get_system_prompt(tier: str) -> str:
    """
    Return a tier-appropriate system prompt that instructs the LLM
    to reply ONLY with a valid JSON object — no preamble, no fences.

    The JSON schema is consistent across all tiers; only the depth
    of each field scales with tier.
    """
    schema = textwrap.dedent("""
        {
          "chat_focus":        "<one-sentence topic of the conversation>",
          "key_concepts":      ["<concept>", ...],
          "user_interests":    ["<interest>", ...],
          "learning_progress": "<what the user understood or struggled with>",
          "summary":           "<prose summary>"
        }
    """).strip()

    tier_instructions = {
        "short": textwrap.dedent("""
            The conversation was short (3 or fewer exchanges).
            - chat_focus: one concise sentence
            - key_concepts: up to 3 items
            - user_interests: up to 2 items
            - learning_progress: one brief sentence
            - summary: 2-3 sentences, under 60 words
        """).strip(),

        "medium": textwrap.dedent("""
            The conversation was moderate in length (4-8 exchanges).
            - chat_focus: one sentence
            - key_concepts: up to 6 items
            - user_interests: up to 4 items
            - learning_progress: 1-2 sentences noting confidence and gaps
            - summary: 4-6 sentences, under 120 words
        """).strip(),

        "detailed": textwrap.dedent("""
            The conversation was long (9+ exchanges).
            - chat_focus: one sentence capturing the overarching theme
            - key_concepts: up to 10 items, ordered by importance
            - user_interests: up to 6 items
            - learning_progress: 2-3 sentences covering what was mastered,
              what was revisited, and any apparent misconceptions
            - summary: thorough prose recap, under 220 words
        """).strip(),
    }

    return textwrap.dedent(f"""
        You are a precise summarizer. Given a conversation transcript, return ONLY
        a single valid JSON object — no markdown fences, no extra text before or after.

        JSON schema:
        {schema}

        Depth guidelines for this conversation ({tier}):
        {tier_instructions[tier]}

        End your reply with {ANSWER_END}.
    """).strip()


def _get_max_tokens(tier: str) -> int:
    return {"short": 200, "medium": 400, "detailed": 700}[tier]


def _extract_json(raw: str) -> dict:
    """
    Safely extract a JSON object from the model's raw output.
    Falls back to a minimal error dict if parsing fails.
    """
    # Strip markdown fences if the model added them despite instructions
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

    # Isolate the first {...} block
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Last-resort fallback — store the raw text so nothing is lost
    return {
        "chat_focus":        "Parsing failed",
        "key_concepts":      [],
        "user_interests":    [],
        "learning_progress": "Unable to parse model output.",
        "summary":           raw[:300],
    }


def load_saved_chats() -> list:
    """Load saved chats from disk, creating the file if absent."""
    try:
        with open(SAVED_CHATS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        open(SAVED_CHATS_FILE, "a").close()
        return []


def process_chat(chat: list, cfg) -> dict:
    """
    Summarize a completed chat session into a structured log entry.

    Returns a dict with two top-level keys:
      - "metadata" : chat_id, timestamp, message_count, exchange_count, tier
      - "summary"  : parsed JSON with chat_focus, key_concepts,
                     user_interests, learning_progress, summary
    """
    metadata   = _build_metadata(chat)
    tier       = metadata["tier"]
    transcript = _build_transcript(chat)
    sys_prompt = _get_system_prompt(tier)
    max_tokens = _get_max_tokens(tier)

    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        {sys_prompt}
        <|im_end|>
        <|im_start|>user
        Conversation Transcript:
        {transcript}
        <|im_end|>
        <|im_start|>assistant
        {ANSWER_START}
    """)

    result = run_llama_cpp(
        prompt,
        cfg.gen_model,
        max_tokens=max_tokens,
        temperature=0.2,
    )
    raw    = result["choices"][0]["text"].strip()
    parsed = _extract_json(raw)

    return {
        "metadata": metadata,
        "summary":  parsed,
    }


def update_saved_chats(new_chat: list, cfg) -> list:
    """
    Summarize new_chat, attach metadata, and persist to disk.

    Returns the updated full list of saved chats.
    """
    if not new_chat:
        return load_saved_chats()

    saved_chats = load_saved_chats()
    entry       = process_chat(new_chat, cfg)
    saved_chats.append(entry)
    update_biodata_skills

    with open(SAVED_CHATS_FILE, 'w') as f:
        json.dump(saved_chats, f, indent=4)
    update_biodata_skills(cfg.gen_model, new_chat)

    return saved_chats