import textwrap
import json
import uuid
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from src.generator import (
    text_cleaning, get_llama_model, run_llama_cpp,
    answer, format_prompt, ANSWER_START, ANSWER_END
)
from .update_biodata import update_biodata_skills

SAVED_CHATS_FILE = 'saved_chats.json'

LOGS_DIR = 'logs'
SESSION_GAP_SECONDS = 3600   # gap > 1h between two log files marks a new session

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
            The conversation was short (3 or fewer user/assistant pairs).
            - chat_focus: one concise sentence
            - key_concepts: up to 3 items
            - user_interests: up to 2 items
            - learning_progress: one brief sentence
            - summary: 2-3 sentences, under 60 words
            
            Example output for a conversation about SQL joins:
            {
            "chat_focus": "Understanding inner vs. outer joins in SQL.",
            "key_concepts": ["inner join", "left outer join", "join keys"],
            "user_interests": ["query writing", "relational databases"],
            "learning_progress": "Confident with inner joins; confused about NULL handling in outer joins.",
            "summary": "The student worked through several inner-join queries..."
            }
        """).strip(),

        "medium": textwrap.dedent("""
            The conversation was moderate in length (4-8 user/assistant pairs).
            - chat_focus: one sentence
            - key_concepts: up to 6 items
            - user_interests: up to 4 items
            - learning_progress: 1-2 sentences noting confidence and gaps
            - summary: 4-6 sentences, under 120 words
            
            Example output for a conversation about SQL joins:
            {
            "chat_focus": "Understanding inner vs. outer joins in SQL and why we use each.",
            "key_concepts": ["inner join", "left outer join", "join keys", "join functionalities", "SQL"],
            "user_interests": ["query writing", "relational databases", "SQL"],
            "learning_progress": "Confident with inner joins; confused about NULL handling in outer joins. Needs more practice on join keys and SQL.",
            "summary": "The student worked through several inner-join queries..."
            }
        """).strip(),

        "detailed": textwrap.dedent("""
            The conversation was long (9+ user/assistant pairs).
            - chat_focus: one sentence capturing the overarching theme
            - key_concepts: up to 10 items, ordered by importance
            - user_interests: up to 6 items
            - learning_progress: 2-3 sentences covering what was mastered,
              what was revisited, and any apparent misconceptions
            - summary: thorough prose recap, under 220 words
            
            Example output for a conversation about SQL joins:
            {
            "chat_focus": "Understanding inner vs. outer joins in SQL, and exploring when and why to use each type of join. ",
            "key_concepts": ["inner join", "left outer join", "join keys", "join functionalities", "SQL", "optimizers"],
            "user_interests": ["query writing", "relational databases", "optimizers", "SQL"],
            "learning_progress": "Confident with inner joins; confused about NULL handling in outer joins. Looking into the role of optimizers...",
            "summary": "The student worked through several inner-join queries..."
            }
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
    try:
        with open(SAVED_CHATS_FILE, 'r') as f:
            content = f.read().strip()
            if not content:   # file is empty
                return []
            return json.loads(content)
    except FileNotFoundError:
        open(SAVED_CHATS_FILE, "a").close()
        return []
    except json.JSONDecodeError:
        return [] 

def _log_has_history_personalization(path: Path) -> bool:
    """
    Return whether the log file at `path` was produced with
    enable_history_personalization=True.

    Defaults to True if the field is missing or the file is unreadable —
    we'd rather over-recover than silently drop a valid session.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return True  # assume recoverable; downstream will skip if malformed
    return bool(
        data.get('config_state', {}).get('enable_history_personalization', True)
    )

def _parse_log_timestamp(filename: str) -> Optional[float]:
    stem = filename.rsplit('.', 1)[0]
    if not stem.startswith('chat_'):
        return None
    try:
        local_dt = datetime.strptime(stem[len('chat_'):], '%Y%m%d_%H%M%S')
        return local_dt.timestamp()
    except ValueError:
        return None


def _last_saved_unix_timestamp(saved: list) -> Optional[float]:
    latest = None
    for entry in saved:
        ts_str = entry.get('metadata', {}).get('timestamp', '')
        if not ts_str:
            continue
        try:
            utc_dt = datetime.fromisoformat(ts_str.rstrip('Z')).replace(tzinfo=timezone.utc)
            ts = utc_dt.timestamp()
        except ValueError:
            continue
        if latest is None or ts > latest:
            latest = ts
    return latest


def _reconstruct_chat_from_logs(log_paths: list) -> list:
    chat = []
    for path in log_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        user_msg      = data.get('original_query') or data.get('query')
        assistant_msg = data.get('full_response')
        if not user_msg or not assistant_msg:
            continue
        chat.append({'role': 'user',      'content': user_msg})
        chat.append({'role': 'assistant', 'content': assistant_msg})
    return chat


def recover_last_unsaved_chat(cfg) -> Optional[list]:
    logs_dir = Path(LOGS_DIR)
    if not logs_dir.is_dir():
        return None

    candidates = []
    for path in logs_dir.glob('chat_*.json'):
        ts = _parse_log_timestamp(path.name)
        if ts is not None:
            candidates.append((ts, path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])

    cutoff = _last_saved_unix_timestamp(load_saved_chats())
    if cutoff is not None:
        candidates = [(ts, p) for ts, p in candidates if ts > cutoff]
    if not candidates:
        return None

    sessions = [[candidates[0]]]
    for i in range(1, len(candidates)):
        if candidates[i][0] - candidates[i - 1][0] > SESSION_GAP_SECONDS:
            sessions.append([])
        sessions[-1].append(candidates[i])
    last_session_paths = [p for _, p in sessions[-1]]
    
    most_recent_log = last_session_paths[-1]
    if not _log_has_history_personalization(most_recent_log):
        return None

    chat = _reconstruct_chat_from_logs(last_session_paths)
    if not chat:
        return None

    update_saved_chats(chat, cfg)
    return chat

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
    
    print("Raw LLM Output for Chat Summarization:")
    
    print(result)
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
    entry = process_chat(new_chat, cfg)
    saved_chats.append(entry)

    with open(SAVED_CHATS_FILE, 'w') as f:
        json.dump(saved_chats, f, indent=4)
    update_biodata_skills(cfg.gen_model, new_chat)

    return saved_chats