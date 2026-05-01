"""
Query enhancement techniques for improved retrieval (use only one):
- HyDE (Hypothetical Document Embeddings): Generate hypothetical answer for better retrieval
- Query Enrichment: LLM-based query expansion
"""

import json
import textwrap
from pathlib import Path
from typing import Optional
from src.generator import ANSWER_END, ANSWER_START, run_llama_cpp, text_cleaning



def generate_hypothetical_document(
    query: str,
    model_path: str,
    max_tokens: int = 100,
    **llm_kwargs
) -> str:
    """
    HyDE: Generate a hypothetical answer to improve retrieval quality.
    Concept: Hypothetical answers are semantically closer to actual documents than queries.
    Ref: https://arxiv.org/abs/2212.10496
    """
    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a database systems expert. Generate a concise, technical answer using precise database terminology.
        Write in the formal academic style of Database System Concepts (Silberschatz, Korth, Sudarshan).
        Use specific terms for: relational model concepts (relations, tuples, attributes, keys, schemas), 
        SQL and query languages, transactions (ACID properties, concurrency control, recovery), 
        storage structures (indexes, B+ trees), normalization (functional dependencies, normal forms), 
        and database design (E-R model, decomposition).
        Focus on definitions, mechanisms, and technical accuracy rather than examples.
        <|im_end|>
        <|im_start|>user
        Question: {query}
        
        Generate a precise and a concise answer (2-4 sentences) using appropriate technical terminology. End with {ANSWER_END}.
        <|im_end|>
        <|im_start|>assistant
        {ANSWER_START}
        """)
    
    prompt = text_cleaning(prompt)
    hypothetical = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=max_tokens,
        **llm_kwargs
    )
    
    return hypothetical.strip()

def correct_query_grammar(
    query: str,
    model_path: str,
    **llm_kwargs
) -> str:
    """
    Corrects spelling and grammatical errors in the query to improve keyword matching.
    """
    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a helpful assistant that corrects search queries.
        Your task is to correct any spelling or grammatical errors in the user's query.
        Do not answer the question. Output ONLY the corrected query.
        <|im_end|>
        <|im_start|>user
        Original Query: {query}
        <|im_end|>
        <|im_start|>assistant
        """)

    prompt = text_cleaning(prompt)
    corrected_query = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=len(query.split()) * 2,
        temperature=0,
        **llm_kwargs
    )

    # If model returns empty or hallucinated long text, return original
    cleaned = corrected_query["choices"][0]["text"].strip()
    if not cleaned or len(cleaned) > len(query) * 2:
        return query

    return cleaned

def expand_query_with_keywords(
    query: str,
    model_path: str,
    max_tokens: int = 64,
    **llm_kwargs
) -> str:
    """
    Query Expansion: Generates related keywords and synonyms.
    This helps retrieval when the user uses different vocabulary than the documents.
    """
    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a search optimization expert.
        Generate 3 alternative versions of the user's query using synonyms and related technical terms.
        Output the alternative queries separated by newlines. Do not provide explanations.
        <|im_end|>
        <|im_start|>user
        Query: {query}
        <|im_end|>
        <|im_start|>assistant
        """)

    prompt = text_cleaning(prompt)
    expansion = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=max_tokens,
        temperature=0.5,
        **llm_kwargs
    )

    query_lines = [query]
    query_lines.extend([line.strip() for line in expansion["choices"][0]["text"].split('\n') if line.strip()])

    query_lines = [line.split('.', 1)[-1].strip() if '.' in line[:3] else line for line in query_lines]

    return query_lines



def load_recent_chat_summaries(
    n: int = 10,
    saved_chats_path: str = "saved_chats.json",
) -> list[dict]:
    """
    Load the n most recent chat summary entries from saved_chats.json,
    sorted by timestamp (newest last so the LLM reads them chronologically).
    """
    path = Path(saved_chats_path)
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            saved_chats = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    if not saved_chats:
        return []

    sorted_chats = sorted(
        saved_chats,
        key=lambda e: e.get("metadata", {}).get("timestamp", ""),
    )

    return sorted_chats[-n:]


def _format_chat_summaries(entries: list[dict]) -> str:
    """
    Render the list of chat summary entries into a compact, LLM-readable block.
    Each entry exposes only the fields useful for query personalization.
    """
    lines = []
    for i, entry in enumerate(entries, 1):
        meta    = entry.get("metadata", {})
        summary = entry.get("summary",  {})

        timestamp = meta.get("timestamp", "unknown time")
        focus     = summary.get("chat_focus",        "N/A")
        concepts  = summary.get("key_concepts",      [])
        interests = summary.get("user_interests",    [])
        progress  = summary.get("learning_progress", "N/A")

        lines.append(f"[Session {i} — {timestamp}]")
        lines.append(f"  Focus     : {focus}")
        lines.append(f"  Concepts  : {', '.join(concepts) if concepts else 'none'}")
        lines.append(f"  Interests : {', '.join(interests) if interests else 'none'}")
        lines.append(f"  Progress  : {progress}")
        lines.append("")  # blank line between sessions

    return "\n".join(lines).strip()


def enhance_query_with_chat_history(
    query: str,
    model_path: str,
    saved_chats_path: str = "saved_chats.json",
    n_sessions: int = 10,
    max_tokens: int = 128,
    **llm_kwargs,
) -> str:
    """
    Chat-History Enhancement: Rewrites a query using patterns extracted from
    the student's last n_sessions summarized chat sessions.

    Looks for recurring topics, known struggles, and established interests
    across sessions to surface the most relevant retrieval context — without
    changing the query's core intent.

    Parameters
    ----------
    query             : The raw user query to enhance.
    model_path        : Path to the llama.cpp model.
    saved_chats_path  : Path to saved_chats.json (default: "saved_chats.json").
    n_sessions        : How many recent sessions to consider (default: 10).
    max_tokens        : Max tokens for the rewritten query.
    """
    entries = load_recent_chat_summaries(n=n_sessions, saved_chats_path=saved_chats_path)

    if not entries:
        return query

    history_block = _format_chat_summaries(entries)

    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a search query optimizer for an educational database systems assistant.
        You have access to a log of the student's recent chat sessions, each summarized
        with a focus topic, key concepts covered, personal interests, and learning progress.

        Your task is to rewrite the student's query to better reflect:
        - Topics they have been repeatedly studying (recurring key_concepts)
        - Areas where they have struggled or shown gaps (learning_progress)
        - Their demonstrated interests across sessions (user_interests)

        Guidelines:
        - Do NOT answer the question. Output ONLY the rewritten query.
        - Keep it concise (one sentence if possible).
        - Only add context that is genuinely relevant — if the history adds nothing
          useful for this query, return the original query unchanged.
        - Do not reference the chat history explicitly in the rewritten query
          (e.g. do not write "based on your previous sessions...").

        Examples:
        History shows: repeated questions on B+ trees, struggles with index selection
        Original : How are indexes used?
        Rewritten: How do B+ tree indexes work and how should an index be selected for a given query workload?

        History shows: interest in transactions, recent confusion about isolation levels
        Original : What is serialisability?
        Rewritten: What is serializability and how does it relate to the isolation levels a student recently studied?
        <|im_end|>
        <|im_start|>user
        Recent Session History (oldest → newest):
        {history_block}

        Original Query: {query}
        <|im_end|>
        <|im_start|>assistant
        Rewritten Query:
        """)

    prompt = text_cleaning(prompt)
    
    print("Prompt for Query Enhancement:")
    print(prompt)
    output = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=max_tokens,
        temperature=0.1,
        **llm_kwargs,
    )
    print("Raw LLM Output for Query Enhancement using Biodata:")
    print(output)

    rewritten = output["choices"][0]["text"].strip()

    for prefix in ("rewritten query:", "query:"):
        if rewritten.lower().startswith(prefix):
            rewritten = rewritten[len(prefix):].strip()
            
    print(f"Enhanced Query: {rewritten}")

    if not rewritten or len(rewritten) > len(query) * 4:
        return query

    return rewritten



def personalize_query(
    query: str,
    model_path: str,
    biodata_path: str = "biodata.md",
    max_tokens: int = 128,
    **llm_kwargs
) -> str:
    """
    Query Personalization: Rewrites a query using the student's biodata profile
    loaded from a biodata.md file (generated by generate_biodata.py).

    Surfaces retrieval results appropriate for the student's background,
    skill level, and learning context without changing the query's core intent.
    """
    path = Path(biodata_path)
    if not path.exists():
        return query

    content = path.read_text(encoding="utf-8")

    profile = {}
    current_label = None
    buffer = []

    for line in content.splitlines():
        if line.startswith("## "):
            if current_label is not None:
                profile[current_label] = "\n".join(buffer).strip()
            current_label = line[3:].strip()
            buffer = []
        elif line.startswith("# "):
            continue
        else:
            if current_label is not None:
                buffer.append(line)

    if current_label is not None:
        profile[current_label] = "\n".join(buffer).strip()
 
    
    personal_info = profile

    if not personal_info:
        print("No personal information found in biodata.md for query personalization.")
        return query

    _EMPTY = {"unknown", "none", "n/a", "not mentioned", "not provided", ""}
    profile_lines = "\n".join(
        f"- {label.replace('_', ' ').title()}: {value}"
        for label, value in personal_info.items()
        if value and value.strip().lower() not in _EMPTY
    )

    if not profile_lines:
        return query

    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a search query optimizer for an educational database systems assistant.
        Your task is to rewrite a student's query to better reflect their background,
        skill level, and learning context — without changing the core intent of the question.

        Guidelines:
        - Add relevant context clues that help retrieve appropriately-leveled content.
        - Do NOT answer the question. Output ONLY the rewritten query.
        - Keep it concise (one sentence if possible).
        - If the profile adds no useful context, return the original query unchanged.
        - Always ensure the rewritten query is a valid standalone question that could be answered by a document, not a conversation.
        - Use any test scores to indicate the level of detail or technicality appropriate for the student.
        - Use any interests if they suggest a particular angle or application area for the question.

        Examples:
        Profile: sophomore CS student, took Intro to Databases, struggles with normalization
        Original: What is BCNF?
        Rewritten: What is BCNF and how does it differ from 3NF, for a student who understands functional dependencies?

        Profile: strong in Python, no prior SQL experience
        Original: How do joins work?
        Rewritten: How do SQL joins work for someone coming from a Python background with no prior SQL experience?
        <|im_end|>
        <|im_start|>user
        Student Profile:
        {profile_lines}

        Original Query: {query}
        <|im_end|>
        <|im_start|>assistant
        Rewritten Query:
        """)

    prompt = text_cleaning(prompt)
    output = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=max_tokens,
        temperature=0.1,
        **llm_kwargs
    )
    
    print("Raw LLM Output for Query Enhancement using Biodata:")
    print(output)

    rewritten = output["choices"][0]["text"].strip()

    for prefix in ("rewritten query:", "query:"):
        if rewritten.lower().startswith(prefix):
            rewritten = rewritten[len(prefix):].strip()

    if not rewritten or len(rewritten) > len(query) * 4:
        return query

    return rewritten

def decompose_complex_query(
    query: str,
    model_path: str,
    **llm_kwargs
) -> list[str]:
    """
    Breaks a complex multi-part question into sub-questions.
    Useful for tasks where a single retrieval might miss some parts of the answer.
    """
    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        Break the following complex question into simple, single-step sub-questions.
        If the question is already simple, just output the original question.
        Output each sub-question on a new line. Do not provide explanations.
        <|im_end|>
        <|im_start|>user
        Complex Question: {query}
        <|im_end|>
        <|im_start|>assistant
        """)

    prompt = text_cleaning(prompt)
    output = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=128,
        temperature=0.0,
        **llm_kwargs
    )

    sub_questions = [line.strip() for line in output["choices"][0]["text"].split('\n') if line.strip()]

    # Remove numbering if present
    sub_questions = [line.split('.', 1)[-1].strip() if '.' in line[:3] else line for line in sub_questions]

    return sub_questions

def contextualize_query(
    query: str,
    history: list[dict],
    model_path: str,
    max_tokens: int = 128,
    **llm_kwargs
) -> str:
    """
    Rewrites a query to be standalone based on chat history.
    """
    if not history:
        return query

    # Format history into a compact string
    # We expect history to be list of dicts: [{"role": "user", "content": "..."}, ...]
    conversation_text = ""
    for turn in history[-4:]: # Only look at last 2 turns
        role = "User" if turn["role"] == "user" else "Assistant"
        content = turn["content"]
        conversation_text += f"{role}: {content}\n"

    prompt = textwrap.dedent(f"""\
        <|im_start|>system
        You are a query rewriting assistant. Your task is to rewrite the user's "Follow Up Input" to be a standalone question by replacing pronouns (it, they, this, that) with specific nouns from the "Chat History".
        
        Examples:
        History: 
        User: What is BCNF?
        Assistant: It is a normal form used in database normalization.
        Input: Why is it useful?
        Output: Why is BCNF useful?
        
        History:
        User: Explain the ACID properties.
        Assistant: ACID stands for Atomicity, Consistency, Isolation, Durability.
        Input: Give me an example of the first one.
        Output: Give me an example of Atomicity.

        History:
        User: Who created Python?
        Assistant: Guido van Rossum.
        Input: what is sql?
        Output: what is sql?
        <|im_end|>
        <|im_start|>user
        Chat History:
        {conversation_text}
        
        Follow Up Input: {query}
        
        Output:
        <|im_end|>
        <|im_start|>assistant
        """)

    prompt = text_cleaning(prompt)
    output = run_llama_cpp(
        prompt,
        model_path,
        max_tokens=max_tokens,
        temperature=0.1,
        **llm_kwargs
    )

    rewritten = output["choices"][0]["text"].strip()
    
    # If model hallucinates or errors, fall back to original query
    if not rewritten or len(rewritten) > len(query) * 2:
        return query
        
    return rewritten