import textwrap
import json
from generator import text_cleaning, get_llama_model, run_llama_cpp, answer, format_prompt

SAVED_CHATS_FILE = 'saved_chats.json'
def load_saved_chats():
    """
    Load the saved chats from the JSON file.

    Returns:
        list: The list of saved chats.
    """
    try:
        with open(SAVED_CHATS_FILE, 'r') as f:
            saved_chats = json.load(f)
    except FileNotFoundError:
        open(SAVED_CHATS_FILE, "a")
        saved_chats = []
    return saved_chats



def process_chat(chat, cfg):
    """
    Process a chat to ensure it has the required structure.

    Args:
        chat (dict): The chat to process.

    Returns:
        dict: The processed chat with 'question' and 'answer' keys.
    """
    
    def get_system_prompt(mode = None):
        return textwrap.dedent(f"""
                You are a comprehensive summarizer. Provide a brief, thorough summary given the recorded conversation.
                - Explain the main idea and takeaways clearly and concisely
                - Mention the key concepts the user learned in the conversation
                - Mention what the user is interested in based on the conversation
                - Keep your response within 150 words
                End your reply with {ANSWER_END}.
            """).strip()
    result = run_llama_cpp(chat, cfg.gen_model, max_tokens=cfg.hyde_max_tokens)
    return result["choices"][0]["text"].strip()

def update_saved_chats(new_chat, cfg):
    """
    Update the saved chats with a new chat.

    Args:
        saved_chats (list): The list of saved chats.
        new_chat (dict): The new chat to add to the saved chats.

    Returns:
        list: The updated list of saved chats.
    """
    saved_chats = load_saved_chats()
    saved_chats.append(process_chat(new_chat))
    with open(SAVED_CHATS_FILE, 'w') as f:
        json.dump(saved_chats, f, indent=4)
    return saved_chats