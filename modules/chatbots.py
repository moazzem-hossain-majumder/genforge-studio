"""Lab 3: Simple chatbot (Blenderbot) + modern LLM chatbot (SmolLM2)."""

import gradio as gr

from core.models import get_blenderbot, instruct_chat


# ---- Simple seq2seq chatbot (Blenderbot) -----------------------------------

def blenderbot_reply(message, history):
    tokenizer, model = get_blenderbot()

    history_lines = []
    for user_msg, bot_msg in history[-3:]:
        history_lines.append(f"User: {user_msg}")
        if bot_msg:
            history_lines.append(f"Bot: {bot_msg}")
    history_string = "\n".join(history_lines)

    prompt = history_string + f"\nUser: {message}\nBot:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
        do_sample=True,
        temperature=0.6,
        top_p=0.85,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ---- Modern instruct LLM chatbot (SmolLM2) --------------------------------

def llm_chat_reply(message, history):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Give short and concise responses.",
        }
    ]
    for user_msg, bot_msg in history[-5:]:
        messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    return instruct_chat(messages, max_new_tokens=80, temperature=0.5)


def build_tab():
    with gr.Tab("💬 Chatbots"):
        gr.Markdown("### Simple Chatbot vs. Modern LLM Chatbot")
        with gr.Tab("Simple (Blenderbot, seq2seq)"):
            gr.ChatInterface(fn=blenderbot_reply, type="tuples")
        with gr.Tab("Modern (SmolLM2, instruct)"):
            gr.ChatInterface(fn=llm_chat_reply, type="tuples")
