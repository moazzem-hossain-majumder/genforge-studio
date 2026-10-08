"""
Lab 5: Voice assistant.

Original lab used OpenAI GPT for the reply and IBM Watson STT/TTS
(sandbox-only URLs). Replaced with: Whisper (STT) + shared instruct LLM
(reply) + MMS-TTS (TTS) — all local, all free.
"""

import numpy as np
import gradio as gr

from core.models import get_whisper, get_tts, instruct_chat


def transcribe(audio_path):
    if not audio_path:
        return ""
    pipe = get_whisper()
    return pipe(audio_path, batch_size=8)["text"]


def synthesize_speech(text):
    if not text:
        return None
    tokenizer, model = get_tts()
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with __import__("torch").no_grad():
        output = model(**inputs).waveform
    audio = output.squeeze().cpu().numpy()
    return (model.config.sampling_rate, audio)


def voice_assistant_turn(audio_path, history):
    history = history or []
    user_text = transcribe(audio_path)
    if not user_text.strip():
        return history, None, "Didn't catch that — try again."

    messages = [
        {
            "role": "system",
            "content": (
                "Act like a personal assistant. You can answer questions, "
                "translate sentences, summarize things, and give recommendations. "
                "Keep replies brief."
            ),
        }
    ]
    for user_msg, bot_msg in history[-5:]:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": user_text})

    reply_text = instruct_chat(messages, max_new_tokens=120)
    reply_audio = synthesize_speech(reply_text)

    history = history + [(user_text, reply_text)]
    return history, reply_audio, reply_text


def build_tab():
    with gr.Tab("🎙️ Voice Assistant"):
        gr.Markdown(
            "### Voice Assistant\n"
            "Record a question. It's transcribed locally (Whisper), answered "
            "by a local LLM, and spoken back (local TTS) — no OpenAI/Watson."
        )
        chat_state = gr.State([])
        audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speak")
        ask_btn = gr.Button("Ask")
        chat_out = gr.Chatbot(label="Conversation")
        audio_out = gr.Audio(label="Spoken reply", autoplay=True)
        reply_text_out = gr.Textbox(visible=False)

        ask_btn.click(
            voice_assistant_turn,
            inputs=[audio_in, chat_state],
            outputs=[chat_state, audio_out, reply_text_out],
        ).then(lambda h: h, inputs=chat_state, outputs=chat_out)
