"""
Lab 6: Business AI Meeting Companion.

Original lab used watsonx.ai (Llama-4-Maverick) for summarization.
Replaced with the shared local instruct LLM.
"""

import gradio as gr

from core.models import get_whisper, instruct_chat


def summarize_meeting(audio_file):
    if not audio_file:
        return "", "Please upload an audio file."

    pipe = get_whisper()
    transcript = pipe(audio_file, batch_size=8)["text"]

    messages = [
        {
            "role": "system",
            "content": "You summarize meeting transcripts into a concise bulleted list of key points.",
        },
        {
            "role": "user",
            "content": f"List the key points with details from the following context:\n{transcript}",
        },
    ]
    key_points = instruct_chat(messages, max_new_tokens=400, temperature=0.2)
    return transcript, key_points


def build_tab():
    with gr.Tab("📝 Meeting Companion"):
        gr.Markdown(
            "### Business AI Meeting Companion\n"
            "Upload meeting audio (mp3/wav). It's transcribed (Whisper) and "
            "summarized into key points by a local LLM."
        )
        audio_in = gr.Audio(sources=["upload"], type="filepath", label="Meeting audio")
        run_btn = gr.Button("Transcribe & summarize")
        transcript_out = gr.Textbox(label="Transcript", lines=8)
        summary_out = gr.Textbox(label="Key points", lines=10)
        run_btn.click(summarize_meeting, inputs=audio_in, outputs=[transcript_out, summary_out])
