"""
GenForge Studio — one Gradio app covering every lab from
"Building Generative AI-Powered Applications with Python", using only
free, local models (no watsonx/OpenAI/Watson API keys needed).

Run: python app.py
"""

import gradio as gr

from modules import (
    image_captioning,
    chatbots,
    voice_assistant,
    meeting_companion,
    pdf_rag_chatbot,
    voice_translator,
    career_tools,
)

with gr.Blocks(title="GenForge Studio") as demo:
    gr.Markdown(
        "# 🧠 GenForge Studio\n"
        "Every lab from *Building Generative AI-Powered Applications with "
        "Python*, running on free local models — no API keys required. "
        "Models load lazily the first time you use each tab."
    )

    image_captioning.build_tab()
    chatbots.build_tab()
    voice_assistant.build_tab()
    meeting_companion.build_tab()
    pdf_rag_chatbot.build_tab()
    voice_translator.build_tab()
    career_tools.build_tab()

    gr.Markdown(
        "---\n"
        "*Note: the Flask + JS chatbot lab lives separately in "
        "`flask_chat_app/` — run it with `python flask_chat_app/app.py`.*"
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
