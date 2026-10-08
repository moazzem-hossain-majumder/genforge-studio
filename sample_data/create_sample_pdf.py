import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

pdf_path = os.path.join(os.path.dirname(__file__), "project_report.pdf")
doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)

styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    "DocTitle",
    parent=styles["Heading1"],
    fontSize=22,
    leading=26,
    textColor=colors.HexColor("#1e293b"),
    spaceAfter=12
)
heading2_style = ParagraphStyle(
    "DocHeading2",
    parent=styles["Heading2"],
    fontSize=14,
    leading=18,
    textColor=colors.HexColor("#0f766e"),
    spaceBefore=14,
    spaceAfter=6
)
body_style = ParagraphStyle(
    "DocBody",
    parent=styles["BodyText"],
    fontSize=10,
    leading=14,
    textColor=colors.HexColor("#334155"),
    spaceAfter=8
)

story = [
    Paragraph("GenForge Studio: Architecture & Capabilities Overview", title_style),
    Paragraph("<b>Author:</b> GenForge Engineering Team | <b>Version:</b> 1.0.0", body_style),
    Spacer(1, 10),
    Paragraph("1. Executive Summary", heading2_style),
    Paragraph("GenForge Studio is a unified generative AI desktop platform created to bring state-of-the-art open-source AI capabilities to local workstations without requiring paid cloud API keys or internet access during inference. All processing runs entirely on local GPU and CPU hardware.", body_style),
    Paragraph("2. Core Systems & Architecture", heading2_style),
    Paragraph("The platform integrates multiple specialized foundation models into a unified runtime: BLIP for multimodal vision captioning, Blenderbot for conversational dialogue, SmolLM2 for instruction-following and document summarization, OpenAI Whisper Tiny for offline speech-to-text, and MMS-TTS for multi-language vocal synthesis.", body_style),
    Paragraph("3. Retrieval-Augmented Generation (RAG)", heading2_style),
    Paragraph("The document question-answering module employs LangChain coupled with ChromaDB vector storage and all-MiniLM-L6-v2 sentence embeddings. Incoming PDFs are split into 1000-character segments with 100-character semantic overlaps, indexed into local vector indices, and queried through MMR (Maximal Marginal Relevance) retrieval.", body_style),
    Paragraph("4. Performance Benchmarks & Milestones", heading2_style),
    Paragraph("In initial testing on modern RTX hardware, average response latency was under 800ms for chat interactions and under 1.5 seconds for audio transcription. Total VRAM footprint across active concurrent modules remains bounded under 4 gigabytes.", body_style),
]

doc.build(story)
print(f"Created sample PDF at: {pdf_path}")
