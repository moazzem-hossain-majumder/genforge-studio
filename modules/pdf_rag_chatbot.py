"""
Lab 7: PDF Chatbot (RAG).

Original lab used ChatWatsonx (watsonx.ai). Replaced with a local
HuggingFace pipeline wrapped for LangChain, and the shared embeddings
model. Vector store is Chroma, same as the lab.
"""

import gradio as gr

from core.models import get_embeddings, INSTRUCT_LLM_MODEL_ID

_state = {"chain": None}


def _build_local_llm():
    from transformers import pipeline
    from langchain_huggingface import HuggingFacePipeline

    hf_pipe = pipeline(
        "text-generation",
        model=INSTRUCT_LLM_MODEL_ID,
        max_new_tokens=300,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.2,
    )
    return HuggingFacePipeline(pipeline=hf_pipe)


def process_pdf(pdf_file):
    if pdf_file is None:
        return "Please upload a PDF."

    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain.chains import RetrievalQA

    loader = PyPDFLoader(pdf_file.name if hasattr(pdf_file, "name") else pdf_file)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(documents)

    embeddings = get_embeddings()
    db = Chroma.from_documents(texts, embedding=embeddings, persist_directory="./chroma_db")

    llm = _build_local_llm()
    _state["chain"] = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr", search_kwargs={"k": 3}),
        return_source_documents=False,
        input_key="question",
    )
    return f"Indexed {len(texts)} chunks from the PDF. Ask away below."


def ask_pdf(question, history):
    if _state["chain"] is None:
        return history + [(question, "Please upload and process a PDF first.")]
    output = _state["chain"].invoke({"question": question})
    answer = output["result"]
    return history + [(question, answer)]


def build_tab():
    with gr.Tab("📄 PDF Chatbot (RAG)"):
        gr.Markdown(
            "### Chat With Your PDF\n"
            "Upload a PDF, then ask questions about it. Retrieval-augmented "
            "generation using local embeddings + Chroma + a local LLM."
        )
        pdf_in = gr.File(label="PDF file", file_types=[".pdf"])
        process_btn = gr.Button("Process PDF")
        status_out = gr.Textbox(label="Status")
        process_btn.click(process_pdf, inputs=pdf_in, outputs=status_out)

        chatbot = gr.Chatbot(label="Ask about the document")
        question_in = gr.Textbox(label="Your question")
        ask_btn = gr.Button("Ask")
        ask_btn.click(ask_pdf, inputs=[question_in, chatbot], outputs=chatbot)
