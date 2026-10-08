# Build GenForge Studio From Scratch — A Learning Guide

This is not a copy-paste reference. It's written so you build the project
one working piece at a time, understanding *why* each piece exists, in the
order a developer would naturally add them. By the end you'll have built
the same project you already have — but you'll understand every line.

Work through this in order. Each chapter ends with a **checkpoint**: run
something, see it work, before moving on. Don't skip checkpoints — a
broken checkpoint is much easier to debug than five chapters of broken code.

---

## Part 0 — Concepts you need before writing any code

Skim this once, then refer back as these words come up later.

**Virtual environment (venv).** A private, isolated copy of Python +
installed packages, separate from the rest of your system. Every Python
project should have its own, so Project A needing `transformers==4.41.2`
and Project B needing `transformers==5.0.0` don't collide.

**Hugging Face & "models."** Hugging Face hosts pre-trained AI models —
files containing millions/billions of learned numbers (weights) plus config.
A "model ID" like `Salesforce/blip-image-captioning-base` is just an address
on huggingface.co. The `transformers` Python library downloads and runs
these for you.

**Tokenizer vs. model.** Models don't understand text directly — they
understand numbers. A **tokenizer** converts text ↔ numbers (token IDs). You
almost always load a matching tokenizer + model pair, from the same model ID.

**Pipeline vs. manual loading.** `transformers.pipeline(...)` is a
convenience wrapper that hides the tokenizer/model split for common tasks
(e.g. `"automatic-speech-recognition"`). Manual loading
(`AutoModelForCausalLM.from_pretrained(...)`) gives you more control —
needed when you want to customize generation behavior, as we will.

**CPU vs. GPU / CUDA.** Models run as either: CPU (works everywhere, slower)
or GPU via CUDA (NVIDIA-only, much faster, limited by VRAM). PyTorch
(`torch`) is the library underneath `transformers` that actually does the
math; a "CUDA build" of torch is required for GPU to work at all.

**Gradio.** A Python library that turns functions into web UIs.
`gr.Interface(fn=my_function, inputs=..., outputs=...)` is the simplest
form. `gr.Blocks()` with `gr.Tab(...)` inside gives you multi-tab layouts —
that's what this project uses, since it has many features in one app.

**LangChain, embeddings, RAG, vector store.** To let an LLM answer
questions about *your* PDF (which it was never trained on), you: (1) split
the PDF into chunks, (2) convert each chunk to a list of numbers
("embedding") that captures its meaning, (3) store those in a **vector
store** (Chroma here), (4) when the user asks a question, embed the
question too and find the most similar chunks, (5) stuff those chunks into
the LLM's prompt as context. This whole pattern is **Retrieval-Augmented
Generation (RAG)**. LangChain is a library that wires these steps together.

**Flask.** A minimal Python web framework for building your own HTTP
routes (`@app.route("/chatbot")`) — used here for the one standalone
non-Gradio piece, so you also see how a backend API + separate JS frontend
works, not just Gradio's all-in-one style.

---

## Part 1 — Environment setup

**1.1** Install Python 3.10+, VS Code, and Git if you don't have them.

**1.2** Create the project folder and a virtual environment:

```powershell
mkdir genforge-studio
cd genforge-studio
python -m venv my_env
my_env\Scripts\activate
```

Your prompt should now start with `(my_env)`. Every `pip install` from
here on installs into this isolated environment, not your system Python.

**1.3** Install ffmpeg (needed later for audio):
```powershell
winget install ffmpeg
```

**Checkpoint:** run `python --version` and `ffmpeg -version` — both should
print version info with no errors.

---

## Part 2 — Your first working piece: Image Captioning

We start here because it's the smallest model (fast to download) and
touches the two libraries you'll use everywhere: `transformers` and `gradio`.

**2.1** Install the first round of packages:
```powershell
pip install gradio transformers torch Pillow numpy
```

**2.2** Create `app.py` with just this:

```python
import numpy as np
import gradio as gr
from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.ndarray) -> str:
    raw_image = Image.fromarray(input_image).convert("RGB")
    inputs = processor(images=raw_image, text="the image of", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    return processor.decode(outputs[0], skip_special_tokens=True)

demo = gr.Interface(fn=caption_image, inputs=gr.Image(), outputs="text")
demo.launch()
```

Read it top to bottom: load a processor (tokenizer-equivalent for images)
and model once, at startup. `caption_image` is a plain Python function —
Gradio's only job is wrapping it in a web form. `processor(...)` turns the
image into tensors the model understands; `model.generate(...)` runs
inference; `processor.decode(...)` turns the output tokens back into text.

**Checkpoint:** `python app.py`, open the printed URL, upload any photo,
click Submit. First run downloads ~1GB — expect a wait. You should get a
one-sentence caption back.

**2.3 — Now understand the design problem this creates.** If you keep
adding features this way (loading models at the top of the file), by the
time you have 7 features sharing 2-3 of the *same* underlying models
(e.g. the chatbot LLM used in 4 different tabs), you'd load it 4 times —
wasting gigabytes of RAM for no reason. Next chapter fixes this *before*
it becomes a problem, not after.

---

## Part 3 — The shared model registry pattern

This is the single most important architectural idea in this project.
Get it right here and every later feature just plugs into it.

**3.1** Create a `core/` folder with an empty `core/__init__.py` and a
`core/models.py`. The idea: **one function per model, each returning a
cached instance** — load once, reuse everywhere.

```python
_cache = {}

def get_blip():
    if "blip" not in _cache:
        from transformers import AutoProcessor, BlipForConditionalGeneration
        processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        _cache["blip"] = (processor, model)
    return _cache["blip"]
```

The trick: `_cache` is a plain dict at module level, so it persists for
the life of the program. First call loads and stores; every call after
that just returns the stored tuple. The `import` is *inside* the function
on purpose — that's "lazy loading": the `transformers` classes aren't even
imported until this specific function is first called, so starting the app
is instant and nothing downloads until a tab is actually used.

**3.2** Move your image captioning code to use it — create
`modules/__init__.py` (empty) and `modules/image_captioning.py`:

```python
import numpy as np
import gradio as gr
from PIL import Image
from core.models import get_blip

def caption_image(input_image: np.ndarray) -> str:
    processor, model = get_blip()
    raw_image = Image.fromarray(input_image).convert("RGB")
    inputs = processor(images=raw_image, text="the image of", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    return processor.decode(outputs[0], skip_special_tokens=True)

def build_tab():
    with gr.Tab("🖼️ Image Captioning"):
        img_in = gr.Image(label="Upload an image")
        btn = gr.Button("Generate caption")
        out = gr.Textbox(label="Caption")
        btn.click(caption_image, inputs=img_in, outputs=out)
```

Notice the new shape: a `build_tab()` function that constructs the UI, and
plain functions that do the work. This is the pattern *every* module in
the project follows — you'll repeat it, not reinvent it, for the rest of
this guide.

**3.3** Rewrite `app.py` to assemble tabs instead of holding logic:

```python
import gradio as gr
from modules import image_captioning

with gr.Blocks(title="GenForge Studio") as demo:
    gr.Markdown("# 🧠 GenForge Studio")
    image_captioning.build_tab()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

**Checkpoint:** run it again, confirm captioning still works exactly as
before. Nothing user-facing changed — you just gave the code a shape that
scales. That's the whole point of this chapter.

---

## Part 4 — Second feature: Chatbots (why the registry pays off)

Now add two chatbots — a small seq2seq model (Blenderbot) and a modern
instruct-tuned LLM (SmolLM2/Phi-3). This is where you'll *feel* why
Part 3 mattered: this instruct LLM gets reused by four more features later.

**4.1** Install: `pip install accelerate sentencepiece`

**4.2** Add two more getters to `core/models.py`:

```python
def get_blenderbot():
    if "blenderbot" not in _cache:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
        _cache["blenderbot"] = (tokenizer, model)
    return _cache["blenderbot"]

def get_instruct_llm():
    if "instruct_llm" not in _cache:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        _cache["instruct_llm"] = (tokenizer, model)
    return _cache["instruct_llm"]
```

**4.3** The instruct LLM needs a chat-formatting step every other feature
using it will also need — so write it *once* as a shared helper, in the
same file:

```python
def instruct_chat(messages, max_new_tokens=300, temperature=0.6):
    """messages: list of {"role": "system"/"user"/"assistant", "content": str}"""
    import torch
    tokenizer, model = get_instruct_llm()
    tokenized = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True, max_length=1024, truncation=True,
    )
    with torch.inference_mode():
        outputs = model.generate(
            tokenized["input_ids"], attention_mask=tokenized["attention_mask"],
            max_new_tokens=max_new_tokens, temperature=temperature, top_p=0.9,
            do_sample=True, repetition_penalty=1.3, no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
        )
    input_length = tokenized["input_ids"].shape[-1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()
```

`apply_chat_template` is doing something specific: instruct models are
trained on conversations formatted a particular way (special tokens marking
who's speaking). This function handles that formatting for you from a
plain list of role/content dicts — every future feature that talks to the
LLM will call `instruct_chat([...])` instead of repeating this logic.

**4.4** Now `modules/chatbots.py` — note both chat functions take
`(message, history)`, the exact signature `gr.ChatInterface` expects:

```python
import gradio as gr
from core.models import get_blenderbot, instruct_chat

def blenderbot_reply(message, history):
    tokenizer, model = get_blenderbot()
    lines = []
    for user_msg, bot_msg in history[-3:]:
        lines.append(f"User: {user_msg}")
        if bot_msg:
            lines.append(f"Bot: {bot_msg}")
    prompt = "\n".join(lines) + f"\nUser: {message}\nBot:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=0.6)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def llm_chat_reply(message, history):
    messages = [{"role": "system", "content": "You are a helpful AI assistant. Give short and concise responses."}]
    for user_msg, bot_msg in history[-5:]:
        messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})
    return instruct_chat(messages, max_new_tokens=80, temperature=0.5)

def build_tab():
    with gr.Tab("💬 Chatbots"):
        with gr.Tab("Simple (Blenderbot)"):
            gr.ChatInterface(fn=blenderbot_reply, type="tuples")
        with gr.Tab("Modern (Instruct LLM)"):
            gr.ChatInterface(fn=llm_chat_reply, type="tuples")
```

**4.5** Add it to `app.py`: `from modules import image_captioning, chatbots`
and `chatbots.build_tab()` under the previous tab.

**Checkpoint:** test both chat tabs. Notice Blenderbot's replies are
short/quirky — that's a small seq2seq model, working as intended, not a
bug. The instruct LLM should give coherent, on-topic answers.

---

## Part 5 — Voice Assistant (adds audio I/O + TTS)

New concepts here: audio input/output components, speech-to-text (STT) via
Whisper, and text-to-speech (TTS).

**5.1** Add to `core/models.py`:

```python
def get_whisper():
    if "whisper" not in _cache:
        from transformers import pipeline
        _cache["whisper"] = pipeline(
            "automatic-speech-recognition", model="openai/whisper-tiny.en", chunk_length_s=30,
        )
    return _cache["whisper"]

def get_tts(model_id="facebook/mms-tts-eng"):
    key = f"tts::{model_id}"
    if key not in _cache:
        from transformers import VitsModel, AutoTokenizer
        model = VitsModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        _cache[key] = (tokenizer, model)
    return _cache[key]
```

Notice `get_tts` takes a parameter with a default — the cache key
includes the model ID, so calling `get_tts("facebook/mms-tts-spa")` later
(for the translator, Part 8) caches a *separate* Spanish voice model
without any code changes here.

**5.2** `modules/voice_assistant.py` — the new pattern is Gradio's
`gr.Audio` component, and returning `(sampling_rate, numpy_array)` for
playable audio output:

```python
import torch
import gradio as gr
from core.models import get_whisper, get_tts, instruct_chat

def transcribe(audio_path):
    if not audio_path:
        return ""
    return get_whisper()(audio_path, batch_size=8)["text"]

def synthesize_speech(text):
    tokenizer, model = get_tts()
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform
    return (model.config.sampling_rate, output.squeeze().cpu().numpy())

def voice_assistant_turn(audio_path, history):
    history = history or []
    user_text = transcribe(audio_path)
    if not user_text.strip():
        return history, None, "Didn't catch that."
    messages = [{"role": "system", "content": "Act like a personal assistant. Keep replies brief."}]
    for u, b in history[-5:]:
        messages += [{"role": "user", "content": u}, {"role": "assistant", "content": b}]
    messages.append({"role": "user", "content": user_text})
    reply_text = instruct_chat(messages, max_new_tokens=120)
    return history + [(user_text, reply_text)], synthesize_speech(reply_text), reply_text

def build_tab():
    with gr.Tab("🎙️ Voice Assistant"):
        chat_state = gr.State([])
        audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath")
        ask_btn = gr.Button("Ask")
        chat_out = gr.Chatbot()
        audio_out = gr.Audio(autoplay=True)
        reply_text_out = gr.Textbox(visible=False)
        ask_btn.click(
            voice_assistant_turn, inputs=[audio_in, chat_state],
            outputs=[chat_state, audio_out, reply_text_out],
        ).then(lambda h: h, inputs=chat_state, outputs=chat_out)
```

Two things worth noticing: (1) `voice_assistant_turn` reuses
`instruct_chat` from Part 4 — zero new LLM code needed; (2) the
`.then(...)` chain updates the visible Chatbot *after* the click handler
finishes, since `chat_state` (a `gr.State`, invisible) and `chat_out` (the
visible `gr.Chatbot`) are kept as two separate components here.

**Checkpoint:** record a short question, click Ask. You should see your
transcribed text, a reply, and hear it spoken back.

---

## Part 6 — Meeting Companion (reuses everything, adds nothing new)

This one's a good exercise: it needs *zero* new entries in
`core/models.py` — just Whisper (already have it) + `instruct_chat`
(already have it), combined differently.

```python
import gradio as gr
from core.models import get_whisper, instruct_chat

def summarize_meeting(audio_file):
    if not audio_file:
        return "", "Please upload an audio file."
    transcript = get_whisper()(audio_file, batch_size=8)["text"]
    messages = [
        {"role": "system", "content": "You summarize meeting transcripts into a concise bulleted list of key points."},
        {"role": "user", "content": f"List the key points with details from the following context:\n{transcript}"},
    ]
    return transcript, instruct_chat(messages, max_new_tokens=400, temperature=0.2)

def build_tab():
    with gr.Tab("📝 Meeting Companion"):
        audio_in = gr.Audio(sources=["upload"], type="filepath")
        run_btn = gr.Button("Transcribe & summarize")
        transcript_out = gr.Textbox(label="Transcript", lines=8)
        summary_out = gr.Textbox(label="Key points", lines=10)
        run_btn.click(summarize_meeting, inputs=audio_in, outputs=[transcript_out, summary_out])
```

**Checkpoint:** upload a short audio clip with speech in it (a voice memo
works). You should get a transcript and a bulleted summary.

---

## Part 7 — PDF Chatbot (RAG) — the new concepts from Part 0 in code

**7.1** Install: `pip install langchain langchain-community langchain-huggingface chromadb pypdf sentence-transformers`

**7.2** Add the embeddings getter to `core/models.py`:

```python
def get_embeddings():
    if "embeddings" not in _cache:
        from langchain_huggingface import HuggingFaceEmbeddings
        _cache["embeddings"] = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _cache["embeddings"]
```

**7.3** `modules/pdf_rag_chatbot.py` — walk through this against the RAG
steps from Part 0 as you read it: load → split → embed+store → retrieve →
generate.

```python
import gradio as gr
from core.models import get_embeddings

_state = {"chain": None}

def process_pdf(pdf_file):
    if pdf_file is None:
        return "Please upload a PDF."
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain.chains import RetrievalQA
    from transformers import pipeline
    from langchain_huggingface import HuggingFacePipeline

    documents = PyPDFLoader(pdf_file.name).load()                       # 1. load
    texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(documents)  # 2. split
    db = Chroma.from_documents(texts, embedding=get_embeddings(), persist_directory="./chroma_db")          # 3. embed + store

    hf_pipe = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct", max_new_tokens=300)
    llm = HuggingFacePipeline(pipeline=hf_pipe)
    _state["chain"] = RetrievalQA.from_chain_type(                       # 4+5. retrieve + generate, wired together
        llm=llm, chain_type="stuff",
        retriever=db.as_retriever(search_type="mmr", search_kwargs={"k": 3}),
        return_source_documents=False, input_key="question",
    )
    return f"Indexed {len(texts)} chunks. Ask away below."

def ask_pdf(question, history):
    if _state["chain"] is None:
        return history + [(question, "Please process a PDF first.")]
    answer = _state["chain"].invoke({"question": question})["result"]
    return history + [(question, answer)]

def build_tab():
    with gr.Tab("📄 PDF Chatbot (RAG)"):
        pdf_in = gr.File(label="PDF file", file_types=[".pdf"])
        process_btn = gr.Button("Process PDF")
        status_out = gr.Textbox(label="Status")
        process_btn.click(process_pdf, inputs=pdf_in, outputs=status_out)
        chatbot = gr.Chatbot()
        question_in = gr.Textbox(label="Your question")
        ask_btn = gr.Button("Ask")
        ask_btn.click(ask_pdf, inputs=[question_in, chatbot], outputs=chatbot)
```

`RetrievalQA.from_chain_type(chain_type="stuff", ...)` is LangChain doing
step 5 for you: "stuff" means it takes the top-3 retrieved chunks and
literally stuffs their text into the LLM's prompt alongside the question.

**Checkpoint:** upload any text-based PDF, click Process, wait for the
"Indexed N chunks" message, then ask a question you know the answer to
from that PDF. The answer should reference its actual content.

---

## Part 8 — Voice Translator (multi-language TTS via the parameterized getter)

This is where `get_tts(model_id)` from Part 5 pays off — no changes to
`core/models.py` needed, just call it with a different model ID per
language.

**8.1** Add a translation getter:

```python
def get_translator():
    if "translator" not in _cache:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        _cache["translator"] = (tokenizer, model)
    return _cache["translator"]
```

**8.2** `modules/voice_translator.py`:

```python
import torch
import gradio as gr
from core.models import get_whisper, get_translator, get_tts

LANGUAGES = {
    "Spanish": ("spa_Latn", "facebook/mms-tts-spa"),
    "French": ("fra_Latn", "facebook/mms-tts-fra"),
}

def translate_text(text, target_lang_label):
    tokenizer, model = get_translator()
    target_code, _ = LANGUAGES[target_lang_label]
    tokenizer.src_lang = "eng_Latn"
    inputs = tokenizer(text, return_tensors="pt")
    forced_id = tokenizer.convert_tokens_to_ids(target_code)
    generated = model.generate(**inputs, forced_bos_token_id=forced_id, max_new_tokens=200)
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

def run_translation(audio_path, typed_text, target_lang_label):
    source_text = get_whisper()(audio_path, batch_size=8)["text"] if audio_path else (typed_text or "")
    if not source_text.strip():
        return "", "", None
    translated = translate_text(source_text, target_lang_label)
    _, tts_model_id = LANGUAGES[target_lang_label]
    tokenizer, model = get_tts(tts_model_id)          # <-- same function, different model_id
    inputs = tokenizer(translated, return_tensors="pt")
    with torch.no_grad():
        waveform = model(**inputs).waveform
    return source_text, translated, (model.config.sampling_rate, waveform.squeeze().cpu().numpy())

def build_tab():
    with gr.Tab("🌐 Voice Translator"):
        audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath")
        text_in = gr.Textbox(label="...or type English text")
        lang_in = gr.Dropdown(choices=list(LANGUAGES.keys()), value="Spanish")
        run_btn = gr.Button("Translate")
        source_out = gr.Textbox(label="Source text")
        translated_out = gr.Textbox(label="Translation")
        audio_out = gr.Audio(autoplay=True)
        run_btn.click(run_translation, inputs=[audio_in, text_in, lang_in],
                      outputs=[source_out, translated_out, audio_out])
```

**Checkpoint:** type "Good morning" and pick Spanish. Expect Spanish text
plus spoken audio.

---

## Part 9 — Career Tools (pure prompt engineering, no new concepts)

The simplest feature to add — three functions, all just `instruct_chat`
with different prompts. This is a deliberate final exercise: by this
point you should be able to predict the shape of this file before reading
it.

```python
import gradio as gr
from core.models import instruct_chat

def polish_resume(position_name, resume_content, polish_prompt=""):
    prompt = (f"Given the resume content: '{resume_content}', polish it based on "
              f"the following instructions: {polish_prompt} for the {position_name} position."
              if polish_prompt.strip() else
              f"Suggest improvements for the following resume content: '{resume_content}' "
              f"to better align with the {position_name} position.")
    return instruct_chat([{"role": "user", "content": prompt}], max_new_tokens=400, temperature=0.7)

def build_tab():
    with gr.Tab("💼 Career Tools"):
        with gr.Tab("Resume Polisher"):
            pos = gr.Textbox(label="Position Name")
            resume = gr.Textbox(label="Resume Content", lines=15)
            instr = gr.Textbox(label="Instructions (optional)")
            btn = gr.Button("Polish")
            out = gr.Textbox(label="Polished Content", lines=15)
            btn.click(polish_resume, inputs=[pos, resume, instr], outputs=out)
        # cover_letter and career_advice tabs follow the exact same shape —
        # write them yourself now: one prompt-building function, one gr.Button.click
```

Writing the cover-letter and career-advice functions yourself here is the
point — if Parts 1-8 landed, this should take you a few minutes with no
new concepts required.

---

## Part 10 — The standalone Flask + JS chatbot

Every feature so far has been Gradio (Python generates the UI for you).
This one is deliberately different — you write the HTTP route yourself,
and a separate JavaScript file calls it. Useful to understand both styles.

**10.1** `pip install flask flask_cors`, and reuse the Blenderbot getter
from Part 4 in a plain Flask app:

```python
from flask import Flask, request, render_template
from flask_cors import CORS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)
CORS(app)
model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = []

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/chatbot", methods=["POST"])
def handle_prompt():
    input_text = request.get_json()["prompt"]
    conversation_history[:] = conversation_history[-6:]
    prompt = "\n".join(conversation_history) + f"\nUser: {input_text}\nBot:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=0.6)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    conversation_history += [f"User: {input_text}", f"Bot: {response}"]
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

**10.2** `templates/index.html` needs an input box + a script tag; and
`static/script.js` does the actual work — `fetch()` a POST request to
`/chatbot`, and drop the response into the page:

```javascript
async function sendMessage() {
  const prompt = document.getElementById("user-input").value;
  const res = await fetch("http://127.0.0.1:5000/chatbot", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  const text = await res.text();
  // append `text` to the page's chat log
}
```

This is the core of any web app talking to a Python backend: JS sends
JSON, Python reads `request.get_json()`, does work, returns a response.

**Checkpoint:** `python app.py` in this folder, open `127.0.0.1:5000`,
type a message. Also try it without the UI at all, straight from a
terminal, to prove the backend works independent of any frontend:
```powershell
curl -X POST -H "Content-Type: application/json" -d "{\"prompt\": \"hello\"}" http://127.0.0.1:5000/chatbot
```

---

## Part 11 — Making it use your GPU

Everything above runs on CPU as written. To use an NVIDIA GPU:

**11.1** Install the CUDA build of torch (before anything else):
```powershell
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

**11.2** In `core/models.py`, detect it once at the top:
```python
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

**11.3** Every getter that loads a model needs `.to(DEVICE)` added after
`from_pretrained(...)`, e.g.:
```python
model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID).to(DEVICE)
```

**11.4** For the instruct LLM specifically, bigger/better models fit on a
GPU if you *quantize* them (compress weights to 4-bit instead of 32-bit —
smaller memory footprint, small quality tradeoff). This needs the
`bitsandbytes` package and a `BitsAndBytesConfig`:
```python
from transformers import BitsAndBytesConfig
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct", quantization_config=quant_config, device_map="auto",
)
```

**Checkpoint:** `python -c "import torch; print(torch.cuda.is_available())"`
should print `True`. Re-run the app and compare response speed to before.

---

## What you've actually built

Step back and look at the shape: **one shared model cache**
(`core/models.py`), **one file per feature** (`modules/*.py`, each
exporting a `build_tab()`), and **one file that assembles them**
(`app.py`). Every feature after the second one was mostly "call an
existing getter, write a new UI function" — that's the payoff of the
registry pattern from Part 3, and it's the same reason real production
codebases are organized this way: new features become additive, not
multiplicative, in complexity.

From here: re-read your own `README.md` (Testing each module) and go
through every tab's checkpoint again — should now be second nature. If
something breaks, you'll be debugging code you actually understand.
