"""
Central place where every heavy model is loaded, ONCE, on first use.

Every module in `modules/` imports from here instead of loading its own
copy of a model. This keeps memory usage sane when many tabs share the
same underlying LLM/Whisper/TTS model.

GPU support: if a CUDA GPU is available, every model is moved to it
automatically. The instruct LLM defaults to a 4-bit quantized
Phi-3-mini-4k-instruct on GPU (~2.5 GB VRAM, leaves plenty of room for
Whisper + BLIP + NLLB + TTS running alongside it on an 8 GB card). On a
CPU-only machine it falls back to the smaller SmolLM2-1.7B-Instruct in
float32 so it still runs comfortably.

Swap model IDs / the quantization choice below to tune quality vs.
memory without touching any module code.
"""

import warnings
import torch

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Model IDs (change these to swap models project-wide) -----------------
BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"
BLENDERBOT_MODEL_ID = "facebook/blenderbot-400M-distill"

# Used only when a CUDA GPU is available (loaded 4-bit quantized).
INSTRUCT_LLM_MODEL_ID_GPU = "microsoft/Phi-3-mini-4k-instruct"
# Used as a CPU fallback — smaller and fast enough without a GPU.
INSTRUCT_LLM_MODEL_ID_CPU = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

WHISPER_MODEL_ID = "openai/whisper-tiny.en"
TTS_MODEL_ID = "facebook/mms-tts-eng"
TRANSLATION_MODEL_ID = "facebook/nllb-200-distilled-600M"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Exposed for modules that need the raw model id (e.g. the RAG chatbot's
# HuggingFacePipeline, which builds its own pipeline rather than using
# instruct_chat()).
INSTRUCT_LLM_MODEL_ID = (
    INSTRUCT_LLM_MODEL_ID_GPU if DEVICE == "cuda" else INSTRUCT_LLM_MODEL_ID_CPU
)

_cache = {}


def get_blip():
    """Image captioning model (BLIP)."""
    if "blip" not in _cache:
        from transformers import AutoProcessor, BlipForConditionalGeneration
        processor = AutoProcessor.from_pretrained(BLIP_MODEL_ID)
        model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID).to(DEVICE)
        _cache["blip"] = (processor, model)
    return _cache["blip"]


def get_blenderbot():
    """Simple sequence-to-sequence chatbot."""
    if "blenderbot" not in _cache:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(BLENDERBOT_MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(BLENDERBOT_MODEL_ID).to(DEVICE)
        _cache["blenderbot"] = (tokenizer, model)
    return _cache["blenderbot"]


def get_instruct_llm():
    """
    Shared instruct-tuned causal LM, used for: modern chatbot, voice
    assistant replies, meeting summarization, and the career tools
    (resume polish / cover letter / career advice).

    On CUDA: Phi-3-mini-4k-instruct, loaded 4-bit (bitsandbytes) — much
    higher quality than the tiny CPU models, small VRAM footprint.
    On CPU: SmolLM2-1.7B-Instruct in float32.
    """
    if "instruct_llm" not in _cache:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if DEVICE == "cuda":
            from transformers import BitsAndBytesConfig

            model_id = INSTRUCT_LLM_MODEL_ID_GPU
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            try:
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=quant_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            except Exception:
                # bitsandbytes not available/working (e.g. plain Windows
                # without WSL) — fall back to fp16, still GPU-resident.
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                )
        else:
            model_id = INSTRUCT_LLM_MODEL_ID_CPU
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="cpu", torch_dtype=torch.float32
            )

        _cache["instruct_llm"] = (tokenizer, model)
    return _cache["instruct_llm"]


def get_whisper():
    """Speech-to-text pipeline, shared by voice assistant, meeting
    companion, and voice translator."""
    if "whisper" not in _cache:
        from transformers import pipeline
        _cache["whisper"] = pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL_ID,
            chunk_length_s=30,
            device=0 if DEVICE == "cuda" else -1,
        )
    return _cache["whisper"]


def get_tts(model_id=None):
    """Text-to-speech (VITS/MMS). Pass a different `model_id` to get a
    voice for another language (e.g. 'facebook/mms-tts-spa' for Spanish)."""
    model_id = model_id or TTS_MODEL_ID
    key = f"tts::{model_id}"
    if key not in _cache:
        from transformers import VitsModel, AutoTokenizer
        model = VitsModel.from_pretrained(model_id).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        _cache[key] = (tokenizer, model)
    return _cache[key]


def get_translator():
    """NLLB-200 machine translation pipeline."""
    if "translator" not in _cache:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_ID).to(DEVICE)
        _cache["translator"] = (tokenizer, model)
    return _cache["translator"]


def get_embeddings():
    """Sentence-transformers embeddings for the RAG / PDF chatbot."""
    if "embeddings" not in _cache:
        from langchain_huggingface import HuggingFaceEmbeddings
        _cache["embeddings"] = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_ID,
            model_kwargs={"device": DEVICE},
        )
    return _cache["embeddings"]


def instruct_chat(messages, max_new_tokens=300, temperature=0.6):
    """
    Convenience helper: run the shared instruct LLM over a list of
    {"role", "content"} chat messages and return the generated text.
    Used by chatbots.py, voice_assistant.py, meeting_companion.py and
    career_tools.py so the prompting logic lives in one place.
    """
    tokenizer, model = get_instruct_llm()
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        max_length=1024,
        truncation=True,
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
        )
    input_length = tokenized["input_ids"].shape[-1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return response.strip()
