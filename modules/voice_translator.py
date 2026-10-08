"""
Lab 8: Voice Translation Assistant (Babel Fish).

Original lab used watsonx.ai for translation and Watson STT/TTS
(sandbox-only URLs). Replaced with Whisper (STT), NLLB-200 (MT), and
MMS-TTS (TTS) — all local.
"""

import torch
import gradio as gr

from core.models import get_whisper, get_translator, get_tts

# NLLB language codes for a handful of common targets.
# Paired with an MMS-TTS checkpoint for spoken output where available.
LANGUAGES = {
    "Spanish": ("spa_Latn", "facebook/mms-tts-spa"),
    "French": ("fra_Latn", "facebook/mms-tts-fra"),
    "German": ("deu_Latn", "facebook/mms-tts-deu"),
    "Bengali": ("ben_Beng", "facebook/mms-tts-ben"),
    "Hindi": ("hin_Deva", "facebook/mms-tts-hin"),
    "Japanese": ("jpn_Jpan", "facebook/mms-tts-jpn"),
}


def translate_text(text, target_lang_label):
    try:
        from core.models import TRANSLATION_MODEL_ID, DEVICE
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_ID, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_ID, local_files_only=True).to(DEVICE)
        target_code, _ = LANGUAGES[target_lang_label]
        tokenizer.src_lang = "eng_Latn"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_code)
        generated = model.generate(
            **inputs, forced_bos_token_id=forced_bos_token_id, max_new_tokens=200
        )
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    except Exception:
        from core.models import instruct_chat
        prompt = (
            f"Translate the following English sentence to {target_lang_label}. "
            f"Output only the translated text, nothing else:\n{text}"
        )
        return instruct_chat([{"role": "user", "content": prompt}], max_new_tokens=150)


def synthesize(text, target_lang_label):
    _, tts_model_id = LANGUAGES[target_lang_label]
    try:
        from transformers import VitsModel, AutoTokenizer
        from core.models import DEVICE
        model = VitsModel.from_pretrained(tts_model_id, local_files_only=True).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(tts_model_id, local_files_only=True)
    except Exception:
        try:
            tokenizer, model = get_tts()
        except Exception:
            return None
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        waveform = model(**inputs).waveform
    audio = waveform.squeeze().cpu().numpy()
    return (model.config.sampling_rate, audio)


def run_translation(audio_path, typed_text, target_lang_label):
    if audio_path:
        pipe = get_whisper()
        source_text = pipe(audio_path, batch_size=8)["text"]
    else:
        source_text = typed_text or ""

    if not source_text.strip():
        return "", "", None

    translated = translate_text(source_text, target_lang_label)
    spoken = synthesize(translated, target_lang_label)
    return source_text, translated, spoken


def build_tab():
    with gr.Tab("🌐 Voice Translator"):
        gr.Markdown(
            "### Voice Translation Assistant (Babel Fish)\n"
            "Speak or type English, pick a target language, get translated "
            "text and (where a voice model exists) spoken audio."
        )
        with gr.Row():
            audio_in = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speak (optional)")
            text_in = gr.Textbox(label="...or type English text")
        lang_in = gr.Dropdown(choices=list(LANGUAGES.keys()), value="Spanish", label="Target language")
        run_btn = gr.Button("Translate")
        source_out = gr.Textbox(label="Recognized/typed source text")
        translated_out = gr.Textbox(label="Translation")
        audio_out = gr.Audio(label="Spoken translation", autoplay=True)
        run_btn.click(
            run_translation,
            inputs=[audio_in, text_in, lang_in],
            outputs=[source_out, translated_out, audio_out],
        )
