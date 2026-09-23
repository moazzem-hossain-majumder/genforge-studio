"""
Lab 4: Flask-based chatbot web app (Flask backend + vanilla JS frontend).

Kept as a standalone mini-app (rather than folded into the Gradio suite)
because the point of this lab is specifically Flask route design +
static/template file structure. Uses the same local Blenderbot model as
Lab 3 — no cloud dependency, matching the original lab exactly.
"""

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
    data = request.get_json()
    input_text = data["prompt"]

    conversation_history[:] = conversation_history[-6:]
    history = "\n".join(conversation_history)
    prompt = history + f"\nUser: {input_text}\nBot:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
        do_sample=True,
        temperature=0.6,
        top_p=0.85,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    conversation_history.append(f"User: {input_text}")
    conversation_history.append(f"Bot: {response}")

    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
