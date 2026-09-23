"""Lab 1: Give Meaningful Names to Your Photos with AI (BLIP)."""

import io
import requests
from bs4 import BeautifulSoup
from PIL import Image
import numpy as np
import gradio as gr

from core.models import get_blip


def caption_image(input_image: np.ndarray) -> str:
    if input_image is None:
        return "Please upload an image."
    processor, model = get_blip()
    raw_image = Image.fromarray(input_image).convert("RGB")
    inputs = processor(images=raw_image, text="the image of", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    return processor.decode(outputs[0], skip_special_tokens=True)


def caption_page_images(url: str, max_images: int = 15) -> str:
    """Scrape a web page and caption every image on it (alt-text generator)."""
    if not url:
        return "Please provide a URL."
    processor, model = get_blip()
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=15)
    except Exception as exc:
        return f"Could not fetch URL: {exc}"

    soup = BeautifulSoup(response.text, "html.parser")
    img_elements = soup.find_all("img")[:max_images]

    lines = []
    for img_element in img_elements:
        img_url = img_element.get("src") or img_element.get("data-src")
        if not img_url or img_url.endswith(".svg"):
            continue
        if img_url.startswith("//"):
            img_url = "https:" + img_url
        elif img_url.startswith("/"):
            from urllib.parse import urljoin
            img_url = urljoin(url, img_url)

        try:
            r = requests.get(img_url, timeout=10, headers=headers)
            raw_image = Image.open(io.BytesIO(r.content))
            if raw_image.size[0] * raw_image.size[1] < 200:
                continue
            raw_image = raw_image.convert("RGB")
            inputs = processor(images=raw_image, text="the image of", return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=50)
            caption = processor.decode(out[0], skip_special_tokens=True)
            lines.append(f"{img_url} : {caption}")
        except Exception:
            continue

    if not lines:
        return "No captionable images found on that page."
    return "\n".join(lines)


def build_tab():
    with gr.Tab("🖼️ Image Captioning"):
        gr.Markdown(
            "### Image Captioning & Alt-Text\n"
            "Caption a single image, or scrape a whole web page and "
            "caption every image on it (batch alt-text generation)."
        )
        with gr.Tab("Single image"):
            img_in = gr.Image(label="Upload an image")
            caption_btn = gr.Button("Generate caption")
            caption_out = gr.Textbox(label="Caption")
            caption_btn.click(caption_image, inputs=img_in, outputs=caption_out)

        with gr.Tab("Batch: caption a web page's images"):
            url_in = gr.Textbox(
                label="Page URL", placeholder="https://en.wikipedia.org/wiki/IBM"
            )
            scrape_btn = gr.Button("Scrape & caption all images")
            scrape_out = gr.Textbox(label="url : caption", lines=15)
            scrape_btn.click(caption_page_images, inputs=url_in, outputs=scrape_out)
