import os
import torch
import scipy.io.wavfile as wavfile
import sys
sys.path.insert(0, ".")
from core.models import get_tts

text = "Welcome to the quarterly engineering sync. Today we are reviewing the launch of GenForge Studio, our offline generative AI platform."
print("Synthesizing sample audio...")
tokenizer, model = get_tts()
inputs = tokenizer(text, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model(**inputs).waveform
audio = output.squeeze().cpu().numpy()

wav_path = os.path.join(os.path.dirname(__file__), "sample_meeting.wav")
wavfile.write(wav_path, model.config.sampling_rate, audio)
print(f"Generated sample audio at: {wav_path}")
