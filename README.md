
#!pip install transformers datasets accelerate torch scipy

from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile as wav

# Load pretrained model
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

# Enter your text prompt
prompt = "happy piano melody"

# Generate music
inputs = processor(text=[prompt], padding=True, return_tensors="pt")
audio_values = model.generate(**inputs, max_new_tokens=256)

# Save output as WAV
wav.write("generated_music.wav", rate=model.config.audio_encoder.sampling_rate,
          data=audio_values[0, 0].detach().numpy())

print("Music generated and saved as 'generated_music.wav'")



sir preactical's 


#Text-Music (=15-2es)
import torch, torchaudio
from transformers import AutoProcessor, MusicgenForConditionalGeneration
MODEL = "facebook/musicgen-small"
processor = AutoProcessor.from_pretrained(MODEL)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL)
# prompt = "warm lo-fi beat with soft piano and vinyl crackle"
prompt = "An Indian classical music performance with sitar and tabla, gentl
inputs = processor(text = [prompt], return_tensors = "pt")
model.generation_config.do_sample = True
model.generation_config.guidance_scale = 3.0 # creativity vs. prompt adhere
model.generation_config.max_new_tokens = 50 * 18 # ~50 tokens/sec -> ~18s
audio = model.generate(**inputs) # (1, channels samples)
sr = model.config.audio_encoder.sampling_rate
torchaudio.save("text_music.wav", audio[0].cpu(), sr)
print("Saved text_music.wav")
