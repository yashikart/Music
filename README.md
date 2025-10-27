
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
