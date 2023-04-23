from whisper_jax import FlaxWhisperPipline
import time

# instantiate pipeline
pipeline = FlaxWhisperPipline("openai/whisper-small")

# JIT compile the forward call - slow, but we only do once
s = time.time()
text = pipeline("audio.mp3")

# used cached function thereafter - super fast!!
text = pipeline("audio.mp3")
e = time.time()
print(text)
print("time elapsed: ", e - s)