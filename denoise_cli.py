"""Usage: python denoise_cli.py noisy.mp3 clean.wav [--strength 0.8] [--adaptive]"""
import argparse
from pathlib import Path

from audio.noise_filter import reduce_noise
from audio.preprocessor import decode_to_mono, encode_wav

RATE = 16_000

p = argparse.ArgumentParser()
p.add_argument("input")
p.add_argument("output")
p.add_argument("--strength", type=float, default=0.8)
p.add_argument("--adaptive", action="store_true", help="for changing noise (slower)")
a = p.parse_args()

audio = decode_to_mono(Path(a.input), RATE)
cleaned = reduce_noise(audio, RATE, strength=a.strength, stationary=not a.adaptive)
Path(a.output).write_bytes(encode_wav(cleaned, RATE))
print(f"Wrote {a.output}")