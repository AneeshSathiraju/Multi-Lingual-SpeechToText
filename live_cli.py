"""Live transcription from the local microphone. Run: python live_cli.py --language hi"""
import argparse
import shutil

import numpy as np

from config import Settings
from realtime.chunker import UtteranceChunker, rms
from realtime.microphone import Microphone
from realtime.stream import LiveTranscriber
from recognition.transcriber import Transcriber


def calibrate(mic, seconds=1.0) -> float:
    """Measure room noise for a second and set the threshold above it."""
    levels = []
    for block in mic.blocks():
        levels.append(rms(block))
        if len(levels) >= int(seconds / 0.1):
            break
    return max(0.005, float(np.median(levels)) * 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--language", default=None, help="en, hi, te ... (omit to auto-detect per utterance)")
    p.add_argument("--threshold", type=float, default=None, help="speech RMS threshold (default: auto-calibrate)")
    p.add_argument("--device", default=None, help="mic device index (see: python -m sounddevice)")
    p.add_argument("--out", default=None, help="append final transcripts to this UTF-8 file")
    args = p.parse_args()

    settings = Settings()
    transcriber = Transcriber(settings.model_size, settings.device, settings.compute_type)
    width = shutil.get_terminal_size().columns - 1

    def show(event):
        if event.type == "partial":
            print("\r" + ("... " + event.text)[-width:].ljust(width), end="", flush=True)
        else:
            print("\r" + f"[{event.language}] {event.text}".ljust(width))
            if args.out:
                with open(args.out, "a", encoding="utf-8") as f:
                    f.write(event.text + "\n")

    with Microphone(device=args.device) as mic:
        if args.threshold is None:
            print("Calibrating: stay quiet for 1 second...")
            threshold = calibrate(mic)
        else:
            threshold = args.threshold
        print(f"Threshold {threshold:.4f}. Speak now. Ctrl+C to stop.")
        session = LiveTranscriber(transcriber, args.language, UtteranceChunker(threshold=threshold))
        try:
            for block in mic.blocks():
                for event in session.feed(block):
                    show(event)
        except KeyboardInterrupt:
            pass
        for event in session.flush():
            show(event)


if __name__ == "__main__":
    main()