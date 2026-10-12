"""Generate a vox-synth hybrid note with a trained MorphNet-L, with vowel automation.

    PYTHONPATH=src python experiments/morph_generate.py --checkpoint experiments/output/morph/v2/checkpoints/best.pt \
        --vocal voice.wav --synth synth.wav --midi 60 --alpha 0.5 --pitch synth --vowels "a@0,o@0.8" --out hybrid.wav

``--vowels``: empty keeps the voice's own syllables and vowels; one vowel ("a") sings it; automation points
"vowel@seconds,..." glide between vowels (i, e, eh, a, o, u). ``--vowel-amount`` 0..1 blends with the voice's
own vowels. Both notes must be at the same key (``--midi``, MIDI 60 = C4); 1.2 s is used.
"""
import argparse

import numpy as np
import soundfile as sf

from wavelet2vec.morph_large import generate_hybrid, load_morph_large


def parse_vowels(text):
    if not text:
        return None
    if "@" not in text:
        return text.strip()
    points = []
    for item in text.split(","):
        vowel, at = item.split("@")
        points.append((float(at), vowel.strip()))
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocal", required=True)
    parser.add_argument("--synth", required=True)
    parser.add_argument("--midi", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.5, help="vocal amount (0.5 = half vocal, half synth)")
    parser.add_argument("--pitch", choices=("synth", "vocal"), default="synth", help="whose pitch the hybrid keeps")
    parser.add_argument("--vowels", default="", help='e.g. "a", or automation "a@0,o@0.8"')
    parser.add_argument("--vowel-amount", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    vocal, rate_v = sf.read(args.vocal, always_2d=True)
    synth, rate_s = sf.read(args.synth, always_2d=True)
    if rate_v != rate_s:
        raise SystemExit("vocal and synth must share a sample rate")
    model = load_morph_large(args.checkpoint)
    audio = generate_hybrid(model, vocal.T, synth.T, args.midi, args.alpha, args.pitch, parse_vowels(args.vowels),
                            args.vowel_amount, sample_rate=rate_v)
    audio *= 0.99 / max(float(np.abs(audio).max()), 0.99)
    sf.write(args.out, audio.T, 192000, subtype="FLOAT")
    print(f"{args.out}: 192 kHz stereo, alpha {args.alpha}, pitch from {args.pitch}, vowels {args.vowels or 'as sung'}")


if __name__ == "__main__":
    main()
