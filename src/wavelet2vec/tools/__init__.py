"""Analysis tools built on wavelet2vec's audio helpers."""
from wavelet2vec.tools.vocal_detector import (
    DEFAULT_VOWELS,
    PARTS,
    VOWEL_DESCRIPTIONS,
    VOWEL_FORMANTS,
    VocalTrack,
    detect_vocal_parts,
    formant_envelope_db,
)

__all__ = [
    "DEFAULT_VOWELS", "PARTS", "VOWEL_DESCRIPTIONS", "VOWEL_FORMANTS", "VocalTrack",
    "detect_vocal_parts", "formant_envelope_db",
]
