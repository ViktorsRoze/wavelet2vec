from wavelet2vec.adsr import ADSR, apply_adsr_character, apply_adsr_volume
from wavelet2vec.character import (
    SoundCharacter,
    apply_character,
    average_characters,
    character_from_files,
    character_from_folder,
    extract_character,
)
from wavelet2vec.conv_encoder import WaveletConvEncoder
from wavelet2vec.embedder import (
    Wavelet2Vec,
    Wavelet2VecConfig,
    cosine_similarity,
    pairwise_similarity,
)
from wavelet2vec.perform import perform
from wavelet2vec.sequence import AutomationCurve, render_sequence
from wavelet2vec.slicing import (
    read_cue_points,
    slice_cues,
    slice_grid,
    slice_markers,
    slice_onsets,
)
from wavelet2vec.style_transfer import blend, style_transfer

__version__ = "0.4.0"

__all__ = [
    "ADSR",
    "AutomationCurve",
    "SoundCharacter",
    "Wavelet2Vec",
    "Wavelet2VecConfig",
    "WaveletConvEncoder",
    "apply_adsr_character",
    "apply_adsr_volume",
    "apply_character",
    "average_characters",
    "blend",
    "character_from_files",
    "character_from_folder",
    "cosine_similarity",
    "extract_character",
    "pairwise_similarity",
    "perform",
    "read_cue_points",
    "render_sequence",
    "slice_cues",
    "slice_grid",
    "slice_markers",
    "slice_onsets",
    "style_transfer",
]
