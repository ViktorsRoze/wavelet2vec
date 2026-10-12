from wavelet2vec.adsr import ADSR, apply_adsr_character, apply_adsr_volume
from wavelet2vec.augment import (
    crop_bars,
    effect_chain,
    invert_polarity as invert_polarity_augment,
    mono_mix,
    random_augment,
    swap_channels,
    time_stretch,
)
from wavelet2vec.character import (
    SoundCharacter,
    apply_character,
    average_characters,
    character_from_files,
    character_from_folder,
    extract_character,
)
from wavelet2vec.brightness_model import (
    BrightnessNet,
    load_brightness_model,
    save_brightness_model,
)
from wavelet2vec.conv_encoder import WaveletConvEncoder
from wavelet2vec.data import (
    ContrastiveViewsDataset,
    LadderAxisDataset,
    SceneDetectionDataset,
    detection_collate,
)
from wavelet2vec.datasets import (
    attack_ladder,
    axis_labels,
    filter_ladder,
    lowpass_pair,
    noise_mix_ladder,
    render_ladder_dataset,
    sine_distortion_ladder,
)
from wavelet2vec.decompose import hpss, shape_transients, transient_split
from wavelet2vec.embedder import (
    EMBEDDING_VERSION,
    Wavelet2Vec,
    Wavelet2VecConfig,
    cosine_similarity,
    pairwise_similarity,
)
from wavelet2vec.eval import (
    EmbeddingStandardizer,
    knn_purity,
    recall_at_k,
    retrieval_report,
)
from wavelet2vec.expression import AXIS_RANGES, render_expression
from wavelet2vec.frontend import WaveletFrontend
from wavelet2vec.imagine import (
    imagine_highs,
    imagine_tonality,
    imagine_transient,
    pitch_shift,
)
from wavelet2vec.index import SimilarityIndex
from wavelet2vec.frame import (
    FrameSpec,
    analyze,
    load_coefficients as load_frame_coefficients,
    reconstruction_stats,
    save_coefficients as save_frame_coefficients,
    synthesize,
)
from wavelet2vec.harmony import (
    F0Track,
    KeyEstimate,
    NoteEvent,
    chromagram,
    estimate_key,
    segment_notes,
    track_chords,
    track_f0,
)
from wavelet2vec.perform import perform
from wavelet2vec.torch_frame import (
    analyze_batch,
    analyze_torch,
    synthesize_torch,
)
from wavelet2vec.rhythm import (
    RHYTHM_DIM,
    RHYTHM_SECTIONS,
    RhythmProfile,
    analyze_rhythm,
    apply_groove,
    estimate_tempo,
    groove_offsets,
    profile_embedding,
    rhythm_embedding,
    swing_offsets,
)
from wavelet2vec.spectral_edit import (
    attenuate_region,
    isolate_regions,
    phase_rotate_regions,
    phase_scramble_regions,
    process_regions,
    region_mask,
)
from wavelet2vec.structure import (
    LoopPoints,
    StructureSegments,
    find_loop_points,
    render_loop,
    segment_structure,
)
from wavelet2vec.spatial import (
    SpatialCharacter,
    StereoPhaseSpectrum,
    apply_spatial_character,
    design_width,
    extract_spatial_character,
    spatial_matrix,
    stereo_phase_spectrum,
)
from wavelet2vec.sequence import AutomationCurve, render_sequence
from wavelet2vec.slicing import (
    read_cue_points,
    slice_cues,
    slice_grid,
    slice_markers,
    slice_onsets,
)
from wavelet2vec.style_transfer import blend, style_transfer
from wavelet2vec.transient_model import (
    TransientNet,
    load_transient_model,
    save_transient_model,
)
from wavelet2vec.tonality_model import (
    TonalityNet,
    load_tonality_model,
    save_tonality_model,
)
from wavelet2vec.vision import (
    CocoBuilder,
    Event,
    ImageMeta,
    bbox_to_event,
    coeff_image,
    event_to_bbox,
    export_dataset,
    image_to_uint8,
    sequence_tokens,
)
from wavelet2vec.scenes import (
    midi_to_hz,
    random_scene,
    render_drum,
    render_note,
    render_scene,
    weak_labels,
)

from wavelet2vec.hybrid import (
    CompactBand,
    HYBRID_VERSION,
    HybridAudioFrontend,
    HybridConfig,
    HybridEncoding,
    HybridLatents,
    HybridTokenEncoder,
)

from wavelet2vec.sliding import (
    HybridSequence,
    SlidingHybridEncoder,
    SlidingTokenBatch,
    SlidingWindowConfig,
)

from wavelet2vec.multiresolution import MultiResolutionLatents, MultiResolutionTokenEncoder
from wavelet2vec.stereo_adapter import (
    STEREO_FEATURE_NAMES, StereoFeatureConfig, StereoFeatures, StereoLatents,
    StereoTokenEncoder, hybrid_stereo_features,
)
from wavelet2vec.training import SourceExample, SourceBatch, collate_sources, separation_loss
from wavelet2vec.feature_cache import FeatureCache, FeatureCacheKey, model_fingerprint, source_fingerprint

from wavelet2vec.context import MusicContextConfig, MusicContextEncoder, MusicContextOutput
from wavelet2vec.self_supervised import (
    AudioBatch, collate_audio, MaskedPredictionConfig, sample_span_mask,
    StudentPrediction, MaskedMusicOutput, MaskedMusicModel,
)
from wavelet2vec.pretraining import PretrainingConfig, LengthBucketSampler, MusicPretrainer

from wavelet2vec.target_statistics import FixedTargetStatistics

from wavelet2vec.corpus import AudioRecord, AudioManifest, AudioCropConfig, AudioWindow, AudioWindowDataset, collate_windows
from wavelet2vec.corpus_training import (
    TrainingTargetMean, fit_training_target_mean, visible_neighbor_prediction,
    evaluate_masked_corpus, CorpusRunConfig, CorpusPretrainingRun,
)

from wavelet2vec.synthesis import (
    SynthesisConfig, SourceSynthesis, HybridSourceDecoder, HybridSeparator,
    project_source_coefficients, separate_audio,
)

from wavelet2vec.multitrack import (
    MultitrackRecord, MultitrackManifest, MultitrackWindow, MultitrackDataset, collate_multitracks,
)
from wavelet2vec.separation_training import SeparationRunConfig, SeparationTrainingRun, evaluate_separation

from wavelet2vec.rendering import (
    SourceRenderConfig, SeparationCancelled, FileSeparationResult, iter_separate_reader, separate_file,
)

from wavelet2vec.inference_bundle import (
    SeparatorBundle, save_separator_bundle, load_separator_bundle, inspect_separator_bundle,
)

__version__ = "0.35.0"

__all__ = [
    "SeparatorBundle", "save_separator_bundle", "load_separator_bundle", "inspect_separator_bundle",
    "SourceRenderConfig", "SeparationCancelled", "FileSeparationResult", "iter_separate_reader", "separate_file",
    "MultitrackRecord", "MultitrackManifest", "MultitrackWindow", "MultitrackDataset", "collate_multitracks",
    "SeparationRunConfig", "SeparationTrainingRun", "evaluate_separation",
    "SynthesisConfig", "SourceSynthesis", "HybridSourceDecoder", "HybridSeparator",
    "project_source_coefficients", "separate_audio",
    "AudioRecord", "AudioManifest", "AudioCropConfig", "AudioWindow", "AudioWindowDataset", "collate_windows",
    "TrainingTargetMean", "fit_training_target_mean", "visible_neighbor_prediction", "evaluate_masked_corpus",
    "CorpusRunConfig", "CorpusPretrainingRun",
    "FixedTargetStatistics",
    "MusicContextConfig", "MusicContextEncoder", "MusicContextOutput",
    "AudioBatch", "collate_audio", "MaskedPredictionConfig", "sample_span_mask",
    "StudentPrediction", "MaskedMusicOutput", "MaskedMusicModel",
    "PretrainingConfig", "LengthBucketSampler", "MusicPretrainer",
    "SourceExample",
    "SourceBatch",
    "collate_sources",
    "separation_loss",
    "FeatureCache",
    "FeatureCacheKey",
    "model_fingerprint",
    "source_fingerprint",
    "MultiResolutionLatents",
    "MultiResolutionTokenEncoder",
    "STEREO_FEATURE_NAMES",
    "StereoFeatureConfig",
    "StereoFeatures",
    "StereoLatents",
    "StereoTokenEncoder",
    "hybrid_stereo_features",
    "HybridSequence",
    "SlidingHybridEncoder",
    "SlidingTokenBatch",
    "SlidingWindowConfig",
    "CompactBand",
    "HYBRID_VERSION",
    "HybridAudioFrontend",
    "HybridConfig",
    "HybridEncoding",
    "HybridLatents",
    "HybridTokenEncoder",
    "ADSR",
    "AXIS_RANGES",
    "AutomationCurve",
    "BrightnessNet",
    "load_brightness_model",
    "save_brightness_model",
    "TonalityNet",
    "load_tonality_model",
    "save_tonality_model",
    "TransientNet",
    "imagine_transient",
    "load_transient_model",
    "save_transient_model",
    "CocoBuilder",
    "ContrastiveViewsDataset",
    "EMBEDDING_VERSION",
    "EmbeddingStandardizer",
    "knn_purity",
    "recall_at_k",
    "retrieval_report",
    "LadderAxisDataset",
    "SceneDetectionDataset",
    "detection_collate",
    "Event",
    "F0Track",
    "FrameSpec",
    "ImageMeta",
    "KeyEstimate",
    "LoopPoints",
    "NoteEvent",
    "StructureSegments",
    "RHYTHM_DIM",
    "RHYTHM_SECTIONS",
    "RhythmProfile",
    "SimilarityIndex",
    "SoundCharacter",
    "SpatialCharacter",
    "Wavelet2Vec",
    "WaveletFrontend",
    "Wavelet2VecConfig",
    "WaveletConvEncoder",
    "analyze",
    "analyze_batch",
    "analyze_rhythm",
    "analyze_torch",
    "apply_groove",
    "attenuate_region",
    "estimate_tempo",
    "groove_offsets",
    "isolate_regions",
    "phase_rotate_regions",
    "phase_scramble_regions",
    "pitch_shift",
    "process_regions",
    "region_mask",
    "swing_offsets",
    "hpss",
    "load_frame_coefficients",
    "reconstruction_stats",
    "save_frame_coefficients",
    "shape_transients",
    "spatial_matrix",
    "stereo_phase_spectrum",
    "StereoPhaseSpectrum",
    "synthesize",
    "synthesize_torch",
    "transient_split",
    "apply_adsr_character",
    "apply_adsr_volume",
    "apply_character",
    "apply_spatial_character",
    "attack_ladder",
    "average_characters",
    "axis_labels",
    "bbox_to_event",
    "blend",
    "coeff_image",
    "crop_bars",
    "effect_chain",
    "design_width",
    "event_to_bbox",
    "export_dataset",
    "extract_spatial_character",
    "filter_ladder",
    "image_to_uint8",
    "imagine_highs",
    "imagine_tonality",
    "lowpass_pair",
    "midi_to_hz",
    "mono_mix",
    "noise_mix_ladder",
    "random_augment",
    "random_scene",
    "render_drum",
    "render_ladder_dataset",
    "render_note",
    "render_scene",
    "sine_distortion_ladder",
    "weak_labels",
    "character_from_files",
    "character_from_folder",
    "chromagram",
    "cosine_similarity",
    "estimate_key",
    "extract_character",
    "find_loop_points",
    "segment_structure",
    "pairwise_similarity",
    "perform",
    "profile_embedding",
    "read_cue_points",
    "render_expression",
    "render_loop",
    "rhythm_embedding",
    "render_sequence",
    "segment_notes",
    "sequence_tokens",
    "track_chords",
    "track_f0",
    "slice_cues",
    "slice_grid",
    "slice_markers",
    "slice_onsets",
    "swap_channels",
    "style_transfer",
    "time_stretch",
    "invert_polarity_augment",
]
