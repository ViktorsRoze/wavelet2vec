from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from wavelet2vec.adsr import ADSR
from wavelet2vec.audio_io import list_audio_files, load_audio, save_audio
from wavelet2vec.character import (
    SoundCharacter,
    apply_character,
    character_from_files,
    character_from_folder,
)
from wavelet2vec.embedder import (
    Wavelet2Vec,
    Wavelet2VecConfig,
    cosine_similarity,
    pairwise_similarity,
)
from wavelet2vec.perform import perform
from wavelet2vec.sequence import MODULATION_TARGETS, AutomationCurve, render_sequence
from wavelet2vec.slicing import slice_cues, slice_grid, slice_markers, slice_onsets
from wavelet2vec.style_transfer import blend, style_transfer


def _load_character(target: str) -> SoundCharacter:
    """A character from a saved .npz, a folder of sounds, or a single file."""
    path = Path(target)
    if path.suffix.lower() == ".npz":
        return SoundCharacter.load(path)
    if path.is_dir():
        return character_from_folder(path)
    return character_from_files([path])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wavelet2vec",
        description="Deterministic, phase-aware musical embeddings and embedding-guided morphing.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    embed = commands.add_parser("embed", help="Embed a file or folder of snippets.")
    embed.add_argument("--input", required=True, help="Audio file or folder of snippets.")
    embed.add_argument("--output", default="embeddings.npz", help="Output .npz path.")
    embed.add_argument("--pairwise", help="Optional CSV path for the cosine similarity matrix.")
    embed.add_argument("--query", help="Optional audio file: print snippets most similar to it.")
    embed.add_argument("--top-k", type=int, default=5, help="Number of query matches to show.")
    embed.add_argument("--analysis-sample-rate", type=int, default=22050)
    embed.add_argument("--n-bands", type=int, default=64)
    embed.add_argument("--no-conv", action="store_true", help="Disable the convolutional section.")

    transfer = commands.add_parser(
        "transfer", help="Make a source sound more similar to a target by a given amount."
    )
    transfer.add_argument("--source", required=True, help="Sound to transform.")
    transfer.add_argument(
        "--target",
        required=True,
        help="Sound whose character to borrow: a file, a folder (group character), or a saved .npz character.",
    )
    transfer.add_argument("--amount", type=float, required=True, help="0 = source, 1 = full match.")
    transfer.add_argument("--output", required=True, help="Output audio path.")
    transfer.add_argument(
        "--adsr",
        nargs=4,
        type=float,
        metavar=("A", "D", "S", "R"),
        help="Shape the morph amount over time with an ADSR (seconds, seconds, level, seconds).",
    )
    transfer.add_argument("--no-spectrum", action="store_true", help="Skip spectral envelope transfer.")
    transfer.add_argument("--no-envelope", action="store_true", help="Skip temporal envelope transfer.")
    transfer.add_argument("--max-gain-db", type=float, default=24.0)

    character = commands.add_parser(
        "character", help="Extract the average character of a group of sounds to a .npz file."
    )
    character.add_argument(
        "--inputs", nargs="+", required=True, help="Audio files and/or folders to average."
    )
    character.add_argument("--output", required=True, help="Output .npz character path.")

    sequence = commands.add_parser(
        "sequence",
        help="Render a sequence of snippets with automation-driven character development.",
    )
    sequence.add_argument("--inputs", nargs="+", required=True, help="Snippets, in playback order.")
    sequence.add_argument("--output", required=True, help="Output audio path.")
    sequence.add_argument(
        "--character",
        help="Character to develop toward: a file, a folder, or a saved .npz character.",
    )
    sequence.add_argument(
        "--adsr",
        nargs=4,
        type=float,
        metavar=("A", "D", "S", "R"),
        help="Character ADSR per snippet (attack/decay/release seconds, sustain level).",
    )
    sequence.add_argument(
        "--volume-adsr",
        nargs=4,
        type=float,
        metavar=("A", "D", "S", "R"),
        help="Volume ADSR applied to every snippet.",
    )
    sequence.add_argument(
        "--automation",
        help="Automation clip controlling the sequence: .csv, .json, .flp, or .fst.",
    )
    sequence.add_argument(
        "--modulate",
        choices=MODULATION_TARGETS,
        default="amount",
        help="Which morph parameter the automation drives (default: amount).",
    )
    sequence.add_argument("--max-amount", type=float, default=1.0)
    sequence.add_argument("--gap", type=float, default=0.0, help="Silence between snippets, seconds.")
    sequence.add_argument("--max-gain-db", type=float, default=24.0)

    blend_cmd = commands.add_parser(
        "blend", help="Create a new sound similar to several references by given amounts."
    )
    blend_cmd.add_argument("--inputs", nargs="+", required=True, help="Two or more reference sounds.")
    blend_cmd.add_argument(
        "--weights", nargs="+", type=float, required=True, help="One weight per reference."
    )
    blend_cmd.add_argument("--output", required=True, help="Output audio path.")
    blend_cmd.add_argument(
        "--carrier",
        type=int,
        help="Index of the reference providing the fine structure (default: largest weight).",
    )
    blend_cmd.add_argument("--no-spectrum", action="store_true", help="Skip spectral envelope transfer.")
    blend_cmd.add_argument("--no-envelope", action="store_true", help="Skip temporal envelope transfer.")
    blend_cmd.add_argument("--max-gain-db", type=float, default=24.0)

    perform_cmd = commands.add_parser(
        "perform",
        help="Transform one long DAW render note by note; output stays sample-aligned.",
    )
    perform_cmd.add_argument("--input", required=True, help="The rendered performance (one long file).")
    perform_cmd.add_argument("--output", required=True, help="Output audio path (same length as input).")
    perform_cmd.add_argument(
        "--character",
        help="Character to morph toward: a file, a folder, or a saved .npz character.",
    )
    perform_cmd.add_argument(
        "--slice",
        choices=("grid", "cues", "markers", "onsets"),
        default="grid",
        dest="slice_mode",
        help="How note starts are found (default: grid).",
    )
    perform_cmd.add_argument("--bpm", type=float, help="Grid slicing: project tempo.")
    perform_cmd.add_argument(
        "--division", type=int, default=8, help="Grid slicing: 8 = 8th notes, 16 = 16ths."
    )
    perform_cmd.add_argument("--offset-beats", type=float, default=0.0)
    perform_cmd.add_argument(
        "--spacing", type=float, help="Grid slicing: literal seconds between note starts."
    )
    perform_cmd.add_argument("--markers", help="Marker slicing: CSV/text file of times in seconds.")
    perform_cmd.add_argument("--sensitivity", type=float, default=0.5, help="Onset slicing: 0..1.")
    perform_cmd.add_argument(
        "--automation",
        help="Automation clip over the render's timeline: .csv, .json, .flp, or .fst.",
    )
    perform_cmd.add_argument("--modulate", choices=MODULATION_TARGETS, default="amount")
    perform_cmd.add_argument(
        "--adsr",
        nargs=4,
        type=float,
        metavar=("A", "D", "S", "R"),
        help="Character ADSR per note.",
    )
    perform_cmd.add_argument(
        "--volume-adsr",
        nargs=4,
        type=float,
        metavar=("A", "D", "S", "R"),
        help="Volume ADSR applied to every note.",
    )
    perform_cmd.add_argument("--max-amount", type=float, default=1.0)
    perform_cmd.add_argument("--boundary-fade", type=float, default=0.005)
    perform_cmd.add_argument("--max-gain-db", type=float, default=24.0)
    return parser


def _run_embed(args: argparse.Namespace) -> None:
    config = Wavelet2VecConfig(
        analysis_sample_rate=args.analysis_sample_rate,
        n_bands=args.n_bands,
        include_conv=not args.no_conv,
    )
    embedder = Wavelet2Vec(config)

    input_path = Path(args.input)
    if input_path.is_file():
        embeddings = {input_path.name: embedder.embed_file(input_path)}
    else:
        embeddings = embedder.embed_folder(input_path)
    if not embeddings:
        raise FileNotFoundError(f"No audio files found under {input_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **embeddings)
    print(f"saved {len(embeddings)} embeddings (dim={embedder.dim}) to {output_path}")

    if args.pairwise:
        names, matrix = pairwise_similarity(embeddings)
        pairwise_path = Path(args.pairwise)
        pairwise_path.parent.mkdir(parents=True, exist_ok=True)
        header = "," + ",".join(names)
        rows = [
            f"{name}," + ",".join(f"{value:.4f}" for value in matrix[index])
            for index, name in enumerate(names)
        ]
        pairwise_path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
        print(f"saved pairwise similarity matrix to {pairwise_path}")

    if args.query:
        query = embedder.embed_file(args.query)
        ranked = sorted(
            ((cosine_similarity(query, vector), name) for name, vector in embeddings.items()),
            reverse=True,
        )
        print(f"most similar to {args.query}:")
        for similarity, name in ranked[: args.top_k]:
            print(f"  {similarity:.4f}  {name}")


def _run_transfer(args: argparse.Namespace) -> None:
    source, source_sr = load_audio(args.source)
    character = _load_character(args.target)
    if args.adsr:
        from wavelet2vec.adsr import apply_adsr_character

        morphed = apply_adsr_character(
            source,
            source_sr,
            character,
            ADSR(*args.adsr),
            max_amount=args.amount,
            transfer_spectrum=not args.no_spectrum,
            transfer_envelope=not args.no_envelope,
            max_gain_db=args.max_gain_db,
        )
    else:
        morphed = apply_character(
            source,
            source_sr,
            character,
            amount=args.amount,
            transfer_spectrum=not args.no_spectrum,
            transfer_envelope=not args.no_envelope,
            max_gain_db=args.max_gain_db,
        )
    save_audio(args.output, morphed, source_sr)
    print(f"saved transfer (amount={args.amount}) to {args.output}")


def _run_character(args: argparse.Namespace) -> None:
    paths: list[Path] = []
    for item in args.inputs:
        item_path = Path(item)
        paths.extend(list_audio_files(item_path) if item_path.is_dir() else [item_path])
    if not paths:
        raise FileNotFoundError("No audio files found in the given inputs.")
    character = character_from_files(paths)
    character.save(args.output)
    print(f"saved character of {len(paths)} sounds to {args.output}")


def _run_sequence(args: argparse.Namespace) -> None:
    snippets, rates = [], []
    for path in args.inputs:
        waveform, sample_rate = load_audio(path)
        snippets.append(waveform)
        rates.append(sample_rate)
    character = _load_character(args.character) if args.character else None
    automation = AutomationCurve.from_file(args.automation) if args.automation else None
    result, output_sr = render_sequence(
        snippets,
        rates,
        character=character,
        character_adsr=ADSR(*args.adsr) if args.adsr else None,
        volume_adsr=ADSR(*args.volume_adsr) if args.volume_adsr else None,
        automation=automation,
        modulate=args.modulate,
        max_amount=args.max_amount,
        gap_seconds=args.gap,
        max_gain_db=args.max_gain_db,
    )
    save_audio(args.output, result, output_sr)
    print(f"saved sequence of {len(snippets)} snippets to {args.output}")


def _run_blend(args: argparse.Namespace) -> None:
    if len(args.inputs) != len(args.weights):
        raise ValueError("--inputs and --weights must have the same length.")
    sounds, rates = [], []
    for path in args.inputs:
        waveform, sample_rate = load_audio(path)
        sounds.append(waveform)
        rates.append(sample_rate)
    result, output_sr = blend(
        sounds,
        rates,
        args.weights,
        carrier=args.carrier,
        transfer_spectrum=not args.no_spectrum,
        transfer_envelope=not args.no_envelope,
        max_gain_db=args.max_gain_db,
    )
    save_audio(args.output, result, output_sr)
    weights = ", ".join(f"{weight:g}" for weight in args.weights)
    print(f"saved blend (weights=[{weights}]) to {args.output}")


def _run_perform(args: argparse.Namespace) -> None:
    waveform, sample_rate = load_audio(args.input)
    n_samples = waveform.shape[1]
    if args.slice_mode == "grid":
        if args.bpm is None and args.spacing is None:
            raise ValueError("Grid slicing needs --bpm (with --division) or --spacing.")
        starts = slice_grid(
            n_samples,
            sample_rate,
            bpm=args.bpm,
            division=args.division,
            offset_beats=args.offset_beats,
            spacing_seconds=args.spacing,
        )
    elif args.slice_mode == "cues":
        starts = slice_cues(args.input, n_samples)
    elif args.slice_mode == "markers":
        if not args.markers:
            raise ValueError("Marker slicing needs --markers FILE.")
        starts = slice_markers(args.markers, sample_rate, n_samples)
    else:
        starts = slice_onsets(waveform, sample_rate, sensitivity=args.sensitivity)

    result = perform(
        waveform,
        sample_rate,
        _load_character(args.character) if args.character else None,
        starts,
        automation=AutomationCurve.from_file(args.automation) if args.automation else None,
        character_adsr=ADSR(*args.adsr) if args.adsr else None,
        volume_adsr=ADSR(*args.volume_adsr) if args.volume_adsr else None,
        modulate=args.modulate,
        max_amount=args.max_amount,
        boundary_fade_seconds=args.boundary_fade,
        max_gain_db=args.max_gain_db,
    )
    save_audio(args.output, result, sample_rate)
    print(
        f"performed {len(starts)} notes at {sample_rate} Hz "
        f"({n_samples} samples, length preserved) -> {args.output}"
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "embed":
        _run_embed(args)
    elif args.command == "transfer":
        _run_transfer(args)
    elif args.command == "blend":
        _run_blend(args)
    elif args.command == "character":
        _run_character(args)
    elif args.command == "sequence":
        _run_sequence(args)
    elif args.command == "perform":
        _run_perform(args)


if __name__ == "__main__":
    main()
