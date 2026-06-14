from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from wavelet2vec.adsr import ADSR, apply_adsr_character, apply_adsr_volume
from wavelet2vec.audio_io import resample
from wavelet2vec.character import SoundCharacter, _as_channels, apply_character

MODULATION_TARGETS = ("amount", "attack", "decay", "sustain", "release")


@dataclass(frozen=True)
class ControlPoint:
    time_seconds: float
    value: float


class AutomationCurve:
    """A control curve sampled over time, e.g. an FL Studio automation clip.

    Values are expected in [0, 1]. Load from CSV (``time_seconds,value`` or
    ``beat,value`` columns), JSON (``[{"time_seconds": ..., "value": ...}]``
    or ``{"tempo_bpm": ..., "points": [...]}``), or an FL Studio project
    (``.flp`` / ``.fst``, requires the optional ``pyflp`` package).
    """

    def __init__(self, points: Sequence[ControlPoint] | None = None, default_value: float = 1.0) -> None:
        self.points = sorted(points or [], key=lambda item: item.time_seconds)
        self.default_value = float(default_value)

    @property
    def duration(self) -> float:
        return self.points[-1].time_seconds if self.points else 0.0

    def value_at(self, time_seconds: float) -> float:
        if not self.points:
            return self.default_value
        if time_seconds <= self.points[0].time_seconds:
            return self.points[0].value
        if time_seconds >= self.points[-1].time_seconds:
            return self.points[-1].value
        for index in range(1, len(self.points)):
            left, right = self.points[index - 1], self.points[index]
            if time_seconds <= right.time_seconds:
                span = max(right.time_seconds - left.time_seconds, 1e-8)
                fraction = (time_seconds - left.time_seconds) / span
                return left.value + (right.value - left.value) * fraction
        return self.default_value

    def value_at_fraction(self, fraction: float) -> float:
        """Samples the curve at a normalized position in [0, 1].

        This maps the curve onto any timeline regardless of units, so an
        automation clip drawn over 4 bars can drive a sequence of any length.
        """
        return self.value_at(float(np.clip(fraction, 0.0, 1.0)) * self.duration)

    @classmethod
    def from_points(cls, pairs: Sequence[tuple[float, float]], default_value: float = 1.0) -> "AutomationCurve":
        return cls([ControlPoint(float(t), float(v)) for t, v in pairs], default_value)

    @classmethod
    def from_file(cls, path: str | Path, default_value: float = 1.0, default_bpm: float = 120.0) -> "AutomationCurve":
        curve_path = Path(path)
        suffix = curve_path.suffix.lower()
        if suffix == ".csv":
            return cls._from_csv(curve_path, default_value, default_bpm)
        if suffix == ".json":
            return cls._from_json(curve_path, default_value, default_bpm)
        if suffix in {".flp", ".fst"}:
            return cls._from_fl_studio(curve_path, default_value, default_bpm)
        raise ValueError(
            f"Unsupported automation format: {curve_path.suffix}. "
            "Supported inputs are .csv, .json, .flp, and .fst."
        )

    @classmethod
    def _from_csv(cls, path: Path, default_value: float, default_bpm: float) -> "AutomationCurve":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        points = [
            ControlPoint(_row_time_to_seconds(row, default_bpm), float(row["value"]))
            for row in rows
        ]
        return cls(points, default_value)

    @classmethod
    def _from_json(cls, path: Path, default_value: float, default_bpm: float) -> "AutomationCurve":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            raw_points = payload["points"]
            tempo = float(payload.get("tempo_bpm", default_bpm))
        else:
            raw_points, tempo = payload, default_bpm
        points = [
            ControlPoint(_row_time_to_seconds(raw, tempo), float(raw["value"]))
            for raw in raw_points
        ]
        return cls(points, default_value)

    @classmethod
    def _from_fl_studio(cls, path: Path, default_value: float, default_bpm: float) -> "AutomationCurve":
        try:
            import pyflp
        except ImportError as exc:  # pragma: no cover - optional integration
            raise RuntimeError(
                "FL Studio automation parsing requires the optional 'pyflp' package "
                "(pip install wavelet2vec[flstudio]). Alternatively export the "
                "automation to CSV or JSON."
            ) from exc

        project = pyflp.parse(path)
        ppq = int(getattr(project, "ppq", 96) or 96)
        tempo = float(getattr(project, "tempo", default_bpm) or default_bpm)
        points: list[ControlPoint] = []
        for channel in getattr(project, "channels", []):
            if "automation" not in type(channel).__name__.lower():
                continue
            for point in channel:
                tick = getattr(point, "position", None)
                raw_value = getattr(point, "value", None)
                # Do not use `or` here: 0 is a legitimate tick and 0.0 a
                # legitimate automation value.
                tick = 0 if tick is None else int(tick)
                value = default_value if raw_value is None else float(raw_value)
                points.append(ControlPoint(tick * 60.0 / (tempo * ppq), value))
        return cls(points, default_value)


def _row_time_to_seconds(row: dict, tempo_bpm: float) -> float:
    if "time_seconds" in row:
        return float(row["time_seconds"])
    if "seconds" in row:
        return float(row["seconds"])
    if "beat" in row:
        return float(row["beat"]) * 60.0 / tempo_bpm
    raise ValueError("Automation rows must contain time_seconds, seconds, or beat.")


def _modulated_settings(
    adsr: ADSR, modulate: str, value: float, max_amount: float
) -> tuple[ADSR, float]:
    """Maps an automation value in [0, 1] onto the character ADSR or amount.

    - ``amount``: morph depth becomes ``max_amount * value``
    - ``attack`` / ``decay`` / ``release``: the stage time is scaled by
      ``1 - value`` (higher automation = faster stage)
    - ``sustain``: the sustain level is scaled by ``value``
    """
    value = float(np.clip(value, 0.0, 1.0))
    if modulate == "amount":
        return adsr, max_amount * value
    if modulate in {"attack", "decay", "release"}:
        scaled = getattr(adsr, modulate) * (1.0 - value)
        return adsr.replace(**{modulate: scaled}), max_amount
    if modulate == "sustain":
        return adsr.replace(sustain=adsr.sustain * value), max_amount
    raise ValueError(f"modulate must be one of {MODULATION_TARGETS}, got {modulate!r}.")


def render_sequence(
    snippets: Sequence[np.ndarray],
    sample_rates: Sequence[int],
    *,
    character: SoundCharacter | None = None,
    character_adsr: ADSR | None = None,
    volume_adsr: ADSR | None = None,
    automation: AutomationCurve | None = None,
    modulate: str = "amount",
    max_amount: float = 1.0,
    gap_seconds: float = 0.0,
    output_sr: int | None = None,
    transfer_spectrum: bool = True,
    transfer_envelope: bool = True,
    max_gain_db: float = 24.0,
) -> tuple[np.ndarray, int]:
    """Renders a sequence of snippets with automation-driven character development.

    Each snippet is placed in order (with optional gaps). For snippet *i*, the
    automation curve is sampled at the snippet's normalized position in the
    sequence and the value modulates one parameter (``modulate``) of the
    character morph — for example, with ``modulate="attack"`` and a rising
    automation clip, each successive snippet grows into the character faster,
    developing the texture across the sequence.

    Per snippet, the processing chain is:

    1. resample to the output rate,
    2. character morph shaped by ``character_adsr`` (time-varying amount) or
       applied statically when no character ADSR is given,
    3. volume ADSR (applied as-is, not modulated).

    Returns ``(waveform [channels, samples], sample_rate)``.
    """
    if not snippets:
        raise ValueError("At least one snippet is required.")
    if len(snippets) != len(sample_rates):
        raise ValueError("snippets and sample_rates must have equal length.")
    if modulate not in MODULATION_TARGETS:
        raise ValueError(f"modulate must be one of {MODULATION_TARGETS}, got {modulate!r}.")

    rate = int(output_sr if output_sr is not None else sample_rates[0])
    prepared = [
        resample(_as_channels(snippet), int(snippet_rate), rate)
        for snippet, snippet_rate in zip(snippets, sample_rates)
    ]
    n_channels = max(part.shape[0] for part in prepared)
    prepared = [
        np.repeat(part, n_channels, axis=0) if part.shape[0] == 1 and n_channels > 1 else part
        for part in prepared
    ]

    gap = np.zeros((n_channels, int(gap_seconds * rate)), dtype=np.float64)
    starts = np.cumsum([0.0] + [part.shape[1] / rate + gap_seconds for part in prepared[:-1]])
    total = starts[-1] + prepared[-1].shape[1] / rate

    rendered: list[np.ndarray] = []
    for index, part in enumerate(prepared):
        value = (
            automation.value_at_fraction(starts[index] / max(total, 1e-8))
            if automation is not None
            else 1.0
        )
        processed = part
        if character is not None:
            adsr, amount = _modulated_settings(
                character_adsr or ADSR(), modulate, value, max_amount
            )
            if character_adsr is not None:
                processed = apply_adsr_character(
                    processed,
                    rate,
                    character,
                    adsr,
                    max_amount=amount,
                    transfer_spectrum=transfer_spectrum,
                    transfer_envelope=transfer_envelope,
                    max_gain_db=max_gain_db,
                )
            else:
                processed = apply_character(
                    processed,
                    rate,
                    character,
                    amount=amount,
                    transfer_spectrum=transfer_spectrum,
                    transfer_envelope=transfer_envelope,
                    max_gain_db=max_gain_db,
                )
        if volume_adsr is not None:
            processed = apply_adsr_volume(processed, rate, volume_adsr)
        rendered.append(np.asarray(processed, dtype=np.float64))
        if gap.shape[1] > 0 and index < len(prepared) - 1:
            rendered.append(gap)

    return np.concatenate(rendered, axis=1).astype(np.float32), rate
