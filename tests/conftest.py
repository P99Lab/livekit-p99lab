#
# Copyright (c) 2026, p99lab
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared test helpers: a scripted stand-in for the model and synthetic audio."""

import numpy as np
import pytest

from livekit import rtc

from livekit_p99lab import Turn1MiniStreamAdapter
from livekit_p99lab._onnx import FRAME_SAMPLES, FRAMES_PER_STEP, STEP_SAMPLES

SAMPLE_RATE = 16000


class FakeStream:
    """Stands in for Turn1MiniStream.

    The score is ``rise_per_sec`` times the length of the silence at the end
    of the audio pushed so far (capped at 1.0), so tests can predict it.
    """

    def __init__(self, rise_per_sec: float = 1.0):
        self.rise_per_sec = rise_per_sec
        self.resets = 0
        self.samples_since_reset = 0
        self.total_samples = 0
        self.reset()
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.samples_since_reset = 0
        self._pending = np.zeros(0, np.float32)
        self._silent_frames = 0
        self._last = 0.0

    @property
    def last_probability(self) -> float:
        return self._last

    def _frame_scores(self, audio: np.ndarray, silent_frames: int) -> tuple[list[float], int]:
        scores = []
        for frame in audio.reshape(-1, FRAME_SAMPLES):
            silent_frames = silent_frames + 1 if np.abs(frame).max() < 1e-3 else 0
            scores.append(min(1.0, self.rise_per_sec * silent_frames * 0.02))
        return scores, silent_frames

    def push(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, np.float32)
        self.samples_since_reset += len(chunk)
        self.total_samples += len(chunk)
        self._pending = np.concatenate([self._pending, chunk])
        out: list[float] = []
        while len(self._pending) >= STEP_SAMPLES:
            scores, self._silent_frames = self._frame_scores(
                self._pending[:STEP_SAMPLES], self._silent_frames
            )
            self._pending = self._pending[STEP_SAMPLES:]
            self._last = scores[-1]
            out.extend(scores)
        return np.asarray(out, np.float32)

    def peek(self) -> float:
        frames = len(self._pending) // FRAME_SAMPLES
        if frames == 0:
            return self._last
        scores, _ = self._frame_scores(self._pending[: frames * FRAME_SAMPLES], self._silent_frames)
        return scores[-1]


def tone(seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """A loud 220 Hz tone as int16 samples: 'speech' for the fake model."""
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return (0.5 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)


def silence(seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Digital silence as int16 samples."""
    return np.zeros(int(seconds * sample_rate), np.int16)


def frames(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE, frame_ms: int = 20, channels: int = 1):
    """Split int16 samples into rtc.AudioFrame objects of transport size."""
    size = sample_rate * frame_ms // 1000
    for start in range(0, len(pcm), size):
        chunk = pcm[start : start + size]
        if channels > 1:
            chunk = np.repeat(chunk, channels)
        yield rtc.AudioFrame(
            data=chunk.tobytes(),
            sample_rate=sample_rate,
            num_channels=channels,
            samples_per_channel=len(chunk) // channels,
        )


def feed(stream, pcm: np.ndarray, sample_rate: int = SAMPLE_RATE, channels: int = 1):
    for frame in frames(pcm, sample_rate, channels=channels):
        stream.push_audio(frame)


@pytest.fixture
def fake_stream():
    return FakeStream()


@pytest.fixture
def make_stream(fake_stream):
    def _make(**overrides):
        opts = {
            "threshold": 0.5,
            "settle_secs": 0.5,
            "pre_speech_ms": 0.0,
            "vad_start_secs": 0.0,
            **overrides,
        }
        return Turn1MiniStreamAdapter(model=fake_stream, **opts)

    return _make
