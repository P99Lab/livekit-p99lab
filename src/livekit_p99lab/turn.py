#
# Copyright (c) 2026, p99lab
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""turn-1-mini as a streaming turn detector for LiveKit Agents.

Usage::

    from livekit_p99lab import Turn1Mini

    session = AgentSession(turn_handling={"turn_detection": Turn1Mini()}, vad=..., stt=..., ...)

The session pushes the user's audio into the detector as it arrives. When its
VAD has heard 200 ms of silence it asks for a prediction; the answer decides
whether the session waits its short or its long endpointing delay before
committing the turn.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque

import numpy as np
import soxr
from livekit import rtc
from livekit.agents.voice.turn import TurnDetectionEvent

from ._onnx import MODEL_SAMPLE_RATE, Turn1MiniStream, resolve_model_path
from .log import logger

THRESHOLD = 0.5
SETTLE_SECS = 0.5
PRE_SPEECH_MS = 500.0
VAD_START_SECS = 0.2


class Turn1Mini:
    """Local end-of-turn detector running the turn-1-mini ONNX model on the CPU.

    Pass it as ``turn_detection`` to an ``AgentSession``. The first use
    downloads a 22 MB model file from Hugging Face; after that nothing touches
    the network. English only.
    """

    def __init__(
        self,
        *,
        threshold: float = THRESHOLD,
        settle_secs: float = SETTLE_SECS,
        pre_speech_ms: float = PRE_SPEECH_MS,
        vad_start_secs: float = VAD_START_SECS,
        variant: str = "float32",
        model_path: str | None = None,
        repo_id: str | None = None,
        revision: str | None = None,
        cpu_count: int = 1,
    ):
        """Initialize the detector.

        Args:
            threshold: P(end of turn) below which the turn is treated as
                unfinished, so the session waits its long endpointing delay.
                The scores are not calibrated: tune this on your own audio.
            settle_secs: How long a prediction may wait for the score to reach
                the threshold as the silence continues, before it answers with
                the highest score seen. The model's score often rises during
                the first few hundred milliseconds of silence.
            pre_speech_ms: Audio from before the detected start of speech that
                the model gets as context.
            vad_start_secs: How long after the true start of speech the
                session's VAD reports it. That much extra audio is replayed
                into the model when a turn starts.
            variant: Published build to use: "float32" (22 MB) or "int8" (7 MB).
            model_path: Local path of a ``turn-1-mini.step*.onnx`` file. Skips
                the download.
            repo_id: Hugging Face repository to download from.
            revision: Revision of that repository.
            cpu_count: CPU threads for inference.
        """
        self._threshold = threshold
        self._settle_secs = settle_secs
        self._pre_speech_ms = pre_speech_ms
        self._vad_start_secs = vad_start_secs
        self._cpu_count = cpu_count
        kwargs = {k: v for k, v in (("repo_id", repo_id), ("revision", revision)) if v is not None}
        self._model_path = model_path or resolve_model_path(variant=variant, **kwargs)

    @property
    def model(self) -> str:
        return "turn-1-mini"

    @property
    def provider(self) -> str:
        return "p99lab"

    def _create_model(self) -> Turn1MiniStream:
        """Open the model. Kept separate so tests can substitute a fake."""
        return Turn1MiniStream(self._model_path, cpu_count=self._cpu_count)

    def stream(self, *, conn_options=None) -> Turn1MiniStreamAdapter:
        """Create the per-session stream the ``AgentSession`` drives."""
        return Turn1MiniStreamAdapter(
            model=self._create_model(),
            threshold=self._threshold,
            settle_secs=self._settle_secs,
            pre_speech_ms=self._pre_speech_ms,
            vad_start_secs=self._vad_start_secs,
        )


class Turn1MiniStreamAdapter:
    """The stream side of :class:`Turn1Mini`.

    The model runs as a live stream that restarts at each turn. LiveKit's
    stream interface carries no start-of-speech event, but the session calls
    ``cancel_inference`` whenever its VAD reports that speech started. The
    first such call after a turn boundary is taken as the start of the turn:
    the model restarts there, seeded with the most recent audio, which is how
    the model was measured. If that call never comes, the model simply keeps
    running from the previous turn boundary.
    """

    def __init__(
        self,
        *,
        model: Turn1MiniStream,
        threshold: float,
        settle_secs: float,
        pre_speech_ms: float,
        vad_start_secs: float,
    ):
        self._model = model
        self._threshold = threshold
        self._settle_secs = settle_secs

        self._resampler: soxr.ResampleStream | None = None
        self._input_rate = 0

        # The most recent audio at the model's rate, replayed at turn start.
        self._recent: deque[np.ndarray] = deque()
        self._recent_samples = 0
        self._keep = int((pre_speech_ms / 1000 + vad_start_secs) * MODEL_SAMPLE_RATE)

        self._awaiting_turn_start = True
        self._fut: asyncio.Future[TurnDetectionEvent] | None = None
        self._fut_peak = 0.0
        self._fut_audio_secs = 0.0
        self._fut_started = 0.0
        self._closed = False

    # region: what the session reads

    @property
    def model(self) -> str:
        return "turn-1-mini"

    @property
    def provider(self) -> str:
        return "p99lab"

    @property
    def is_fallback(self) -> bool:
        return False

    @property
    def prediction_timeout(self) -> float:
        # The settle window is counted in audio time; allow for late frames.
        return self._settle_secs + 0.5

    async def unlikely_threshold(self, language) -> float | None:
        return self._threshold

    async def backchannel_threshold(self, language) -> float | None:
        return None

    async def supports_language(self, language) -> bool:
        # The model listens to audio and was trained on English. An unknown
        # language (no transcript yet) is let through.
        if language is None:
            return True
        return str(language).lower().split("-")[0] in ("en", "english")

    # endregion

    # region: what the session calls

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            return
        audio = self._to_model_rate(frame)
        if len(audio) == 0:
            return
        self._remember(audio)
        probs = self._model.push(audio)
        if self._fut is None or self._fut.done():
            return
        self._fut_audio_secs += len(audio) / MODEL_SAMPLE_RATE
        if len(probs):
            probability = float(probs[-1])
            self._fut_peak = max(self._fut_peak, probability)
            if probability >= self._threshold:
                self._resolve(probability)
                return
        if self._fut_audio_secs >= self._settle_secs:
            self._resolve(self._fut_peak)

    def predict(self) -> asyncio.Future[TurnDetectionEvent]:
        """Start a prediction for the pause the session's VAD just heard."""
        self._settle_pending(0.0)
        fut: asyncio.Future[TurnDetectionEvent] = asyncio.get_running_loop().create_future()
        if self._closed:
            fut.set_result(self._event(1.0))
            return fut
        self._fut = fut
        self._fut_started = time.perf_counter()
        self._fut_audio_secs = 0.0
        self._fut_peak = self._model.peek()
        if self._fut_peak >= self._threshold or self._settle_secs <= 0:
            self._resolve(self._fut_peak)
        return fut

    def cancel_inference(self, *, timed_out: bool = False) -> None:
        """Close the open prediction. Also the session's start-of-speech signal."""
        had_request = self._fut is not None and not self._fut.done()
        self._settle_pending(0.0)
        if self._awaiting_turn_start and not had_request and not timed_out:
            self._start_turn()

    def flush(self, reason: str | None = None) -> None:
        """Turn boundary: the next speech starts a new model stream."""
        self._settle_pending(0.0)
        self._awaiting_turn_start = True

    def end_input(self) -> None:
        self.flush()
        self._closed = True

    async def aclose(self) -> None:
        self.end_input()

    # endregion

    def _start_turn(self) -> None:
        self._awaiting_turn_start = False
        self._model.reset()
        if self._recent:
            self._model.push(np.concatenate(self._recent))

    def _settle_pending(self, probability: float) -> None:
        fut, self._fut = self._fut, None
        if fut is not None and not fut.done():
            fut.set_result(self._event(probability))

    def _resolve(self, probability: float) -> None:
        fut, self._fut = self._fut, None
        if fut is None or fut.done():
            return
        waited = time.perf_counter() - self._fut_started
        logger.debug(
            "turn-1-mini prediction",
            extra={"probability": round(probability, 4), "threshold": self._threshold},
        )
        fut.set_result(self._event(probability, inference_duration=waited))

    @staticmethod
    def _event(probability: float, inference_duration: float | None = None) -> TurnDetectionEvent:
        return TurnDetectionEvent(
            type="eot_prediction",
            end_of_turn_probability=probability,
            last_speaking_time=time.time(),
            inference_duration=inference_duration,
        )

    def _remember(self, audio: np.ndarray) -> None:
        self._recent.append(audio)
        self._recent_samples += len(audio)
        while self._recent and self._recent_samples - len(self._recent[0]) >= self._keep:
            self._recent_samples -= len(self._recent.popleft())

    def _to_model_rate(self, frame: rtc.AudioFrame) -> np.ndarray:
        """Frame to mono float32 at 16 kHz."""
        pcm = np.frombuffer(frame.data, dtype=np.int16)
        if len(pcm) == 0:
            return np.zeros(0, np.float32)
        if frame.num_channels > 1:
            pcm = pcm.reshape(-1, frame.num_channels).mean(axis=1)
        audio = pcm.astype(np.float32) / 32768.0
        rate = frame.sample_rate
        if rate == MODEL_SAMPLE_RATE:
            return audio
        if self._resampler is None or self._input_rate != rate:
            self._resampler = soxr.ResampleStream(
                rate, MODEL_SAMPLE_RATE, 1, dtype="float32", quality="HQ"
            )
            self._input_rate = rate
        return self._resampler.resample_chunk(audio)
