#
# Copyright (c) 2026, p99lab
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for the LiveKit stream adapter. No network, no model file."""

import asyncio

import pytest
from conftest import feed, silence, tone
from livekit.agents.voice.turn import _StreamingTurnDetector, _StreamingTurnDetectorStream

from livekit_p99lab import Turn1Mini


def test_satisfies_livekit_protocols(make_stream, monkeypatch, fake_stream):
    monkeypatch.setattr(Turn1Mini, "_create_model", lambda self: fake_stream)
    detector = Turn1Mini(model_path="unused.onnx")
    assert isinstance(detector, _StreamingTurnDetector)
    assert isinstance(detector.stream(), _StreamingTurnDetectorStream)
    assert (detector.model, detector.provider) == ("turn-1-mini", "p99lab")


async def test_thresholds_and_languages(make_stream):
    stream = make_stream(threshold=0.6)
    assert await stream.unlikely_threshold("en") == 0.6
    assert await stream.backchannel_threshold("en") is None
    assert await stream.supports_language("en-US")
    assert await stream.supports_language(None)
    assert not await stream.supports_language("de")
    assert stream.is_fallback is False
    assert stream.prediction_timeout == pytest.approx(1.0)


async def test_prediction_answers_at_once_when_the_score_is_already_high(make_stream):
    # The fake score rises 1.0 per second of silence: 0.64 s in, it is 0.64.
    stream = make_stream()
    stream.cancel_inference()  # the session's start-of-speech call
    feed(stream, tone(1.6))
    feed(stream, silence(0.64))
    fut = stream.predict()
    assert fut.done()
    assert fut.result().end_of_turn_probability == pytest.approx(0.64)
    assert fut.result().type == "eot_prediction"


async def test_prediction_waits_for_the_score_to_reach_the_threshold(make_stream):
    stream = make_stream()
    stream.cancel_inference()
    feed(stream, tone(1.6))
    feed(stream, silence(0.2))  # where the session asks: score 0.2
    fut = stream.predict()
    assert not fut.done()
    feed(stream, silence(0.24))  # one more step boundary: 0.44 is still short
    assert not fut.done()
    feed(stream, silence(0.2))  # 0.60 at the next boundary
    assert fut.done()
    assert fut.result().end_of_turn_probability >= 0.5


async def test_prediction_gives_up_after_the_settle_window(make_stream, fake_stream):
    fake_stream.rise_per_sec = 0.1
    stream = make_stream(settle_secs=0.5)
    stream.cancel_inference()
    feed(stream, tone(1.6))
    feed(stream, silence(0.2))
    fut = stream.predict()
    feed(stream, silence(0.48))
    assert not fut.done()
    feed(stream, silence(0.04))
    assert fut.done()
    assert 0.0 < fut.result().end_of_turn_probability < 0.5  # the highest score seen


async def test_speech_resuming_closes_the_prediction_as_not_finished(make_stream, fake_stream):
    stream = make_stream()
    stream.cancel_inference()
    feed(stream, tone(1.6))
    feed(stream, silence(0.2))
    fut = stream.predict()
    resets = fake_stream.resets
    stream.cancel_inference()  # the user spoke again
    assert fut.result().end_of_turn_probability == 0.0
    assert fake_stream.resets == resets  # same turn: the model stream continues


async def test_a_new_prediction_supersedes_the_open_one(make_stream):
    stream = make_stream()
    stream.cancel_inference()
    feed(stream, tone(1.6))
    feed(stream, silence(0.2))
    first = stream.predict()
    second = stream.predict()
    assert first.result().end_of_turn_probability == 0.0
    assert not second.done()


async def test_first_speech_after_a_turn_boundary_restarts_the_model(make_stream, fake_stream):
    stream = make_stream(pre_speech_ms=500, vad_start_secs=0.2)
    feed(stream, silence(5.0))  # the agent is talking; the user is quiet
    assert fake_stream.resets == 0
    stream.cancel_inference()  # start of speech
    assert fake_stream.resets == 1
    # Seeded with the last 0.7 s only, give or take one 20 ms frame.
    assert fake_stream.samples_since_reset == pytest.approx(0.7 * 16000, abs=320)

    feed(stream, tone(1.0))
    stream.cancel_inference()  # speech resumes mid-turn: no restart
    assert fake_stream.resets == 1

    stream.flush(reason="turn committed")
    feed(stream, silence(1.0))
    stream.cancel_inference()
    assert fake_stream.resets == 2


async def test_flush_closes_the_open_prediction(make_stream):
    stream = make_stream()
    stream.cancel_inference()
    feed(stream, tone(1.6))
    feed(stream, silence(0.2))
    fut = stream.predict()
    stream.flush()
    assert fut.done()


async def test_a_timeout_cancel_is_not_taken_as_start_of_speech(make_stream, fake_stream):
    stream = make_stream()
    stream.cancel_inference(timed_out=True)
    assert fake_stream.resets == 0


@pytest.mark.parametrize("rate,channels", [(8000, 1), (16000, 1), (24000, 1), (48000, 1), (48000, 2)])
async def test_audio_reaches_the_model_at_16k_mono(make_stream, fake_stream, rate, channels):
    stream = make_stream()
    stream.cancel_inference()
    feed(stream, tone(2.0, rate), sample_rate=rate, channels=channels)
    assert fake_stream.total_samples == pytest.approx(2.0 * 16000, rel=0.02)


async def test_closed_stream_ignores_audio_and_answers_finished(make_stream, fake_stream):
    stream = make_stream()
    await stream.aclose()
    feed(stream, tone(1.0))
    assert fake_stream.total_samples == 0
    assert stream.predict().result().end_of_turn_probability == 1.0
    await asyncio.sleep(0)
