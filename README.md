# livekit-p99lab

[turn-1-mini](https://huggingface.co/p99lab/turn-1-mini) as a turn detector for
[LiveKit Agents](https://github.com/livekit/agents). turn-1-mini is a 5.3M-parameter streaming end-of-turn model by
p99lab. It listens to the user's audio, runs on one CPU core, and needs no GPU and no network after the first
download. English only.

Status: alpha. Unit-tested and checked against LiveKit's Silero VAD on recorded phone-call turns; see
[What has been checked](#what-has-been-checked).

## Install

```bash
pip install git+https://github.com/P99Lab/livekit-p99lab
```

Requires `livekit-agents` 1.8 or later (below 2.0). The first use downloads a 22 MB model file from Hugging Face.

## Use

```python
from livekit.agents import AgentSession
from livekit.plugins import silero
from livekit_p99lab import Turn1Mini

session = AgentSession(
    turn_handling={"turn_detection": Turn1Mini()},
    vad=silero.VAD.load(),
    stt=..., llm=..., tts=...,
)
```

That one line replaces the session's turn detector. It needs a VAD in the session (Silero's defaults are fine) and
works with the speech-to-text, LLM, text-to-speech pipeline. A realtime speech-to-speech model that decides turns on
its provider's servers does not consult a local turn detector.

## How it decides

- The session streams the user's audio into the model as it arrives (about 1 to 2 ms of CPU per 160 ms of audio).
- When the session's VAD has heard 200 ms of silence, the session asks for a prediction. If the model's score is at
  or above `threshold`, the answer is immediate.
- Otherwise the prediction waits up to `settle_secs` for the score to reach the threshold as the silence continues,
  then answers with the highest score it saw. The score often rises in the first few hundred milliseconds of silence.
- A score at or above the threshold lets the session commit the turn after its short endpointing delay
  (`min_delay`). A lower score makes it wait the long one (`max_delay`), unless the user speaks again first.
- The model restarts at the first speech after each committed turn, with a short stretch of audio from before the
  speech as context.

## Options

| Parameter | Default | Meaning |
|---|---:|---|
| `threshold` | `0.5` | Score below which the turn is treated as unfinished. Not calibrated: tune it on your own audio |
| `settle_secs` | `0.5` | How long a prediction may wait for the score to reach the threshold |
| `pre_speech_ms` | `500` | Audio from before the detected start of speech that the model gets as context |
| `vad_start_secs` | `0.2` | How late the session's VAD reports the start of speech; that much more audio is replayed at turn start |
| `variant` | `"float32"` | `"float32"` (22 MB) or `"int8"` (7 MB) |
| `model_path` | `None` | A local `turn-1-mini.step*.onnx` file; skips the download |
| `cpu_count` | `1` | CPU threads for inference |

The session's own endpointing delays still apply and are set where they always are:
`turn_handling={"turn_detection": Turn1Mini(), "endpointing": {"min_delay": 0.3, "max_delay": 2.5}}`.

## What has been checked

- The detector and its stream satisfy LiveKit's streaming turn detector interfaces, and an `AgentSession` accepts it.
- 16 unit tests cover the call order the session uses: audio in, prediction, speech resuming, turn boundaries, sample
  rates from 8 to 48 kHz, mono and stereo.
- Driven with LiveKit's Silero VAD in the session's call order on 60 recorded English phone-call turns: 88% of
  predictions asked in the final silence answered at or above 0.5, and 8% of those asked in a mid-turn pause did.
- Not yet checked: a live call in a running LiveKit room.

## Limits

- English only. The model was trained on English and reports other languages as unsupported.
- The scores are not calibrated probabilities.
- Short answers (a name, one word) and spoken digit groups are the model's weakest cases: endings can score below the
  threshold and wait for `max_delay`, and a pause inside a phone number can score as an ending.

## Licence

BSD 2-Clause for this package. The model is released under Apache 2.0.
