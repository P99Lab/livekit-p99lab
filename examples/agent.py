#
# Copyright (c) 2026, p99lab
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A LiveKit voice agent with a switch between VAD-only turn detection and turn-1-mini.

    TURN_DETECTION=vad          python examples/agent.py dev
    TURN_DETECTION=turn-1-mini  python examples/agent.py dev

Needs LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET in the environment, and uses LiveKit Inference for
speech-to-text, the LLM and text-to-speech. Swap in your own plugins if you prefer.
"""

import logging
import os

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.plugins import silero

from livekit_p99lab import Turn1Mini

MODE = os.environ.get("TURN_DETECTION", "turn-1-mini")
logging.getLogger("livekit_p99lab").setLevel(logging.DEBUG)


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    session = AgentSession(
        turn_handling={"turn_detection": Turn1Mini() if MODE == "turn-1-mini" else "vad"},
        vad=silero.VAD.load(),
        stt="deepgram/nova-3",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-3",
    )
    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions=(
                "You take down the details of a car accident for an insurance claim. Ask one short question at a "
                "time: name, phone number, where it happened, what happened. Keep replies under ten words."
            )
        ),
    )
    await session.generate_reply(instructions=f"Say hello and ask for the caller's name. Mode: {MODE}.")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
