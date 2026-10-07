#
# Copyright (c) 2026, p99lab
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LiveKit Agents turn detector for turn-1-mini, a small streaming end-of-turn model by p99lab."""

from .turn import Turn1Mini, Turn1MiniStreamAdapter

__all__ = ["Turn1Mini", "Turn1MiniStreamAdapter"]
__version__ = "0.1.0"
