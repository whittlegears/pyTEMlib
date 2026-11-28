"""Names the analysis server shares with its clients over the wire.

They are asyncroscopy's (``asyncroscopy.core.names``, ``asyncroscopy.contracts``), copied so that pyTEMlib does not
depend on asyncroscopy; asyncroscopy's tests check that both say the same.
"""

from __future__ import annotations

# Who is calling and which agent run the call belongs to. HTTP header names ignore case, and servers read them lower-cased.
CALLER_HEADER = "X-Asyncroscopy-Caller"
RUN_HEADER = "X-Asyncroscopy-Run"

# Job states. A worker owns a job in RUNNING_JOB_STATES. ACTIVE_JOB_STATES adds a reservation
# not yet started. FINISHED_JOB_STATES never change again.
RUNNING_JOB_STATES = frozenset({"queued", "running", "aborting", "paused_suspended", "paused"})
ACTIVE_JOB_STATES = RUNNING_JOB_STATES | {"reserved"}
FINISHED_JOB_STATES = frozenset({"done", "failed", "aborted"})

ENVELOPE_VERSION = "1.1.0"  # the job envelope schema's schema_version
