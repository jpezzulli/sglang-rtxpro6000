# SPDX-License-Identifier: Apache-2.0
"""Early TP>1 NCCL P2P guidance for consumer-PCIe hosts (Pennyroyal SM120).

RTX PRO 6000 workstations have reported NCCL startup failures on TP>1 when
peer access (P2P) is unavailable or mis-advertised -- commonly PCIe ACS or
BIOS IOMMU settings. NCCL itself often surfaces this only as a hang or a
low-level transport error well inside ``init_process_group``. This module
emits one clear guidance line *before* any scheduler process initializes
torch.distributed, and recognizes ``NCCL_P2P_DISABLE=1`` as the operator's
opt-in workaround without ever setting environment or BIOS state itself.

Deliberate limits: guidance only.  Peer access on the *operator's* host is
not verified here (that would need a probe, and TP2 itself is not yet
hardware-qualified on this image), we do not probe ``can_device_access_peer``
(that would force a CUDA context onto the wrong device before the platform's
NUMA/affinity setup), we do not set ``NCCL_*``, and we do not claim a
diagnosis -- exactly one rank prints the hint, the operator decides.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

_P2P_ISOLATION_HINT = (
    "This launch did not verify GPU peer access on this host. If it hangs or "
    "fails inside NCCL transport init, NCCL_P2P_DISABLE=1 exported in the "
    "launcher environment can help isolate a PCIe ACS/IOMMU peer-access "
    "problem; that forces staging through host memory, so expect a possible "
    "inter-GPU throughput/latency cost. The recipe honors the value "
    "unchanged and never sets it for you."
)


def p2p_startup_guidance(
    *,
    tp_size: int,
    backend: Optional[str] = "nccl",
    node_rank: int = 0,
    tp_rank: int = 0,
    env: Mapping[str, str] | None = None,
) -> Optional[str]:
    """Return the guidance line this process should log, or None.

    Only the first TP rank of the first node logs (one hint per launch, not
    one per rank); TP1 -- the only topology qualified for Pennyroyal so far
    -- never sees a word of it, so a single-GPU launch cannot mistake this
    for a problem.
    """
    if tp_size is None or int(tp_size) <= 1:
        return None
    if str(backend).lower() != "nccl":
        # Non-NCCL backends (mooncake, gloo) own their transport messaging.
        return None
    if int(node_rank) != 0 or int(tp_rank) != 0:
        return None
    environ = os.environ if env is None else env
    # Only the exact NCCL value "1" means disabled; anything else (even "2"
    # or "true") leaves NCCL's own parsing in charge and gets no claim from
    # us beyond the isolation hint.
    if environ.get("NCCL_P2P_DISABLE") == "1":
        return (
            f"TP={tp_size}: NCCL_P2P_DISABLE=1 is exported, so NCCL will "
            f"stage TP traffic through host memory instead of using P2P; "
            f"that works around P2P problems at a possible throughput and "
            f"latency cost. " + _P2P_ISOLATION_HINT
        )
    return (
        f"TP={tp_size}: TP>1 on this consumer-PCIe image is experimental and "
        f"not yet hardware-qualified. " + _P2P_ISOLATION_HINT
    )
