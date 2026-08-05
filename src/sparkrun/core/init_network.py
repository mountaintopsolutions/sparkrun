"""Distributed-init network preference — vocabulary and resolution chain.

Native multi-node runtimes advertise *one* rendezvous address to every node
(``--master-addr`` / ``--dist-init-addr``) and pin their control-plane socket
interfaces (``NCCL_SOCKET_IFNAME``, ``GLOO_SOCKET_IFNAME``, ``NODE_IP`` →
``VLLM_HOST_IP``, …) alongside it.  A DGX Spark cluster has at least two
networks that address can live on:

* the **management** network — the default-route interface the hosts are
  SSH'd on (typically 1 GbE);
* the **fabric** — the ConnectX-7 / InfiniBand RoCE link the collective
  actually moves tensors over (typically 200 GbE, a separate subnet).

Historically sparkrun always chose management and fell back to the fabric only
when a worker could not reach the management head address (see
:func:`sparkrun.runtimes._init_network.select_init_network`).  That is the
right *automatic* behavior, but it is not always the right *operator*
behavior: a hand-rolled compose deployment on the same hardware pins
``MASTER_ADDR`` / ``VLLM_HOST_IP`` to the fabric so the torch-distributed
bootstrap, vLLM's message queue, and NCCL's out-of-band handshake all ride the
fast link.  This module is the knob that lets a recipe, cluster, or config say
so explicitly, without having to lie about the SSH host list.

Three values:

``auto``
    Legacy behavior: prefer management, substitute the fabric only when a
    worker cannot reach the management head address.
``fabric``
    Prefer the fabric.  Falls back to management (with a warning) when the
    fabric address set is incomplete or unreachable — a preference, never a
    hard failure.
``management``
    Pin management unconditionally; never substitute the fabric.

Resolution is the usual sparkrun layering, highest precedence first: CLI →
recipe → cluster → config → ``auto``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparkrun.core.cluster_manager import ClusterDefinition
    from sparkrun.core.config import SparkrunConfig
    from sparkrun.core.recipe import Recipe

INIT_NETWORK_AUTO = "auto"
INIT_NETWORK_FABRIC = "fabric"
INIT_NETWORK_MANAGEMENT = "management"

#: Canonical values, in the order they're offered on the CLI / in docs.
INIT_NETWORK_CHOICES: tuple[str, ...] = (
    INIT_NETWORK_AUTO,
    INIT_NETWORK_FABRIC,
    INIT_NETWORK_MANAGEMENT,
)

#: Spellings accepted as input and folded onto a canonical value.  ``ib`` is
#: how the code path itself is named (``InitNetworkSelection.network == "ib"``)
#: and how operators say it; ``cx7`` is the DGX Spark NIC.
_ALIASES: dict[str, str] = {
    "ib": INIT_NETWORK_FABRIC,
    "infiniband": INIT_NETWORK_FABRIC,
    "cx7": INIT_NETWORK_FABRIC,
    "mgmt": INIT_NETWORK_MANAGEMENT,
}


class InitNetworkError(ValueError):
    """Raised for an unrecognized ``init_network`` value."""


def normalize_init_network(value) -> str | None:
    """Fold *value* onto a canonical :data:`INIT_NETWORK_CHOICES` entry.

    Returns ``None`` for ``None`` / empty (meaning "this layer expresses no
    preference"), so callers can use it directly as a chain layer.  Raises
    :class:`InitNetworkError` for a non-empty value that isn't a known
    spelling — a typo'd ``init_network: fabrik`` must fail loudly rather than
    silently running the rendezvous on the slow network.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise InitNetworkError("init_network must be a string, got %r" % (value,))
    name = value.strip().lower()
    if not name:
        return None
    name = _ALIASES.get(name, name)
    if name not in INIT_NETWORK_CHOICES:
        raise InitNetworkError("Unknown init_network %r. Valid values: %s" % (value, ", ".join(INIT_NETWORK_CHOICES)))
    return name


def resolve_init_network(
    *,
    cli: str | None = None,
    recipe: "Recipe | None" = None,
    cluster: "ClusterDefinition | None" = None,
    config: "SparkrunConfig | None" = None,
) -> str:
    """Return the effective init-network preference for a launch.

    Layers, highest precedence first: *cli* → ``recipe.init_network`` →
    ``cluster.init_network`` → ``config.init_network`` → :data:`INIT_NETWORK_AUTO`.
    Each layer is normalized, so any layer may use an alias (``ib``) and an
    unset layer (``None`` / ``""``) simply defers to the next.

    Non-string attribute values are ignored rather than rejected — this runs
    deep inside the launch path where ``recipe`` / ``cluster`` are frequently
    ``MagicMock``\\ s whose auto-attributes would otherwise poison the chain
    (the same guard :func:`sparkrun.orchestration.executor._coerce_str`
    applies).  Real typos are caught at the parse boundaries, which call
    :func:`normalize_init_network` directly and raise.
    """
    for value in (
        cli,
        getattr(recipe, "init_network", None),
        getattr(cluster, "init_network", None),
        getattr(config, "init_network", None),
    ):
        if not isinstance(value, str):
            continue
        name = normalize_init_network(value)
        if name:
            return name
    return INIT_NETWORK_AUTO
