"""``sparkrun.api.run`` — launch an inference workload from the library API.

Orchestrates the full launch path:

1. Resolve recipe / cluster / hosts / runtime.
2. Apply overrides; resolve recipe (runtime selection finalized).
3. Run the scheduler via :func:`sparkrun.api.schedule` to compute placement.
4. Apply orthogonal constraints (solo, ``max_nodes``).
5. Delegate to :func:`sparkrun.core.launcher.launch_inference`.
6. Translate the launcher's :class:`LaunchResult` into :class:`RunResult`.

The function raises :class:`~sparkrun.api.SparkrunError` (or a
subclass) for any failure; on success it returns a populated
:class:`RunResult`.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from sparkrun.api._context import resolve_sctx
from sparkrun.api._errors import (
    InsufficientCapacity,
    LayoutRequired,
    SparkrunError,
)
from sparkrun.api._models import RunOptions, RunResult

if TYPE_CHECKING:
    from sparkrun.core.context import SparkrunContext
    from sparkrun.core.scheduler import RankAssignment

logger = logging.getLogger(__name__)


def run(options: RunOptions, *, sctx: "SparkrunContext | None" = None) -> RunResult:
    """Launch the workload described by *options* and return a :class:`RunResult`.

    Args:
        options: Inputs for the launch.
        sctx: Optional shared :class:`SparkrunContext`.  When omitted a
            fresh session is built; callers chaining multiple ``api.*``
            calls can construct one ``sctx`` and pass it to share
            config / registry-manager / cluster-manager state.

    Raises:
        :class:`InsufficientCapacity`: Scheduler can't fit the workload.
        :class:`LayoutRequired`: Cluster needs an explicit ``recipe.layout``.
        :class:`~sparkrun.api.RecipeNotFound`: Recipe lookup failed.
        :class:`~sparkrun.api.HostsUnreachable`: No usable host source.
        :class:`~sparkrun.api.TrustRejected`: Recipe hooks rejected.
        :class:`SparkrunError`: For other launch failures.
    """
    from sparkrun.api._resolve import (
        resolve_cluster,
        resolve_recipe,
        resolve_runtime,
    )
    from sparkrun.core.launcher import launch_inference
    from sparkrun.orchestration.job_metadata import (
        derive_placement_token_from_hosts,
        generate_cluster_id,
        generate_intent_id,
        generate_placement_token,
        parse_cluster_id,
    )

    sctx = resolve_sctx(sctx)
    started_at = time.time()
    config = sctx.config

    # 1. Resolve inputs.  `resolve_cluster` always returns a populated
    # ClusterDefinition (anonymous when only --hosts was given) so
    # downstream code never has to branch on ``cluster is None``.
    cluster_def = resolve_cluster(options.cluster, options.hosts, sctx=sctx, config=config)

    # Transport prepare: for provider-backed clusters (e.g. Thunder) this
    # refreshes ephemeral connection details (fresh IP/port, SSH key, managed
    # ssh alias) BEFORE any SSH runs — the occupancy status query inside
    # ``resolve_effective_hosts`` below is the first SSH.  No-op for plain-SSH
    # clusters, so existing clusters pay nothing.
    from sparkrun.api._resolve import prepare_transport

    prepare_transport(cluster_def, dry_run=bool(getattr(options, "dry_run", False)))

    recipe = resolve_recipe(options.recipe, sctx=sctx, overrides=options.overrides)
    hosts = list(cluster_def.hosts)
    runtime = resolve_runtime(recipe, sctx=sctx)

    # Scheduler selection chain: caller > recipe > cluster > greedy default.
    from sparkrun.core.scheduler import FALLBACK_DEFAULT_SCHEDULER, get_scheduler, resolve_scheduler_selector

    effective_scheduler, _scheduler_defaulted = resolve_scheduler_selector(
        cli=options.scheduler,
        recipe=getattr(recipe, "scheduler", None),
        cluster=getattr(cluster_def, "scheduler", None),
    )
    if _scheduler_defaulted:
        logger.debug("No scheduler configured (recipe/cluster); using default %r", FALLBACK_DEFAULT_SCHEDULER)

    # Apply the cluster's SSH user (if any) to the config so downstream
    # SSH operations (executor.run / distribution / build_ssh_kwargs)
    # log in as the right user.  Matches the CLI's resolution chain
    # where ``_resolve_hosts_or_exit`` applies ``cluster.user`` to
    # ``config.ssh_user`` before launch.
    if getattr(cluster_def, "user", None):
        try:
            config.ssh_user = cluster_def.user
        except Exception:
            logger.debug("Failed to apply cluster SSH user to config", exc_info=True)

    # 2. Compute placement via the single shared authority
    # (:func:`sparkrun.api._hosts.resolve_effective_hosts`).  This is the
    # same path the CLI ``run`` command and the benchmark flow use, so all
    # three place identically — the scheduler's ``hosts_used`` IS the
    # effective host list, ``runtime.world_size()`` is baked into the
    # request, and ``max_nodes`` / solo are applied as orthogonal
    # constraints.  ``notes`` (human-readable trim messages) are rendered
    # by the CLI shell, not the library, so the API discards them.
    from sparkrun.api._hosts import resolve_effective_hosts

    # Deterministic intent for this launch (recipe + overrides).  Passed to the
    # scheduler so a relaunch / resume of the same workload subtracts its own
    # still-running containers from the occupancy snapshot instead of treating
    # them as foreign load.  Reused below as the composed cluster_id's intent.
    intent_id = generate_intent_id(recipe, options.overrides)

    placement: "RankAssignment | None"
    is_solo_request = bool(options.solo) or recipe.mode == "solo"
    host_list, is_solo, _notes, placement = resolve_effective_hosts(
        list(hosts),
        recipe,
        options.overrides,
        cluster_def=cluster_def,
        runtime=runtime,
        sctx=sctx,
        solo=is_solo_request,
        scheduler=effective_scheduler,
        exclude_intent_id=intent_id,
    )

    # 3a. Compute intent_id + placement_token; compose cluster_id.
    # The launcher honours ``cluster_id_override`` so we hand it the
    # composed cluster_id rather than letting it derive one from
    # (recipe, hosts).
    #
    # The placement token's source depends on the scheduler:
    #   * Deterministic scheduler (greedy): derive the token from the
    #     candidate host set, exactly as the lookup paths
    #     (``stop`` / ``status`` / ``--ensure`` / ``derive_cluster_id``) do.
    #     Relaunching an identical workload then yields the same cluster_id
    #     and replaces the prior deployment — sparkrun 0.2.x semantics.
    #     We hash the *input* candidate hosts (not the trimmed ``host_list``)
    #     so the launched id matches what those lookup paths compute.
    #   * Status-aware scheduler (occupancy-*): use a fresh random token so
    #     the same intent placed on different host sets across launches gets
    #     distinct identifiers and never collides.
    try:
        scheduler_plugin = get_scheduler(effective_scheduler, v=sctx.variables)
        deterministic_placement = bool(getattr(scheduler_plugin, "deterministic_placement", False))
    except ValueError:
        # Unresolvable selector (e.g. a typo, or a single-host run that
        # short-circuited the scheduler so the name was never validated):
        # fall back to a random token — it can never collide.
        deterministic_placement = False
    if deterministic_placement:
        placement_token = derive_placement_token_from_hosts(hosts)
    else:
        placement_token = generate_placement_token()
    cluster_id_for_launch = options.cluster_id_override or generate_cluster_id(intent_id, placement_token)
    # Recover intent + token from the override when one was supplied so
    # the result still carries accurate metadata.
    if options.cluster_id_override:
        try:
            parsed_intent, parsed_token = parse_cluster_id(options.cluster_id_override)
            intent_id = parsed_intent
            placement_token = parsed_token
        except ValueError:
            # Non-canonical override (e.g. a user-supplied label) — keep
            # the freshly-computed intent_id but blank the token so
            # downstream consumers don't surface a fake one.
            placement_token = ""

    # 3a-bis. Evict this intent's superseded deployments.  ``exclude_intent_id``
    # above told the scheduler "my own containers aren't foreign load, I'm
    # replacing them" — this is the half that actually replaces them.  It is a
    # no-op on the deterministic (greedy) path, where the relaunch reuses the
    # prior cluster_id and the runtime's step-1 cleanup already removes those
    # containers by name.
    if not options.dry_run:
        _evict_superseded_deployments(
            intent_id=intent_id,
            cluster_id_for_launch=cluster_id_for_launch,
            candidate_hosts=hosts,
            target_hosts=host_list,
            cluster_def=cluster_def,
            config=config,
            sctx=sctx,
        )

    # 3b. Experimental k8s JobSet path (gated by the api.run.k8s feature flag).
    # When the resolved executor is k8s AND the flag is on, route to the
    # native Kubernetes launcher instead of the SSH-oriented launch_inference.
    # Flag off → fall through to the legacy k8s-executor-over-SSH draft.
    if config.is_feature_enabled("api.run.k8s"):
        from sparkrun.orchestration.executor import ExecutorUnavailableError, resolve_executor_name

        try:
            _executor_name = resolve_executor_name(
                cli_overrides=_build_executor_overrides(options),
                recipe=recipe,
                cluster=cluster_def,
                runtime=runtime,
                config=config,
                v=sctx.variables,
            )
        except ExecutorUnavailableError:
            _executor_name = None
        if _executor_name == "k8s":
            from sparkrun.api._run_k8s import run_k8s

            return run_k8s(
                options,
                sctx,
                recipe=recipe,
                runtime=runtime,
                cluster_def=cluster_def,
                host_list=host_list,
                placement=placement,
                is_solo=is_solo,
                cluster_id=cluster_id_for_launch,
                intent_id=intent_id,
                placement_token=placement_token,
                effective_scheduler=effective_scheduler,
                started_at=started_at,
            )

    # 4. Translate options → launch_inference kwargs.
    launch_kwargs: dict[str, Any] = {
        "recipe": recipe,
        "runtime": runtime,
        "host_list": host_list,
        "overrides": dict(options.overrides),
        "config": config,
        "v": sctx.variables,
        "sctx": sctx,
        "is_solo": is_solo,
        "transfer_mode": options.transfer_mode,
        "transfer_interface": options.transfer_interface,
        "cache_dir": options.cache_dir,
        "local_cache_dir": options.local_cache_dir,
        "dry_run": options.dry_run,
        "detached": options.detached,
        "follow": options.follow,
        "ray_port": options.ray_port,
        "dashboard_port": options.dashboard_port,
        "dashboard": options.dashboard,
        "init_port": options.init_port,
        "init_network": options.init_network,
        "executor_config": _build_executor_overrides(options),
        "extra_docker_opts": list(options.extra_docker_opts) if options.extra_docker_opts else None,
        "rootless": not options.rootful,
        "auto_user": not options.rootful,
        "cluster": cluster_def,
        "placement": placement,
        "trust": bool(options.trust),
        "sync_tuning": options.sync_tuning,
        "topology": options.topology,
        "cluster_id_override": cluster_id_for_launch,
        "recipe_ref": options.recipe_ref,
        "preserve_model_perms": options.preserve_model_perms,
        "skip_model_fan_out": options.skip_model_fan_out,
    }

    # 5. Launch.
    try:
        result = launch_inference(**launch_kwargs)
    except (KeyboardInterrupt, SystemExit):
        raise
    except (InsufficientCapacity, LayoutRequired, SparkrunError):
        # Typed API errors flow through unchanged.
        raise
    except Exception as e:
        raise SparkrunError("launch_inference failed: %s" % e) from e

    # 6. Build RunResult.
    metadata: dict[str, Any] = {
        "recipe": getattr(recipe, "qualified_name", None) or getattr(recipe, "name", None),
        "model": getattr(recipe, "model", None),
        "container_image": result.container_image,
        "serve_port": result.serve_port,
        "effective_cache_dir": result.effective_cache_dir,
    }
    if result.recipe_ref:
        metadata["recipe_ref"] = result.recipe_ref
    if result.runtime_info:
        metadata["runtime_info"] = dict(result.runtime_info)

    # Recover identifier components from the launcher's final cluster_id
    # in case it differs from the one we composed (e.g. an external
    # caller passed a non-canonical cluster_id_override through).
    final_cluster_id = result.cluster_id
    final_intent_id = intent_id
    final_placement_token = placement_token
    try:
        parsed_intent, parsed_token = parse_cluster_id(final_cluster_id)
        final_intent_id = parsed_intent
        final_placement_token = parsed_token
    except ValueError:
        # Non-canonical cluster_id (manual override) — keep the values
        # we computed pre-launch so RunResult still carries something
        # meaningful.
        pass

    run_result = RunResult(
        cluster_id=final_cluster_id,
        intent_id=final_intent_id,
        placement_token=final_placement_token,
        host_list=tuple(result.host_list),
        placement=placement,
        scheduler=_resolve_scheduler_name(effective_scheduler, sctx),
        runtime=runtime.runtime_name,
        executor=_executor_name_from_result(result),
        started_at=started_at,
        dry_run=options.dry_run,
        is_solo=result.is_solo,
        rc=int(result.rc),
        serve_command=result.serve_command or "",
        container_image=result.container_image or "",
        serve_port=int(result.serve_port or 0),
        effective_cache_dir=result.effective_cache_dir or "",
        runtime_info=dict(result.runtime_info or {}),
        metadata=metadata,
        launch_result=result,
    )
    from sparkrun.telemetry import emit_run_telemetry

    emit_run_telemetry(config, result=run_result, recipe=recipe, cluster=cluster_def, options=options)
    return run_result


def _evict_superseded_deployments(
    *,
    intent_id: str,
    cluster_id_for_launch: str,
    candidate_hosts: list[str],
    target_hosts: list[str],
    cluster_def,
    config,
    sctx: "SparkrunContext | None",
) -> list[str]:
    """Stop this intent's earlier deployments that sit on the hosts we're about to use.

    A launch's ``cluster_id`` is ``sparkrun_<intent_id>_<placement_token>``.
    Under a *deterministic* scheduler (greedy) the token is derived from the
    host set, so a relaunch reuses the prior cluster_id and the runtime's
    "Step 1: clean up existing containers" removes the previous deployment by
    name.  Under a **status-aware scheduler** (``occupancy-*``) the token is
    freshly random, so the new cluster_id can never match the old containers'
    names — step 1 becomes a no-op and the previous deployment keeps running,
    holding VRAM/RAM and the serve port (issue #223).

    ``resolve_effective_hosts(..., exclude_intent_id=...)`` has already
    subtracted this intent's occupancy from the scheduling snapshot on the
    premise that the relaunch *replaces* it.  This function is the half that
    makes that premise true.

    Scope is deliberately narrow:

    * **Same intent only.**  Foreign workloads are never touched — a second
      recipe sharing the cluster is a capacity question for the scheduler,
      not something a launch may unilaterally kill.
    * **Only deployments overlapping** *target_hosts*.  Running the same
      intent twice on disjoint host subsets is a supported use of the random
      placement token (see :func:`generate_placement_token`), so a
      non-overlapping sibling deployment is left alone.
    * An overlapping deployment is torn down **across every host it occupies**
      within *candidate_hosts*, not just the overlapping ones — half a
      distributed job is dead weight either way.

    Best-effort: discovery or teardown failures are logged and swallowed so
    they can't block a launch that may well succeed anyway.

    Returns the cluster_ids that were evicted (empty when there was nothing
    to do).
    """
    import sparkrun.api as api
    from sparkrun.orchestration.executor import query_status_for_cluster
    from sparkrun.orchestration.primitives import build_ssh_kwargs

    try:
        status = query_status_for_cluster(
            cluster_def,
            list(candidate_hosts),
            ssh_kwargs=build_ssh_kwargs(config) if config else {},
            config=config,
            v=sctx.variables if sctx is not None else None,
        )
    except Exception as e:
        logger.debug("Could not query cluster status for eviction; skipping: %s", e)
        return []

    prefix = "sparkrun_%s_" % intent_id
    target = set(target_hosts)
    # cluster_id -> every host in the snapshot it occupies (insertion-ordered).
    occupied: dict[str, list[str]] = {}
    overlapping: list[str] = []
    for entry in status.hosts:
        for workload in entry.workloads:
            cid = workload.cluster_id
            if cid == cluster_id_for_launch:
                continue
            if workload.intent_id != intent_id and not cid.startswith(prefix):
                continue
            hosts_for_cid = occupied.setdefault(cid, [])
            if entry.host not in hosts_for_cid:
                hosts_for_cid.append(entry.host)
            if entry.host in target and cid not in overlapping:
                overlapping.append(cid)

    evicted: list[str] = []
    for cid in overlapping:
        logger.info(
            "Replacing earlier deployment %s of this workload on %s",
            cid,
            ", ".join(occupied[cid]),
        )
        try:
            result = api.stop(cluster_id=cid, hosts=occupied[cid], cluster=cluster_def, sctx=sctx)
        except Exception as e:
            logger.warning("Could not stop earlier deployment %s: %s — it may still hold GPU memory", cid, e)
            continue
        if result.hosts_failed:
            logger.warning(
                "Teardown of earlier deployment %s did not confirm on %s — it may still hold GPU memory",
                cid,
                ", ".join(result.hosts_failed),
            )
        evicted.append(cid)
    return evicted


def _build_executor_overrides(options: RunOptions) -> dict[str, Any]:
    """Flatten ``options.executor`` + ``options.executor_config`` into the
    ``cli_overrides`` dict that ``launch_inference`` forwards to
    :func:`sparkrun.orchestration.executor.resolve_executor`."""
    overrides: dict[str, Any] = {}
    if options.executor:
        overrides["executor"] = options.executor
    if options.executor_config:
        for key, value in options.executor_config.items():
            overrides[key] = value
    return overrides


def _resolve_scheduler_name(effective_scheduler, sctx):
    """Return the registered ``scheduler_name`` for *effective_scheduler*.

    Looking up the scheduler plugin guarantees ``RunResult.scheduler``
    carries the *actually-used* name (e.g. ``"occupancy-sparse"`` when
    the caller relied on the project default) rather than echoing the
    possibly-``None`` selector that was passed in.
    """
    from sparkrun.core.scheduler import FALLBACK_DEFAULT_SCHEDULER, get_scheduler

    try:
        plugin = get_scheduler(effective_scheduler, v=sctx.variables if sctx is not None else None)
        return plugin.scheduler_name
    except Exception:
        return effective_scheduler or FALLBACK_DEFAULT_SCHEDULER


def _executor_name_from_result(result) -> str:
    """Recover the executor's name from the launcher's runtime, if it was set.

    The launcher stamps ``runtime.executor`` during launch; we read its
    ``executor_name`` attribute.  Falls back to ``"docker"`` (the
    library default) when the launcher didn't populate it (e.g. dry-run
    paths that short-circuit before executor resolution).
    """
    executor = getattr(result.runtime, "executor", None)
    if executor is None:
        return "docker"
    return getattr(executor, "executor_name", "docker")


__all__ = ["run"]
