# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**sparkrun** is a CLI tool for launching, managing, and stopping Docker-based LLM inference workloads on NVIDIA DGX
Spark systems. It orchestrates containers over SSH — no Slurm or Kubernetes required. The control machine doesn't need
to be a cluster member; it coordinates DGX Sparks remotely.

Each DGX Spark has one GPU with 128 GB unified memory, so tensor parallelism maps directly to node count (`--tp 2` = 2
hosts).

## Common Commands

```bash
# Install in development mode (editable)
uv sync

# Run full test suite
.venv/bin/python -m pytest tests/ -v

# Run a single test file
.venv/bin/python -m pytest tests/test_recipe.py -v

# Run a specific test
.venv/bin/python -m pytest tests/test_cli.py::test_run_command_basic -v

# Run with coverage
.venv/bin/python -m pytest tests/ --cov=sparkrun --cov-report=term-missing

# Lint (ruff, line-length 140, target py312)
ruff check src/ tests/
ruff format src/ tests/

# Run the CLI directly during development
.venv/bin/sparkrun --help
.venv/bin/sparkrun run --dry-run qwen3-1.7b-vllm

# Sync versions across packages (pyproject.toml + sparkrun-cc-plugin)
python scripts/update-versions.py
python scripts/update-versions.py --check   # CI-friendly verify
```

Versions are tracked in `versions.yaml` at the repo root and synced to all package files via
`scripts/update-versions.py`.

## Architecture

### Source Layout

```
src/sparkrun/
├── cli/                # Click CLI package (see CLI Architecture below)
├── core/               # Core data models, bootstrap, and business logic (see below)
├── runtimes/           # Runtime plugins (see below)
├── orchestration/      # SSH, Docker, InfiniBand, executors, collectives (see below)
├── transports/         # Cluster connectivity seam — how hosts are reached/prepared (ssh default + thunder)
├── platforms/          # HardwarePlatformPlugin registry (DGX Spark + generic NVIDIA today)
├── models/             # HuggingFace model download, distribution, and VRAM estimation
├── containers/         # Container image distribution (docker save/load over SSH)
├── tuning/             # Triton fused MoE kernel tuning for SGLang and vLLM
├── builders/           # Container image builder plugins (docker-pull, eugr)
├── diagnostics/        # Host and run diagnostic collection (NDJSON output)
├── proxy/              # Inference gateway (LiteLLM engine + gateway selection seam)
├── benchmarking/       # Benchmark framework plugins and result export (llama-benchy)
├── utils/              # Shared helpers (coerce_value, suppress_noisy_loggers, etc.)
└── scripts/            # Embedded bash scripts (IB detection, container launch, etc.)
```

### Core Data Models (`core/`)

Core domain logic extracted from the top-level package. All imports use `sparkrun.core.*` (e.g.,
`from sparkrun.core.config import SparkrunConfig`).

| Module                  | Purpose                                                                              |
|-------------------------|--------------------------------------------------------------------------------------|
| `bootstrap.py`          | SAF plugin initialization, runtime / benchmarking / builder / executor discovery     |
| `config.py`             | `SparkrunConfig` — reads `~/.config/sparkrun/config.yaml`, cache dir resolution      |
| `registry.py`           | `RegistryManager` — git-based recipe registry system (see Registry System below)     |
| `recipe.py`             | `Recipe` loading, validation, v1→v2 migration, config chain via SAF Variables         |
| `cluster_manager.py`    | `ClusterManager` — named cluster CRUD (YAML files in `~/.config/sparkrun/clusters/`) |
| `hosts.py`              | Host resolution priority chain (CLI → file → cluster → default)                      |
| `pending_ops.py`        | PID-based lock files for in-progress operations                                      |
| `benchmark_profiles.py` | Benchmark profile discovery, resolution, and rendering across registries             |
| `hardware.py`           | `AcceleratorSpec` / `HostHardware` / `default_dgx_spark_hardware()`                  |
| `hardware_probe.py`     | `probe_host` / `probe_hosts` — combined accelerator + InfiniBand SSH probe           |
| `fingerprint.py`        | Thin shim — accelerator-only parsing on top of the combined probe                    |
| `backend_select.py`     | `select_backends(HostHardware) -> BackendBundle`, `NoMatchingBackendError`           |
| `placement.py`          | `compute_placement()` — rank → (host, local-GPU) honoring `RecipeLayout`             |
| `layout.py`             | `RecipeLayout` / `Placement` dataclasses parsed from recipe `layout:` block          |
| `launcher.py`           | `launch_inference()`, `resolve_per_host_backends()`, `resolve_recipe_trust()`        |
| `init_network.py`       | `resolve_init_network()` — CLI → recipe → cluster → config init-network preference   |

### CLI Architecture (`cli/`)

The CLI was split from a single `cli.py` into a package for maintainability. The `__init__.py` defines the top-level
`main` Click group, registers all subcommands, and provides top-level aliases (`list`, `show`, `search`, `status`).

| Module            | Purpose                                                                                                                                                                                                                                                           |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `__init__.py`     | `main` Click group, command registration, top-level aliases                                                                                                                                                                                                       |
| `_common.py`      | Shared infrastructure: logging setup, Click parameter types (`RECIPE_NAME`, `REGISTRY_NAME`, `RUNTIME_NAME`, `CLUSTER_NAME`, `PROFILE_NAME`), decorators (`host_options`, `dry_run_option`), and reusable helpers (host resolution, recipe loading, VRAM display) |
| `_run.py`         | `run` command — launch inference workloads                                                                                                                                                                                                                        |
| `_stop_logs.py`   | `stop` and `logs` commands — stop workloads and stream container logs                                                                                                                                                                                             |
| `_setup.py`       | `setup` command group — shell completion, SSH mesh, model/container sync, permissions, cache, networking                                                                                                                                                          |
| `_cluster.py`     | `cluster` command group — create/list/show/delete/update saved cluster definitions, cluster status                                                                                                                                                                |
| `_recipe.py`      | `recipe` command group — list/show/search recipes across registries                                                                                                                                                                                               |
| `_registry.py`    | `registry` command group — add/remove/enable/disable/update registries, list/show benchmark profiles                                                                                                                                                              |
| `_benchmark.py`   | `benchmark` command group — run benchmark profiles against inference workloads                                                                                                                                                                                    |
| `_tune.py`        | `tune` command group — run Triton fused MoE kernel tuning (SGLang and vLLM)                                                                                                                                                                                       |
| `_wizard.py`      | `setup wizard` command — guided cluster setup                                                                                                                                                                                                                     |
| `_check.py`       | `setup check` command — non-destructive readiness probe of a cluster's hosts against the wizard's setup steps (ordered `SETUP_CHECKS` registry; seed of a future per-platform step system with paired check/apply stages)                                          |
| `_proxy.py`       | `proxy` command group — thin renderer over `api.proxy` (see Inference Gateway below)                                                                                                                                                                              |
| `_monitor_tui.py` | Textual TUI for `cluster monitor`                                                                                                                                                                                                                                 |
| `ext.py`          | Plugin CLI-command extension point — `register_cli_command(cmd, parent=…)` + `PluggableGroup` (see below)                                                                                                                                                          |

**Plugin CLI commands** (`cli/ext.py`): external plugins add Click commands via
`register_cli_command(command, parent=(...))` (`parent=()` → top-level;
`parent=("cluster","import")` → nested). The top-level `main` is a
`PluggableGroup`: on first command resolution it runs `ensure_cli_extensions`
(→ `init_sparkrun`, which imports plugins so they register, → attach), so plugin
commands appear in `--help` and dispatch like built-ins even though the command
tree is built at import time (before plugins load). Attachment is idempotent and
never clobbers a built-in; command↔plugin mapping is intentionally free-form
(not tied to the transport/executor abstractions). Per-command gating is the
command's own concern.

### Plugin System (SAF)

sparkrun uses [scitrera-app-framework](https://github.com/scitrera/python-app-framework) (SAF) for plugin discovery and
lifecycle. Six extension points are registered:

| Extension point        | Constant       | Module scanned                       | Base class              |
|------------------------|----------------|--------------------------------------|-------------------------|
| `sparkrun.runtime`     | `EXT_RUNTIME`  | `sparkrun.runtimes`                  | `RuntimePlugin`         |
| `sparkrun.builder`     | `EXT_BUILDER`  | `sparkrun.builders`                  | `BuilderPlugin`         |
| `sparkrun.benchmarking`| `EXT_BENCHMARKING` | `sparkrun.benchmarking`          | `BenchmarkingPlugin`    |
| `sparkrun.executor`    | `EXT_EXECUTOR` | `sparkrun.orchestration.executors`   | `Executor`              |
| `sparkrun.scheduler`   | (scheduler)    | `sparkrun.schedulers`                | `Scheduler`             |
| `sparkrun.transport`   | `EXT_TRANSPORT`| `sparkrun.transports`                | `Transport`             |

Key bootstrap flow: `cli/__init__.py` → `core.bootstrap.init_sparkrun()` → SAF `init_framework_desktop()` →
`find_types_in_modules(...)` over each scanned module above → `register_plugin()` for each discovered plugin (schedulers
and transports skip base classes with a blank `scheduler_name` / `transport_name`). Finally
`load_external_plugins(v)` loads any out-of-tree plugins (see External Plugins below).

The `EXT_PLATFORM` constant is defined in `platforms/base.py` for future SAF
entry-point discovery; today `platforms/__init__.py` keeps an ordered
in-process registry that callers iterate via `resolve_platform()`. Platforms
stay in-process (not SAF-scanned) because their resolution is **order-sensitive**
(most-specific `matches()` first) — transports, which select by exact name, moved
onto SAF; platforms did not.

### External Plugins (`core/external_plugins.py`)

Out-of-tree plugins (private executors, transports, runtimes, …) load from
directories listed under `plugins.paths` in `config.yaml`
(`SparkrunConfig.external_plugin_paths`). `load_external_plugins(v)`, called at
the end of `init_sparkrun`, prepends each dir to `sys.path`, imports every
top-level module/package in it, scans each for the six SAF plugin base types and
`register_plugin`s them, then calls an optional module-level `register(v)` hook —
the escape hatch for the still-in-process registries (`platforms`,
`collectives`), which register via `register_platform()` rather than SAF subclass
discovery. Loading is trusted by definition (the config + dirs are user-owned);
a broken plugin logs and is skipped, never breaking startup.

**Gated off by default** behind the `core.external_plugins` feature flag (off on
every channel). When the flag resolves off, the config-driven path returns
immediately **without reading `plugins.paths`** (let alone importing anything) —
the gate uses the same context-free `feature_gate_enabled` resolution as the
executor/transport gates, so no init cycle. Enable via `sparkrun setup features
enable core.external_plugins`. The flag gates only the auto-load (`paths=None`)
path; an explicit `paths=` argument (programmatic / a plugin's own tests)
bypasses it.

Test isolation uses a separate hard kill-switch, `SPARKRUN_NO_EXTERNAL_PLUGINS`
(set by `conftest.isolate_stateful`): the feature flag alone is insufficient
because pytest reads the developer's **real** `~/.config/sparkrun` (the SAF
stateful root isn't "ready" under pytest), so a developer who enabled the flag
would otherwise load their real plugins mid-suite. The kill-switch short-circuits
the auto-load path before the flag is even consulted.

### Runtime Architecture

All runtimes extend `RuntimePlugin` (in `runtimes/base.py`), which itself extends SAF's `Plugin` class. The base class
provides solo-mode orchestration; runtimes override `run()`/`stop()`/`follow_logs()` for multi-node support.

| Runtime              | File                           | Entry Point              | Clustering         | Strategy                                                                                    |
|----------------------|--------------------------------|--------------------------|--------------------|---------------------------------------------------------------------------------------------|
| **vllm-ray**         | `runtimes/vllm_ray.py`         | `VllmRayRuntime`         | Ray head/worker    | `"ray"` — starts Ray cluster, exec serve on head                                            |
| **vllm-distributed** | `runtimes/vllm_distributed.py` | `VllmDistributedRuntime` | Native distributed | `"native"` — each node runs serve independently (no Ray)                                    |
| **sglang**           | `runtimes/sglang.py`           | `SglangRuntime`          | Native distributed | `"native"` — each node runs serve with `--node-rank`                                        |
| **llama-cpp**        | `runtimes/llama_cpp.py`        | `LlamaCppRuntime`        | Experimental RPC   | `"native/rpc"` — workers run `rpc-server`, head connects via `--rpc`                        |
| **trtllm**           | `runtimes/trtllm.py`           | `TrtllmRuntime`          | MPI (native)       | `"native"` — sleep infinity containers + mpirun on head                                     |
| **eugr-vllm**        | `runtimes/eugr_vllm_ray.py`    | `EugrVllmRayRuntime`     | Ray (inherited)    | Extends VllmRayRuntime with eugr container builds and mods (v1 recipe support) (deprecated) |

Runtimes must implement `generate_command()` and `resolve_container()`. The `cluster_strategy()` return value determines
which orchestration path the base class uses.

**Node-command template** (`RuntimePlugin._make_node_command_args`): native
multi-node runtimes (`vllm-distributed`, `sglang`, `trtllm`) emit rank-specific
argv via this template rather than ad-hoc per-runtime construction. Subclasses
override the hook methods (`_node_rank_args`, `_master_args`, etc.) and inherit
the assembly.

**Executor resolution** (`RuntimePlugin._resolve_executor`): runtimes do not
construct executors directly. The base helper delegates to
`orchestration.executor:resolve_executor()` — a single layered chain (CLI →
recipe → `runtime.default_executor()` → per-executor adjustments →
`SparkrunConfig` → per-executor defaults → dataclass field defaults). The
previously-hardcoded `_KNOWN_EXECUTORS` set has been retired; selector
validation queries SAF via `get_extensions(EXT_EXECUTOR)`.

**Trust gating** (`launcher.py:resolve_recipe_trust`): each launch resolves a
single trust verdict shared by `pre_exec` (inside `runtime.run()`) and
`post_exec` / `post_commands` (inside `post_launch_lifecycle`). Local recipes
and default-registry recipes are auto-trusted; third-party registry recipes
prompt unless `--trust` is passed. See `docs/SECURITY.md`.

**Backend bundle**: `RuntimePlugin.run()` accepts a keyword-only
`backends: dict[str, BackendBundle] | None` resolved by
`launcher.resolve_per_host_backends()`. Runtimes route per-host env emission
through `_cluster_ops.resolve_comm_env(ctx, comm_env, backends)`. When
`backends` is `None`, `resolve_comm_env` falls back to the legacy NCCL
generator (byte-identical for NVIDIA hosts).

### Orchestration Layer (`orchestration/`)

All remote operations use **SSH stdin piping** — scripts are generated as Python strings and piped to `ssh <host> bash -s`. No files are ever copied to remote hosts.

- **`ssh.py`** — `RemoteResult` dataclass, `build_ssh_cmd()`, `run_remote_script()`, `run_remote_scripts_parallel()`, `run_rsync_parallel()`, `stream_remote_logs()`
- **`sudo.py`** — `run_with_sudo_fallback()` — tries non-interactive sudo in parallel, then falls back to password-based sudo for failures
- **`docker.py`** — Pure command-string generators (`docker_run_cmd`, `docker_exec_cmd`, etc.), cluster ID generation
- **`distribution.py`** — High-level resource distribution: IB detection, container image and model syncing to target hosts (orchestrates `models/`, `containers/`, and IB detection)
- **`infiniband.py`** — IB detection script generation, NCCL env var computation, IB IP mapping for fast transfers
- **`networking.py`** — ConnectX-7 NIC detection, IP assignment planning, CX7 configuration script generation, host key distribution
- **`primitives.py`** — Higher-level composition: `build_ssh_kwargs()`, `build_volumes()`, `merge_env()`, `detect_infiniband()`, `run_script_on_host()`, `cleanup_containers()`
- **`job_metadata.py`** — Persistent job metadata (cluster_id → recipe mapping) stored in `~/.cache/sparkrun/jobs/`
- **`executor.py`** — Public facade. Re-exports `Executor`, `ExecutorConfig`, `EXT_EXECUTOR`. `resolve_executor()` is the single sanctioned executor entry point; `query_status_for_cluster()` is the single status source (see Status Discovery below).
- **`executors/`** — Executor plugin package. `_base.py` (ABC + dataclass), `docker.py` (default), `local.py` (experimental, no container), `k8s.py` (experimental draft, `kubectl run`-driven). Discovered via SAF. Each declares a `status_scope` (default `"host"`).
- **`collectives/`** — `CollectiveBackend` ABC + implementations: `nccl.py` (default; wraps `infiniband.py`), `rccl.py` (AMD scaffold), `hccl.py` (Intel Gaudi scaffold). `get_backend(vendor)` is the lookup.
- **`hooks.py`** — `pre_exec` / `post_exec` / `post_commands` runners. Trust gating via `_confirm_hook_execution(trust=...)`.

### Distributed Init Network (management vs CX7 fabric)

A multi-node launch has to advertise **one** rendezvous address
(`--master-addr` / `--dist-init-addr`) and pin the control-plane socket
interfaces alongside it (`NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`,
`NODE_IP` → vLLM's `VLLM_HOST_IP`, Ray's `--node-ip-address`). On a DGX Spark
cluster that address can live on the management LAN (the SSH path) or on the
CX7/IB fabric (the link the collective actually uses).

The *detection* has always been there — `infiniband.detect_ib_for_hosts`
resolves per-host fabric IPs + interfaces, and `infiniband.pin_comm_env_to_ib`
re-pins a whole `ClusterCommEnv` onto them. What `core/init_network.py` adds is
the **policy**: historically the fabric was used *only* as a failure fallback
(`select_init_network` substituted it when a worker couldn't reach the
management head). `init_network` makes that an explicit preference:

- `auto` (default) — unchanged legacy behavior, management-first with fabric fallback.
- `fabric` (aliases `ib` / `cx7` / `infiniband`) — prefer the fabric; degrade to
  management **with a warning** if the address map is incomplete or unreachable.
  Never a hard failure.
- `management` (alias `mgmt`) — pin management, skip the reachability probe.

`resolve_init_network()` layers CLI → recipe `init_network:` → cluster
`init_network:` → `SparkrunConfig.init_network` → `auto`, and is called exactly
once per launch in `ClusterContext.build`, landing on `ClusterContext.init_network`.
Two consumers read it:

- **Native runtimes** (vllm-distributed, sglang, trtllm) — `select_init_network`
  in `runtimes/_init_network.py` picks the address set; the existing
  `pin_comm_env_to_ib` call then follows the verdict.
- **Ray** (`vllm-ray`) — has no `select_init_network` step (the head address
  comes back from `ray start`, which advertises `NODE_IP`), so the comm env
  *is* the lever: `_cluster_ops.resolve_comm_env_for_init` pins it when the
  preference is `fabric` and is byte-identical to `resolve_comm_env` otherwise.

SSH always stays on the management addresses — the preference only moves the
workload's own rendezvous/control traffic, so there is no need to pass fabric
IPs as `--hosts`.

### Status Discovery ("what's running where?")

All workload-status discovery flows through **one source**, `api.status`, in two
tiers:

- **`api.status(hosts, cluster=…) -> ClusterStatus`** — the lean *occupancy*
  snapshot (per-host `used_slots`/`free_slots`/`workloads`, `errors`). Consumed
  by the occupancy schedulers, `api/_hosts.py` placement, proxy discovery, and
  `api/_stop.py` teardown (intent→cluster_id). Data-only shape in
  `core/cluster_status.py`.
- **`api.status_report(hosts, cluster=…, cache_dir=…) -> ClusterStatusResult`** —
  the *display* tier: `classify_cluster_status(status(...))` shapes the snapshot
  into groups/solo/idle/pending + cached job-metadata enrichment (the CLI-facing
  aggregate in `core/cluster_manager.py`). Used by `cluster status` and `stop
  --all`.

Under the hood, `api.status` calls
`orchestration/executor.py:query_status_for_cluster(cluster, hosts, …)`, which
**sweeps every enabled executor on the cluster's status substrate and merges**:

- **`Executor.status_scope`** (ClassVar, default `"host"`) is the substrate an
  executor's `query_status` inspects. Executors sharing a scope inspect
  *disjoint* state on the *same* substrate (docker containers vs `local`
  pidfiles on the SSH hosts) and are merged (`ClusterStatus.merge` — N-way
  fold, first snapshot authoritative). A provider executor declares its own
  scope (`k8s`, `modal`) and is queried alone.
- The **cluster's scope** = `status_scope` of the executor it would launch with
  (`resolve_executor_name`, i.e. explicit override → cluster pin →
  config/default). So an SSH/Thunder cluster → `"host"` (docker + local); a
  Modal cluster → `"modal"`; a k8s cluster → `"k8s"`. The scope's default
  executor is queried first (wins per-`cluster_id` collisions); a single failing
  executor is skipped; an unresolvable executor (gated-off provider plugin)
  degrades to an empty snapshot rather than raising.

Each `Executor.query_status(hosts, …)` inspects its own backend (docker `docker
ps`, local pidfile scan, k8s/modal control plane) and returns a `ClusterStatus`.
There is no separate status extension point.

**Mount-source preflight** (`Executor.verify_mount_sources(paths, hosts, …)`) is
the substrate peer of `query_status` on the *write* path: "do these identity-mount
sources already exist where the workload will run?" It validates pre-placed model
weights (an absolute-path `model:` or `cluster_config.resolved_model_path`, which
skip download+distribution) *before* the launch commits to that skip. Host-substrate
executors (docker/local) override it to SSH `test -e` the hosts via the shared
`ssh.verify_host_paths` helper; provider executors (k8s/modal) probe their own
volumes; the base default is a safe no-op (`{}`). Wired at the launch choke point
by `launcher._verify_pre_placed_model` (skipped on `--dry-run`), which raises a
`RecipeError` listing host→missing-path gaps. Best-effort like `query_status`: an
unresolvable executor or unreachable host degrades to "couldn't verify" and never
blocks — only a *confirmed*-missing path fails the launch. This is why an
absolute-path model works from a **remote control machine** that isn't a cluster
member: the check runs on the *targets*, not the control node.

### Live monitoring (telemetry + occupancy)

Monitoring has a second axis alongside occupancy — **telemetry** (per-host/node
util/mem/temp) — abstracted the same way status is, by **substrate scope**:

- **`TelemetryProvider`** (`orchestration/telemetry/`, SAF ext point
  `EXT_TELEMETRY`) is the telemetry peer of `Executor.query_status`, selected by
  `scope` (matches `status_scope`). Telemetry is a substrate property, not
  per-executor — docker and local share one host source — so there is one
  provider per scope. Core ships `HostTelemetryProvider` (scope `"host"`,
  wrapping `ClusterMonitor` / `NvMonitorClusterMonitor`); k8s/modal providers
  ship in their plugins. `get_telemetry_provider(scope)` returns the stateless
  singleton (or `None` → occupancy-only monitoring). A live collection's state
  lives on the `TelemetrySession` from `provider.open(...)`.
- **`api.open_telemetry`** — raw telemetry session for a cluster's scope.
- **`api.open_live_monitor` / `api.live_monitor`** — compose the telemetry
  stream with a background `api.status` occupancy poll into
  `MonitorFrame` snapshots (per-host `HostActivity` = telemetry + the workloads
  occupying the host). Substrate-agnostic: a host cluster combines
  `host_monitor.sh` + docker/local occupancy, a k8s cluster would combine k8s
  metrics + k8s occupancy — clients see the same shape.

The `cluster monitor` TUI drives a `LiveMonitorSession`, so its Jobs column and
detail pane show **all** executors' workloads (docker + local + provider), not
just the docker containers `host_monitor.sh`'s embedded `docker ps` used to
report. The `--simple` / `--json` telemetry-only paths still use `ClusterMonitor`
directly.

### Transport Layer (`transports/`)

The **transport** is the connectivity seam: *how sparkrun reaches / prepares a
cluster's hosts* before the generic SSH machinery runs. It is orthogonal to the
**executor** (`orchestration/executors/`, *how the workload runs on the host*) —
a provider-transport cluster still uses the docker executor.

- **`base.py`** — `Transport` SAF `Plugin` (selector `transport_name`, extension point `EXT_TRANSPORT`) with `prepare(cluster, *, dry_run=…)` (default no-op) and its delete-time counterpart `cleanup_cluster(cluster, *, dry_run=…)` (release out-of-band state — ssh alias/key — on cluster delete). A transport self-gates via `required_feature_flag` (like `Executor`). Discovered via `find_types_in_modules("sparkrun.transports", Transport)` in `core.bootstrap` — mirrors `Executor`.
- **`ssh.py`** — `SshTransport` (`transport_name = "ssh"`), the default; `prepare()` is a no-op, so every existing cluster is byte-identical to before transports existed.
- **`__init__.py`** — SAF-backed resolution: `resolve_transport(name)` / `list_transports()` query `get_extensions(EXT_TRANSPORT)` by `transport_name` (returning the stateless SAF singleton). `prepare_cluster_transport(cluster)` (run/status/logs/stop) and `cleanup_cluster_transport(cluster)` (delete) are the single call-site helpers. `_require_transport_enabled` reads the resolved transport's `required_feature_flag` and fails closed at the `prepare` call site (never a silent SSH downgrade, never SAF `is_multi_extension` hiding) — a gated selector yields a clear "enable it with …" error rather than "unknown transport". `cleanup_cluster_transport` is deliberately **ungated** (teardown must succeed even if the flag was later disabled) and tolerant of an absent transport plugin.
- **Thunder Compute** (`transport: thunder`) is **no longer in core** — it was externalized to the out-of-tree `sparkrun_thunder` plugin (the reference example for the plugin system). It registers `ThunderTransport` (SAF), its own `transports.thunder` feature flag, and the `sparkrun cluster import thunder` command (via `register_cli_command`). Core keeps only the generic seam; a `transport: thunder` cluster fails closed unless the plugin is loaded (`core.external_plugins` + `transports.thunder`).

`ClusterDefinition.transport: str = "ssh"` + `provider_ref` select the transport
(serialized only when non-default). The single wiring is
`api/_resolve.py:prepare_transport(cluster_def)` — called by `api.run` / `api.status`
/ `api.logs` / `api.stop` right after `resolve_cluster`, before any SSH — which
translates `TransportError` → `SparkrunError` for clean CLI errors. **Layering:
`cli/api → transports → {core, orchestration}`; `orchestration` never imports
`transports`.** `sparkrun cluster import thunder` attaches one single-host cluster
per RUNNING instance (attach-only; multi-node out of scope, multi-GPU-per-node
handled by the probe).

### Tailscale Endpoint Publishing (`setup tailscale`)

`sparkrun setup tailscale` (gated behind `cli.setup.tailscale`, off by default) joins cluster
hosts to a **tailnet** and surfaces the inference HTTP endpoint to the rest of the user's network.
It is a **control-plane / endpoint-publishing** feature — orthogonal to the transport seam and NOT
a data-plane path (NCCL stays on InfiniBand/CX7). Auth is via a Tailscale **OAuth client** that mints
a short-lived, pre-authorized, **tagged** auth key per join batch (never a long-lived key in config);
exposure is a **raw tailnet port** (`http://<ip>:<port>/v1`), not `tailscale serve`.

3-layer split mirroring `setup k8s`: `cli/_setup/_tailscale.py` (thin Click group, self-gates like
`_k8s.py`) → `api/tailscale/` (console-free `join`/`status`/`expose`/`down` + dataclasses + errors)
→ `orchestration/tailscale/` (`api.py` stdlib OAuth + key-mint + device REST client, `scripts.py` +
`scripts/tailscale_join*.sh` join scripts driven through `run_with_sudo_fallback`, `local.py` for
control-machine `tailscale ip` probes used by `expose --proxy`). Layering: `cli → api.tailscale →
orchestration.tailscale → {orchestration.ssh/sudo, core.config}`. Design spec: `.slop/tailscale-setup.md`.

### Inference Gateway (`proxy/` + `api/proxy/`)

The **gateway** is the process fronting every discovered inference endpoint
behind one OpenAI-compatible API. Today there is exactly one implementation —
`ProxyEngine` (LiteLLM) — but the vocabulary and the seams are in place so a
second can be added as a peer rather than a special case. One word throughout:
**gateway** is the pluggable family, `proxy` is the user-facing command.

Two mechanisms, deliberately separate:

- **Availability** — `gateway.<name>` feature flag. `gateway.litellm` ships
  **enabled on every channel** (`default=True`, like `executor.docker`); a
  future gateway would ship off. Declared on the implementation as
  `ProxyEngine.required_feature_flag` / `gateway_name`, pre-shaping the
  eventual `GatewayPlugin`.
- **Selection** — exactly one gateway is used at a time, arbitrated in
  `proxy/gateway.py:resolve_gateway()`: an explicit name (`proxy.gateway:` in
  `proxy.yaml`) must be known *and* enabled; with no name, the default wins
  when enabled, else the single remaining enabled gateway, else
  `AmbiguousGatewayError`. The flag registry has **no** notion of
  mutually-exclusive flags — nothing stops a user enabling two, so resolution
  refuses to guess (mirrors `_default_executor_name`).

**Gate placement**: `ProxyEngine.start()` is the *one* enforcement point —
bringing a gateway up, checked before `--dry-run` so a dry run can't advertise
a start that would be refused. `stop` / `status` / `models` / `sync` /
`alias_*` and the auto-discover daemon's `_restart_proxy` path are **ungated**:
a proxy started while the flag was on must stay manageable (and stoppable)
after it is turned off, and the daemon keeps driving the engine it was started
with. Same rule `cleanup_cluster_transport` follows for transports.

`api/proxy/` is the console-free facade (mirrors `api/tailscale/`): `start`,
`stop`, `status`, `models`, `sync`, `add_alias` / `remove_alias` /
`list_aliases`, plus `resolve_gateway` / `list_gateways`. `cli/_proxy.py` is a
renderer over it, and the desktop sidecar calls it directly. `_engine_class()`
is the single place a gateway name becomes an implementation. The state file
records `gateway`, so management paths bind to *what is running* rather than to
what is currently configured. Layering: `cli → api.proxy → sparkrun.proxy →
{core, orchestration}`; `sparkrun.proxy` imports of `api` stay deferred
(`proxy.discovery` imports `sparkrun.api`, so module-level would be circular).

### SSH Access Bootstrap (`api/setup/`)

Every setup phase — CX7 detection, shared-cache detection, the mesh itself —
assumes passwordless control→host SSH. `api/setup/` is the console-free layer
that *establishes* that, and it is the first piece of the wizard written to be
GUI-drivable (the desktop app's sidecar can call it directly instead of
shelling out to a terminal wizard).

| Function                          | Purpose                                                                              |
|-----------------------------------|--------------------------------------------------------------------------------------|
| `probe_ssh_access`                | `BatchMode=yes` reachability sweep → `SshProbe` per host                              |
| `ensure_local_key`                | Find an existing identity, else generate `~/.ssh/sparkrun_ed25519`                    |
| `install_public_key_interactive`  | Install the pubkey on one host via password auth                                      |
| `mesh_ssh_keys_native`            | host↔host key mesh with **no local shell**                                            |

**Everything here runs on a bare Windows control machine.** The only external
binaries are `ssh` and `ssh-keygen` (Windows 10+ ships both); there is no
`bash`, no paramiko. Two design rules follow from that and should not be
relaxed:

- **Key material travels over stdin, never argv.** Scripts are generated with
  the key embedded (single-quoted, or a quoted heredoc for the mesh) and piped
  to `bash -s`, so nothing depends on the *local* platform's command-line
  quoting — the difference between working and mangling a key on Windows.
- **`probe_ssh_access` adds `StrictHostKeyChecking=accept-new`.** `build_ssh_cmd`
  already forces `BatchMode=yes`; with the stock `ask` policy a first-contact
  host fails host-key verification and would be misreported as *unreachable*
  rather than merely unknown.

`SshProbe` distinguishes **auth failure** (host answered, rejected us → a
bootstrap candidate), **host-key failure** (changed key → operator must resolve;
never auto-fixed), and **unreachable** (network/sshd). Only the first is
offered a key install. The install's exit code is treated as a hint —
success is confirmed by re-probing, which is the only trustworthy signal.

**CLI wiring** (`cli/_setup/_ssh.py`): `_ensure_ssh_access` is the wizard's gate
(prints, prompts, persists a generated key as `ssh.key`); `_default_ssh_user`
replaces the old `os.environ.get("USER", "root")` — POSIX-only, so on Windows it
made every first connection as `root`. `_run_ssh_mesh` prefers
`scripts/mesh_ssh_keys.sh` and falls back to `mesh_ssh_keys_native` when local
`bash` is absent. The wizard runs the gate **once**, after the cluster-name and
SSH-username prompts and before any other probe.

> Note: other `os.environ.get("USER", "root")` call sites remain outside setup
> (`cli/_common.py`, `orchestration/primitives.py:is_local_user`,
> `orchestration/distribution.py`, `core/launcher.py`). They affect *launch-time*
> cross-user decisions and are still POSIX-only.

### Recipe System

Recipes are YAML files with fields: `model`, `runtime`, `container`, `command`, `defaults`, `env`, `metadata`,
`min_nodes`, `max_nodes`. The `Recipe` class (`core/recipe.py`) uses SAF `Variables` for config chain resolution —
CLI overrides → recipe defaults → runtime defaults.

Recipe resolution: CLI → `find_recipe()` (module-level function in `core/recipe.py`) → searches bundled recipes, local
`./recipes/`, user config recipes, and git-cloned registries.

Two recipe format versions exist: v1 (eugr-style, auto-detected by `recipe_version: "1"` or presence of `build_args`/
`mods`) and v2 (sparkrun native). vLLM recipes are resolved to either `vllm-ray` (if Ray hints are present) or
`vllm-distributed` (default). See `RECIPES.md` for the full specification.

### Registry System

The `RegistryManager` (`core/registry.py`) tracks recipe collections from remote git repos using sparse checkouts.
Registries are stored in `~/.config/sparkrun/registries.yaml`; cached clones live under `~/.cache/sparkrun/registries/`.

**Registry assets** — recipes, benchmark profiles, tuning configs and mods are all "a named file under a per-registry
subdirectory", so the shape is data, not four code paths. `RegistryAsset` (`RECIPE_ASSET`, `BENCHMARK_ASSET`,
`TUNING_ASSET`, `MODS_ASSET`) names the subpath field, whether the scan recurses, and the extension precedence; the
generic machinery hangs off it:

| Function                   | Role                                                                                     |
|----------------------------|------------------------------------------------------------------------------------------|
| `_iter_registries`         | the one enabled / visibility / name filter (every scan routes through it)                  |
| `asset_dir`                | `<cache>/<entry.<subpath_field>>` when it exists — the four `_*_dir` accessors wrap this   |
| `find_asset_in_registries` | resolve one name; per-registry flat-then-recursive, optional `accept` predicate            |
| `iter_asset_files`         | the *catalog* peer — same rules, so listing and lookup can never disagree                  |
| `qualified_asset_name`     | the typeable `@registry/<relpath>` label used to disambiguate                              |

Two rules are shared by every asset kind and are the reason the scan is not a plain `rglob`:

- **Flat beats nested, per registry.** A flat `<dir>/<name>.yaml` wins and suppresses that registry's recursive scan —
  but never another registry's (the bug fixed in #227).
- **`.yaml` beats a same-stem `.yml`, per directory.** They are one asset spelled two ways; treating them as two would
  produce an "ambiguous" error no name could resolve. The same stem in *different* subdirectories stays two distinct
  assets, so the catalog is never deduped by name.

Ambiguity therefore means "genuinely several assets", and both `RecipeAmbiguousError` and `ProfileAmbiguousError` carry
path-qualified `labels` (shared wording via `format_ambiguity`). Tuning configs and mods share only `asset_dir` —
tuning lookup is by runtime and returns a collection, so it is deliberately not routed through
`find_asset_in_registries`.

**Default registry initialization** (first run, no `registries.yaml`):

1. `_load_registries()` → no file → `_default_registries()`
2. `_default_registries()` calls `_init_defaults_from_manifests()` which clones each URL in `DEFAULT_REGISTRIES_GIT` and
   reads its `.sparkrun/registry.yaml` manifest via `_discover_manifest_entries()`. URLs that fail are skipped
   individually (partial success).
3. Discovered manifest entries are merged with `FALLBACK_DEFAULT_REGISTRIES`: manifest entries take priority by name;
   non-conflicting fallback entries are appended to fill gaps in registry coverage.
4. The combined list is saved to `registries.yaml` for subsequent loads.
5. If all manifest URLs fail, pure `FALLBACK_DEFAULT_REGISTRIES` is returned (offline/no-git safety net).

**Manifest format** (`.sparkrun/registry.yaml` in a git repo): supports both canonical keys (`subpath`,
`tuning_subpath`, `benchmark_subpath`) and short keys (`recipes`, `tuning`, `benchmarks`). Canonical keys take
precedence when both are present.

**Shared clones**: When multiple registries point to the same git URL, a single shared clone is used at `_url_<hash>/`
with per-registry symlinks. Sparse checkout paths are the union of all subpaths for that URL.

**Reserved name prefixes**: Names starting with reserved prefixes (`sparkrun`, `official`, `arena`, etc.) can only be
used by repos hosted under allowed GitHub organizations (`spark-arena`, `scitrera`, `eugr`, `dbotwinick`,
`raphaelamorim`). Enforced via `validate_registry_name()`.

**Tab completion**: `RecipeNameType.shell_complete()` in `_common.py` supports `@registry/recipe` syntax — `@` prefix
lists registries, `@registry/` lists recipes from that registry. Falls back to showing registry names when recipe cache
isn't populated.

### Recipe Catalog ("what recipes exist?")

All recipe *enumeration* flows through one function, **`api.search_recipes(query, …) -> list[RecipeSummary]`**
(`api/_recipes.py`) — the console-free peer of `api.status` for the catalog rather than the cluster. It merges the
configured registries with working-directory recipes, applies the registry/runtime filters, and returns typed
`RecipeSummary` rows (`api/_models.py`; `.to_dict()` yields the legacy `core.recipe.recipe_summary` mapping the CLI
formatters and `--json` consume). `sparkrun list` / `sparkrun recipe search` and their aliases are thin renderers over
it, and the desktop sidecar calls it directly instead of reaching into `RegistryManager`.

Two knobs carry the semantic difference between the commands:

| Knob            | `list`  | `search` | Meaning                                                                       |
|-----------------|---------|----------|-------------------------------------------------------------------------------|
| `unique_names`  | `True`  | `False`  | One row per unqualified name (locals first) vs every copy                     |
| `include_local` | `True`  | `True`   | Include CWD recipes (dropped anyway when a registry filter is set)            |

`unique_names` is why `list` shows a name once while `search` shows all of its variants: a registry's recipe dir is
scanned with `rglob`, so `3x-spark-cluster/foo.yaml` and `4x-spark-cluster/foo.yaml` are *different recipes sharing a
qualified name*. Only literal repeats (same resolved path, e.g. via shared-clone symlinks) are dropped unconditionally
— never dedupe the catalog by name.

**Implicit registry scope**: the positional QUERY of `recipe list` / `recipe search` (and the top-level `list` /
`search` aliases) accepts the same `@registry` syntax recipe *names* use — `@community` and `@community/` mean
`--registry community`, and anything after the `/` becomes the remaining query (`@community/qwen`).
`core/registry.py:resolve_registry_filter()` is the single resolver for both spellings (exposed as
`api.resolve_recipe_filter` for callers that need to name the resolved filter, e.g. in an empty-result message): it
strips the scope, rejects a scope that conflicts with an explicit `--registry`, then validates the resulting name —
unknown or disabled registries raise `RegistryFilterError` (→ `api.InvalidRegistryFilter` → `click.UsageError`)
listing the available ones, rather than silently yielding "No recipes found", whether the name arrived via the
shorthand or `--registry` (matching `registry list-benchmark-profiles`, which already validated upfront). A registry
filter implies `include_hidden` — naming a registry outranks its visibility default. The `RECIPE_QUERY` param type
completes `@` into `@registry/` and delegates to `RECIPE_NAME` past the slash.

`core.recipe.recipe_matches_query()` is the one matching predicate (substring over name / file / model /
description), shared by `RegistryManager.search_recipes` and the CWD scan so a local recipe is found on the same
terms as a registry one.

### Model & Container Distribution

Before launching, sparkrun can pre-sync models and container images from the control machine to target hosts:

- **Models** (`models/`): Downloads from HuggingFace Hub locally via `snapshot_download` (`models/download.py`), then
  rsyncs to targets (`models/distribute.py`, `models/sync.py`). GGUF models use colon syntax (`repo:quant`) for
  selective quant-file download.
- **Containers** (`containers/`): Pulls image locally (`containers/registry.py`), then streams via
  `docker save | ssh docker load` (`containers/distribute.py`, `containers/sync.py`). Checks image IDs to skip hosts
  that already have the correct image.
- **VRAM estimation** (`models/vram.py`): Estimates VRAM usage based on model parameter count, dtype, and quantization.
  Supports HuggingFace model auto-detection to resolve parameter counts. KV cache sizing has two paths: the generic
  `2 * layers * kv_heads * head_dim * bytes` formula, and an **MLA** path for DeepSeek architectures, which cache one
  compressed latent per token per layer. MLA is selected by `kv_lora_rank` in the HF config or by a fixed-slot KV
  layout named in the recipe (`nvfp4_ds_mla` / `fp8_ds_mla`, sized from `_MLA_SLOT_BYTES` and DeepSeek V4's per-layer
  `compress_ratios`). The MLA latent cache is **replicated across TP ranks**, so only pipeline parallelism divides it.

### Kernel Tuning (`tuning/`)

Provides utilities for running Triton fused MoE kernel tuning on DGX Spark and auto-mounting the resulting configs in
inference runs. Common tuning internals live in `tuning/_common.py`; runtime-specific helpers are in `tuning/sglang.py`
and `tuning/vllm.py`. `tuning/sync.py` handles syncing tuning configs from registries to local cache and runtime name
normalization.

### Utilities (`utils/`)

Shared helpers used across multiple modules to avoid circular imports:

- `coerce_value()` — type coercion for CLI string inputs (to int, float, bool)
- `suppress_noisy_loggers()` — silences verbose HTTP/transport loggers
- `resolve_ssh_user()` — SSH user resolution (cluster → config → env → fallback)
- `is_valid_ip()`, `parse_kv_output()`, `load_yaml()` — parsing helpers
- `cli_formatters.py` — Presentation-layer formatting for recipe tables and CLI output

### Config & State Paths

| Path                                 | Purpose                                           |
|--------------------------------------|---------------------------------------------------|
| `~/.config/sparkrun/config.yaml`     | User configuration                                |
| `~/.config/sparkrun/clusters/*.yaml` | Named cluster definitions                         |
| `~/.config/sparkrun/registries.yaml` | Custom recipe registry list                       |
| `~/.cache/sparkrun/registries/`      | Git-cloned recipe registries                      |
| `~/.cache/sparkrun/jobs/`            | Job metadata (cluster_id → recipe mapping)        |
| `~/.cache/sparkrun/pending/`         | PID lock files for in-progress operations         |
| `~/.cache/huggingface/`              | HuggingFace model cache (mounted into containers) |

### Feature Flags (`core/features.py`)

Channel-aware gating for experimental plugins and behavior. Each `FeatureFlag`
(registered in the module-level `FEATURE_FLAGS` registry) carries a
`description`, per-channel `channel_defaults`, and a baseline `default`.
`is_feature_enabled(name, config=…)` resolves with precedence: env override
(`SPARKRUN_FEATURE_<NAME>`, dots→underscores) → `features.<name>` in
`config.yaml` → per-channel default → baseline → fail-closed for unknown flags.

The active channel reuses the release channel from `core/channels.py`
(`stable`/`beta`/`alpha`): `SparkrunConfig.feature_channel` reads
`features.channel`, falling back to `self_update.channel`. Via `channel_defaults`
a flag can be on-by-default for `alpha` while off for `stable`/`beta`. The
built-in flags — `executor.local` and `executor.k8s` (gating the corresponding
experimental executors), `cli.setup.k8s` (gating the entire `sparkrun setup
k8s` command group), `cli.setup.tailscale` (gating the `sparkrun setup
tailscale` group), and `transports.thunder` (gating the Thunder Compute
transport + `cluster import thunder`) — are off by default on **every** channel;
enable them explicitly per-flag. The `setup k8s` group self-gates in its Click
callback (raises pointing at `setup features enable cli.setup.k8s`) and hides
itself from `setup --help` unless the flag resolves on at import; `setup
tailscale` and `cluster import thunder` gate the same way
(`cli.setup.tailscale` / `transports.thunder`), and the Thunder transport also
fails closed at use in `transports.prepare_cluster_transport` so an
already-imported Thunder cluster can't run once the flag is off.

**Docker gate (`executor.docker`)**: the default executor gates like every other
one — for uniformity and a future opt-out — but ships **enabled on every channel**
(`default=True`, no channel overrides). Disabling it (`features.executor.docker:
false`) removes docker from `list_executors()` / explicit selection, and the
baseline-default resolution honors that: `_resolve_executor_name` no longer
hard-codes `"docker"` when no layer names an executor — `_default_executor_name`
returns docker when enabled, else the sole enabled executor, else raises "name
one / set `default_executor`" (never silently runs on a disabled backend).

**Gateway gate (`gateway.litellm`)**: same shape as the docker gate — ships
enabled on every channel, exists so an alternate inference gateway can be added
as a peer. Exclusivity ("one gateway at a time") is arbitrated at *resolution*,
not by the flag registry. See Inference Gateway above.

**Visibility-only gate**: `cli.setup.features` (via `channel_defaults`, **on for
`beta`/`alpha`, off for `stable`**) is different — it does NOT gate execution.
The `setup features` group is always functional; the flag only decides whether it
appears in `setup --help` (`@setup.group("features", hidden=not
_setup_features_visible_at_import())`, no callback raise). So a stable user can
still run `setup features enable <flag>` even though the group is hidden.

**Plugin gating**: a plugin opts in by setting `required_feature_flag = "<flag>"`
(e.g. on `LocalExecutor`/`K8sExecutor`) and self-gates via `is_multi_extension` —
SAF only exposes a multi-extension plugin through `get_extensions` when that hook
returns True, so a gated plugin stays in the plugin registry but is absent from
`list_executors()`, tab-completion, and resolution. `core.bootstrap` stays a pure
discovery loop (no config reads); the gate resolves config itself via
`features.feature_gate_enabled` at registration time (env overrides
short-circuit before any file read). The decision is frozen per-process, which
is fine for the one-shot CLI. An explicitly-requested but gated/unknown executor
raises `ExecutorUnavailableError` (never a silent docker fallback); teardown of a
job whose executor was later disabled will also fail — the accepted cost of
relying on an experimental, opt-in feature.

Manage via `sparkrun setup features {list,enable,disable,reset}` (advanced, under
`setup`). Example `config.yaml`:

```yaml
features:
  channel: alpha          # optional; defaults to self_update.channel
  executor.k8s: true      # explicit per-flag override (beats channel default)
  executor.local: true
```

Tests: the `isolate_stateful` conftest fixture force-enables the two executor
flags via env so the legacy executor suite (which predates gating) keeps
passing. `tests/test_features.py` unit-tests the gate directly
(`K8sExecutor().is_multi_extension(...)`) and exercises exclusion end-to-end in a
clean subprocess (SAF exposes `is_multi_extension` once at registration and the
registry is process-global, so a plugin can't be re-hidden mid-process).

### Testing Patterns

Tests use pytest with `pytest-asyncio`. The `conftest.py` provides an `isolate_stateful` autouse fixture that redirects
SAF's stateful root to `tmp_path`, preventing tests from touching `~/.config/sparkrun/`. The bootstrap singleton (
`_variables`) is reset between tests. All core module imports in tests use `sparkrun.core.*` paths (e.g.,
`from sparkrun.core.registry import RegistryManager`).

All SSH/Docker operations in tests are mocked — no real hosts are needed. Common fixtures: `tmp_recipe_dir` (creates
sample v1/v2 recipes), `cluster_dir`, `hosts_file`, `v` (initialized SAF Variables instance).

Test files cover: benchmarking, bootstrap, CLI commands, CLI recipe integration, cluster manager, config, distribution,
Docker command generation, GGUF handling, host resolution, InfiniBand, networking, orchestration primitives, recipes,
registry (including manifest discovery, fallback merging, shared clones, and reserved name enforcement), runtimes,
embedded scripts, SSH execution, kernel tuning, and VRAM estimation.

### Companion Packages

- **`sparkrun-cc-plugin/`** — Claude Code plugin providing slash commands (`/sparkrun:run`, `/sparkrun:stop`, `/sparkrun:status`, `/sparkrun:list`, `/sparkrun:setup`) and skills for AI-assisted inference management (`run`, `setup`, `registry`).
- **`website/`** — Documentation site built with Astro (Starlight theme), deployed to Cloudflare Pages.

## Key Dependencies

- **`scitrera-app-framework`** (SAF) — Plugin system, lifecycle, variables/config management
- **`vpd`** — YAML reading (`read_yaml`) and command template placeholder substitution (`arg_substitute`)
- **`click`** — CLI framework
- **`huggingface_hub`** — Model downloading (`snapshot_download`)
- **`pyyaml`** — YAML parsing for recipes, clusters, registries
