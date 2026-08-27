# Changelog

All notable changes to sparkrun are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
follows semantic versioning.

For the long-form 0.3.0 narrative, see [`docs/RELEASE_NOTES.md`](docs/RELEASE_NOTES.md).

## [Unreleased]

### Security

- Registry names and asset subpaths are now contained to the registry cache.
  Both arrive from `.sparkrun/registry.yaml` manifests in **remote**
  repositories (`sparkrun registry add <url>`, bootstrap discovery) and both
  become real filesystem paths, with no charset validation anywhere before this:
  `_cache_dir` is `cache_root / name`, so an escaping name resolved outside the
  cache root and `_link_registry_to_shared` would then `shutil.rmtree` it (a
  delete primitive), while `asset_dir` is `_cache_dir(name) / subpath`, so an
  escaping subpath had `iter_asset_files` `rglob` a directory outside the clone
  and `find_recipe` offer whatever YAML it found there as a runnable recipe (a
  read primitive feeding the recipe loader). New
  `assert_safe_registry_name` / `assert_safe_registry_subpath` /
  `assert_safe_registry_entry` enforce this at every entry point: `add_registry`
  and `validate_registry_name` raise, `_discover_manifest_entries` drops the
  offending entry and keeps the rest (raising only when nothing survives, so a
  wholly hostile manifest is never reported as a successful no-op add), and
  `_load_registries_from_file` skips-with-warning — narrower than the enclosing
  `except`, which reverts to the shipped defaults and would let one hand-edited
  entry discard every registry the user has. Requiring each path component to
  start alphanumeric also rules out `.`/`..`, dotfiles, a leading `-` (git would
  read it as an option) and the `_url_<hash>` shared-clone prefix, whose
  collision would have deleted the checkout every registry on that URL shares.
  See `docs/SECURITY.md`.

### Changed

- The `atlas` registry now points at `Atlas-Inf/sparkrun-recipes`; the recipes
  moved there from `Avarok-Cybersecurity/atlas-recipes` along with the project
  itself (source now at <https://github.com/Atlas-Inf/atlas>). Existing installs
  follow the move via `MIGRATED_REGISTRY_URLS`, which rewrites the URL in place
  and keeps each user's `enabled`/`visible`/trust state — the stale clone is
  re-fetched by `_drop_cache_if_url_changed` on the next sync. The layout is
  identical on both sides (a `recipes` subpath), so nothing else changes.
  Correspondingly, `atlas-inf` replaces `avarok-cybersecurity` in
  `EXTERNAL_RESERVED_NAMES`: the reserved `atlas` registry name now belongs to
  the org that publishes the recipes, and a registry claiming it from the former
  org is rejected like any other impersonation.
- The default Atlas container image is now `azeezish/atlas-gb10:latest`
  (previously `avarok/atlas-gb10:latest`), matching what every recipe in the
  new Atlas-Inf registry pins. Only affects `atlas` recipes that do not set
  `container:` themselves.

- Manifest discovery clones blob-filtered and sparse (`--filter=blob:none
  --sparse` + `sparse-checkout set .sparkrun`) instead of pulling the whole
  repository: only `.sparkrun/registry.yaml` is ever read, so the recipe trees
  were wasted transfer on every bootstrap URL. A failed sparse-checkout raises
  rather than falling through to "No `.sparkrun/registry.yaml` manifest found",
  so a clone whose manifest directory was never materialized is not mistaken for
  a repo that declares nothing.

### Added

- `run-recipe.sh` shim: `-v/--volume LOCAL:CONTAINER` (repeatable), matching
  spark-vllm-docker upstream. It maps to `--executor-args "-v ..."`, which the
  docker executor shlex-splits back into the `docker run` argv. As upstream, it
  applies to both solo and multi-node runs (unlike `-p/--publish`, solo-only).
- `tests/test_run_recipe_shim.py` — argv-mapping coverage for the shim, driven
  through its `RUN_RECIPE_DEBUG=1` hook.
- eugr builder: support for `build-and-copy.sh`'s `--exp-b12x` /
  `--experimental-b12x` preset. As upstream, it does not set
  `CUSTOM_BUILD_REQUESTED`, so on its own it only changes *which* prebuilt image
  is pulled: a nightly `:latest` sentinel (or a missing non-pullable eugr image)
  resolves to `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-b12x:latest` — sparkrun's
  mirror of upstream's `eugr/spark-vllm-b12x:latest` — instead of the standard
  nightly. Naming a b12x prebuilt image selects the variant the same way. With a
  custom build flag alongside it the build runs under the `sparkrun-eugr-vllm-b12x`
  local tag (upstream tags `vllm-node-b12x`) and the flag is forwarded verbatim.
  Long-term image pinning resolves against the `nightly-b12x` variant.

### Added

- `sparkrun tune vllm` / `sparkrun tune sglang`: `--timeout SECONDS`, a per-TP
  ceiling on the tuning job itself. `0` (the default) means no ceiling.

### Changed

- Tuning jobs are **unbounded by default**, replacing the fixed 8-hour cap on
  both the vLLM and SGLang paths. A tuning run that is killed at the wire loses
  all of its work, and the normal runtime is measured in hours (vllm-tune's MoE
  phase alone is 1.5-3h), so completion now outranks bounded runtime. Pass
  `--timeout` to get a ceiling back.
- `run-recipe.sh` shim: `--ray`/`--no-ray` combined with `--solo` now warn and
  are ignored instead of erroring, matching upstream's
  `use_ray = args.ray and not is_solo`. The flags are recorded during parsing
  and resolved afterwards, so a trailing `--solo` suppresses them too.
- `run-recipe.sh` shim: `--earlyoom` / `--earlyoom-args` are now rejected with
  the standard "not supported" pointer rather than a bare "unknown option"
  error. sparkrun runs the server as the container foreground process, so there
  is no earlyoom supervisor to substitute.

### Fixed

- `sparkrun logs` no longer deletes a **running** workload's job metadata. The
  log sources were named from the *caller's* host list rather than the
  workload's own placement, so a solo job on one host of a multi-host cluster
  was read as `{cid}_node_0` / `{cid}_head` instead of `{cid}_solo`. The
  liveness precheck then found that container absent, `describe_terminated`
  confirmed it had never existed, and the "confirmed gone → clean up stale
  metadata" branch removed the entry for a deployment that was serving traffic.
  Since the runtime is recorded only in that metadata, every subsequent
  `sparkrun logs <cluster_id>` failed with "No job metadata (or no runtime
  recorded)…" and `sparkrun status` lost the recipe label — with no way back
  short of relaunching. Fixed in three places, each independently sufficient to
  prevent the data loss:
  - The sources are named from the workload's placement: the live status
    snapshot when it shows a lone `_solo` container, else the hosts recorded at
    launch (for the recipe form, where the caller named no hosts to read), else
    the caller's list as before.
  - A container name the snapshot does not report, on a host where the workload
    *is* running, is now inconclusive rather than "confirmed absent" — it means
    sparkrun named the container wrongly, not that the workload stopped.
  - The metadata removal is gated on the whole snapshot: a cluster_id reported
    running anywhere keeps its entry, whatever the precheck concluded about the
    sources.
- `sparkrun logs <cluster_id>` now falls back to the `sparkrun.runtime`
  container label (surfaced by `query_status` as
  `RunningWorkload.runtime_name`) when the job-metadata cache cannot name the
  runtime. A live workload launched from another control machine — or one whose
  cache entry was lost — is readable again instead of being unaddressable by id.
  This also restores useful `logs`/`stop` tab completion for those workloads,
  which is populated from the same cache.
- `logs`/`stop` completion now describes a running workload that has no cached
  job metadata, instead of offering a bare hex cluster_id. The status sweep
  already recovers recipe and runtime from the `sparkrun.recipe` /
  `sparkrun.runtime` container labels, so `running.json` carries them under a
  new `workloads` key and completion renders `recipe runtime on host` as the
  item's help — on zsh and fish, which display it; bash discards help text, so
  the value still has to stand alone there. The key is additive and read
  leniently: a snapshot written by an earlier sparkrun (or one whose
  `workloads` block is unreadable) keeps its running set and simply has no
  descriptions, because discarding the set over a missing annotation would put
  every dead job back in the list. A recipe ref that is a path to a YAML file
  is annotated by its stem: the ref records whatever the launch was handed, so
  a job started from a laptop carries that laptop's absolute path, and the
  directory layout is not what identifies the recipe. Bare names and
  `@registry/name` refs are left alone — both are values a user can type, and
  rewriting them in the annotation would misrepresent what to type.

### Changed

- `load_running_snapshot` returns a `RunningSnapshot` dataclass
  (`running` / `covered` / `workloads`) rather than a
  `(cluster_ids, hosts_covered)` tuple. Adding the third field to a tuple would
  have silently broken every `running, covered = …` destructure, and two
  positional frozensets are easy to transpose. `save_running_snapshot` grows a
  keyword-only `workloads=` argument; its positional signature is unchanged.
  New `ObservedWorkload` record and `observe_workloads()` fold, shared by
  `api.status` and completion's own sweep so the two cannot describe the same
  workload differently.

## [0.3.0] — 2026-07-30

The largest release since 0.1: multiplatform foundations, a console-free
library API, pluggable placement, and a security pass across recipe trust.
The shipped DGX Spark path is preserved — NCCL output, container launch, and
the post-launch lifecycle are unchanged for existing recipes.

### Highlights

- **`sparkrun.api`** — a console-free public library API. It never writes to
  stdout/stderr, never calls `sys.exit()`, and does not import `sparkrun.cli`.
  The CLI is now a renderer over it, so anything the CLI does is reachable
  from Python.
- **Pluggable schedulers.** `occupancy-sparse` (the new default for new
  clusters) spreads workloads onto least-loaded hosts using live cluster
  occupancy; `occupancy-dense` bin-packs; `greedy` is the 0.2.x behavior.
  Clusters with no `scheduler` key keep `greedy`, so upgrading never silently
  moves existing workloads.
- **Unified status.** All "what's running where?" flows through `api.status`,
  which sweeps every enabled executor on a cluster's substrate and merges. A
  native `local` workload and a docker container are invisible to each other's
  introspection; both now appear in `status`, `cluster monitor`, and
  `stop --all`.
- **Multiplatform seams.** `HostHardware` / `AcceleratorSpec` /
  `CollectiveBackend` / `HardwarePlatformPlugin` are the abstractions future
  vendor support builds against. NVIDIA remains the only fully wired vendor;
  AMD (RCCL) and Intel Gaudi (HCCL) ship as scaffolds.
- **Feature flags.** Channel-aware gating (`stable` / `beta` / `alpha`) for
  experimental capabilities, managed with `sparkrun setup features`.
- **Security.** URL-sourced recipes are never auto-trusted, recipe `env` is no
  longer expanded, and container-escape surfaces (bind-mounts, privileged
  `executor_config` keys, executor selection) are trust-gated.

### Added

#### Library API

- `sparkrun.api`: `run`, `stop`, `stop_all`, `logs`, `status`, `status_report`,
  `schedule`, `list_jobs`, `search_recipes`, `resolve_recipe_filter`,
  `benchmark`, `resume_benchmark`, `live_monitor` / `open_live_monitor` /
  `open_telemetry`.
- Sub-namespaces `api.proxy`, `api.setup`, `api.tailscale`, `api.k8s`, each
  with `_ops.py` + `_errors.py` + a re-exporting `__init__.py`.
- Typed error hierarchy under `SparkrunError`: `InsufficientCapacity`,
  `LayoutRequired`, `RecipeNotFound`, `InvalidRegistryFilter`,
  `HostsUnreachable`, `JobNotFound`, `AmbiguousWorkload`, `TrustRejected`,
  plus the benchmark family. Several carry structured detail (e.g.
  `InsufficientCapacity.status` / `.host_list` / `.required`).
- `SparkrunContext` (`sctx`) threaded through every entry point so chained
  calls share config, registry manager, and cluster manager.
- `api/_hosts.py:resolve_effective_hosts` — the single placement authority,
  used by `api.run`, the benchmark flow, and the CLI.

#### Scheduling and placement

- `schedulers/` package: `greedy`, `occupancy-sparse`, `occupancy-dense`,
  discovered via SAF (`EXT_SCHEDULER`).
- `--scheduler` on `run` and `benchmark`; `scheduler:` on recipes; `--scheduler`
  on `cluster create` / `cluster update`.
- `core/limits.py` — per-accelerator usable-memory cap for scheduling and fit
  (`--max-gpu-mem-util`, cluster `accelerator_memory_limits`, platform default;
  GB10 → 0.85). Distinct from the serving `--gpu-memory-utilization`.
- `core/layout.py` — recipe `layout:` block pinning ranks to hosts and local
  accelerator indices, honored verbatim by every scheduler and **required** on
  multi-vendor clusters. Experimental.
- `core/cluster_status.py:ClusterStatus` — data-only occupancy snapshot.

#### Multiplatform

- `core/hardware.py`, `core/hardware_probe.py` (combined accelerator + IB probe
  in one SSH round-trip), `core/backend_select.py`, `core/placement.py`.
- `orchestration/collectives/`: `CollectiveBackend` ABC, `nccl` (default,
  byte-identical to legacy DGX output), `rccl` / `hccl` scaffolds.
- `platforms/`: `HardwarePlatformPlugin` + `dgx_spark`, `nvidia_generic`.
  Order-sensitive in-process registry (most-specific `matches()` wins).
- `runtimes/compatibility.py` — `requires_capability` gate, raising
  `IncompatibleHardwareError` before any side effects.

#### Executors, transports, plugins

- `orchestration/executor.py` facade + `orchestration/executors/`:
  `docker` (default), `local` (experimental), `k8s` (experimental draft).
- `resolve_executor()` — one layered chain (CLI → recipe → cluster → runtime →
  config → baseline). Recipe `executor:` / `executor_config:`;
  `SparkrunConfig.default_executor` / `executor_config`; `cluster create
  --executor` / `--executor-opt`.
- `transports/` — connectivity seam (`Transport`, `EXT_TRANSPORT`), `ssh`
  default with a no-op `prepare()`, plus `cleanup_cluster` for teardown.
- `core/external_plugins.py` — out-of-tree plugin loading from `plugins.paths`,
  scanning the SAF plugin bases and calling an optional `register(v)` hook.
- `cli/ext.py` — `register_cli_command()` + `PluggableGroup`, letting plugins
  attach Click commands to the built-in tree.
- `orchestration/telemetry/` — `TelemetryProvider` per status scope, powering
  `api.live_monitor`.

#### Feature flags and telemetry

- `core/features.py` — `FeatureFlag` registry with per-channel defaults;
  resolution order env → config → channel → baseline → fail-closed.
  `SPARKRUN_FEATURE_<NAME>` env override.
- `sparkrun setup features list|enable|disable|reset`.
- Plugin self-gating via `required_feature_flag`.
- Anonymous, opt-out telemetry (`telemetry/`), `sparkrun setup telemetry`,
  `SPARKRUN_NO_TELEMETRY` per-process override.

#### CLI

- `sparkrun setup check` — non-destructive readiness probe (docker, CDI,
  earlyoom, sudoers, SSH mesh, CX7), with `--json`.
- `sparkrun setup tailscale` (gated) — join hosts to a tailnet and publish an
  inference endpoint, via JIT-minted tagged OAuth keys.
- `sparkrun setup k8s` (gated) — kubectl acquisition, cluster info, least-priv
  service account, Kueue install, JobSet launch.
- `sparkrun cluster import svd|eugr`.
- `sparkrun registry trust|untrust`, `registry add --trust`.
- `sparkrun recipe update`; `--json` on recipe list/search and status.
- `@registry` scope accepted in `list` / `search` queries.
- `logs`: `-f/--follow`, `-a/--all-sources`, `-n/--lines` (default: all).
- `run`: hidden `--scheduler`, `--rebuild`, `--executor-args`, `--dp`.
- `benchmark` categories (`performance` / `perf`, `tools`), `--resume`,
  `--arena`, `--api-key-env`.
- `run-recipe.sh` compatibility shim for the spark-vllm-docker workflow.

#### Runtimes and builders

- `tokenary` runtime — native multi-node TP over its own NCCL bootstrap;
  embedding-only and NER workloads.
- `modular-max` runtime — single-node; `tensor_parallel` maps to local
  `--devices` rather than to host count.
- Recipe-settable `command_binary` for self-dispatching images.
- `--rebuild` / `builder_config.rebuild`.

#### Other

- Absolute-path `model:` plus `Executor.verify_mount_sources()` — pre-placed
  weights are verified on the *targets* before the launch commits to skipping
  download and distribution.
- Cluster-level `env` + `env_file` with `${VAR}` references resolved at launch
  from that file only, plus `sync_source` import provenance.
- `ssh.max_parallel_ssh` (default 20) capping every parallel SSH/rsync fan-out.
- NVIDIA CDI spec generation in the wizard; GPUs requested via CDI rather than
  `--gpus`.
- SSH access bootstrap in the wizard (`api/setup/`), written to run on a bare
  Windows control machine — no local `bash`, no paramiko, key material over
  stdin rather than argv.

### Changed

- **Status discovery** is cluster-scoped and cross-executor
  (`query_status_for_cluster`); `Executor.status_scope` names the substrate.
  `stop --all` and the proxy's endpoint discovery use the same source.
- **Teardown reports honestly.** `docker rm -f … || true` made stop exit codes
  meaningless; teardown now verifies and counts what it removed.
- **The proxy applies changes by config regeneration + managed restart.**
  LiteLLM's `/model/new` / `/model/delete` need a DB-backed model store; the
  management API is used read-only. The restart is skipped when the desired
  model set already matches disk. Gateway selection added
  (`gateway.litellm` flag, `proxy.gateway` pin) behind `api/proxy`.
- **Registry assets are data, not four code paths.** Recipes, benchmark
  profiles, tuning configs, and mods share `RegistryAsset` +
  `find_asset_in_registries` / `iter_asset_files`, so listing and lookup can
  never disagree. Flat beats nested *per registry*; `.yaml` beats a same-stem
  `.yml` *per directory*.
- **All built-in default registries ship trusted**, including `eugr` and
  `atlas`. The legacy-config trust migration derives from
  `FALLBACK_DEFAULT_REGISTRIES` rather than `BOOTSTRAP_REGISTRY_URLS`, so a
  newly-trusted default reaches upgrading users and not just fresh installs.
- **The `eugr` builder is pull-first**, following upstream eugr's July 2026
  move to published images. Sentinel `:latest` refs resolve to the
  authoritative nightly and are pulled; `build-and-copy.sh` runs only when
  `build_args` request a wheels or custom build.
- `RuntimePlugin.run()` gains a keyword-only `backends:` map. Native
  multi-node runtimes emit rank-specific argv via
  `RuntimePlugin._make_node_command_args`.
- vLLM command generation centralized in `runtimes/_vllm_mixin.py:VllmMixin`.
- Recipe log locations are data (`core/log_source.py`), so `sparkrun logs`
  finds output that a plain `docker logs` would miss for
  sleep-infinity + `docker exec` workloads.
- Thunder Compute transport moved out of core into an out-of-tree plugin.
- Contributions target `develop-next` (see `DEVELOPERS.md`).

### Fixed

- **Windows control machines**: scripts piped to a remote shell were
  CRLF-mangled; job metadata and proxy configs were never written; registry
  shared-clone symlinks failed without elevation.
- `auto_port` no longer moves a workload's identity, so `stop` / `logs` still
  resolve it.
- A relaunch evicts the deployment it told the scheduler it replaces.
- Occupancy schedulers no longer count the launching workload's own intent
  against its capacity.
- `cluster monitor`: a host that hangs before its first sample reconnects, and
  the last sample is dropped when a host disconnects.
- Recipe resolution never silently guesses between same-stem recipes;
  ambiguity errors list path-qualified candidates.
- Benchmark identity is tied to recipe content, so an edited recipe starts a
  new run rather than silently resuming state measured against other settings.
- Per-host comm env is pinned to the init network's verdict.
- `HF_HUB_CACHE` is set alongside `HF_HOME` so hf-hub clients find the mounted
  cache.
- The detached serve launches via `docker exec -d` rather than in-shell
  `nohup`.
- CX7 setup pins a coherent port pair on multi-port hosts.
- `-o distributed_executor_backend` overrides a literal command flag.
- v1 recipe brace escapes are collapsed; non-string defaults are guarded.

### Removed

- `resolve_ib_env()` — per-host communication env flows exclusively through
  `resolve_comm_env(ctx, comm_env, backends)`.
- `RuntimePlugin.executor` property and the hardcoded `_KNOWN_EXECUTORS` set.
- The deprecated `placement` module (migrated to `scheduler`).
- The Atlas single-node `validate_recipe` restriction. It was never load
  bearing — `recipe.validate()` issues surface as warnings — so it only
  printed a false claim on working multi-node launches.

### Deprecated

- Recipe topology fields `mode`, `solo_only`, `cluster_only` — use `min_nodes`
  / `max_nodes`.
- The `eugr-vllm` runtime, superseded by the `eugr` builder.

### Security

- **URL-sourced recipes are never auto-trusted.** Recipes fetched from a URL
  (including `@spark-arena/<uuid>` links) previously carried no
  `source_registry` and were treated as local, so their hooks ran with no
  confirmation — a "run this link" code-execution path.
- **Recipe fetch hardening**: https-only, host allowlist, redirect
  re-validation, response size cap.
- **Container-escape surfaces are trust-gated**
  (`_enforce_recipe_mount_trust`): host bind-mounts, the `executor_config`
  privilege keys (`privileged`, `cap_add`, `security_opt`, `devices`, `user`,
  `volumes`), and executor selection (untrusted → `docker` only). A denylist
  refuses the host root, the docker socket, SSH keys, and kernel
  pseudo-filesystems regardless of trust.
- **Recipe `env` is no longer expanded**, preventing a third-party recipe from
  exfiltrating control-machine secrets into a container it controls.
- **Telemetry sends a model identifier only for a confirmed-public HF repo.**
  Everything else becomes a coarse placeholder that records why it was
  withheld — `<hf-private>`, `<local-path>`, `<unknown-visibility>` — so a
  private repo id or a path to local weights never leaves the machine. The
  verdict reads the Hub's `private` / `gated` flags rather than inferring from
  whether a fetch succeeded, since an ambient `HF_TOKEN` resolves a user's own
  private repos. It fails closed: offline, rate-limited, and skipped-detection
  all yield `<unknown-visibility>`. Opted-out users never trigger the lookup —
  the enablement check moved ahead of event construction in
  `emit_run_telemetry` / `emit_benchmark_telemetry`.
- **Trust is a per-registry local decision** stored in the user's
  `registries.yaml`; a repository manifest cannot grant itself trust.
- Proxy bind host is explicit and persisted; an unconfigured proxy still binds
  `0.0.0.0` for compatibility but warns loudly, escalating when no master key
  is set. Proxy state files are `0600` and refuse to follow symlinks.
- Arena OAuth callback bound to a per-flow `state` nonce; CORS allowlist
  restricted to `AUTH_PROXY_BASE`.
- `validate_unix_username()` and `validate_sudoers_path()` gate every sudoers
  interpolation.
- Token/key/password/secret env values masked in docker executor DEBUG output.
- `runtimes/trtllm.py` no longer relaxes SSH host-key checking.
- `_validate_git_url` allowlists `https://`, `git@`, `ssh://`, `file://`.
- `utils/shell.py:quote()` wraps `shlex.quote()`; in-tree shell construction
  routes through it.

[0.3.0]: https://github.com/spark-arena/sparkrun/releases/tag/v0.3.0
