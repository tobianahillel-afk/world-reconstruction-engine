# System invariants

This document collects cross-cutting invariants that every WRE subsystem must preserve. These rules are product semantics, not implementation suggestions.

## Evidence and provenance

1. Raw observations are immutable evidence.
2. Derived evidence does not overwrite its inputs.
3. Estimated geometry is not automatically accepted scene truth.
4. `OBSERVED_RECONSTRUCTED`, `INFERRED` and `GENERATED` provenance classes remain distinguishable through master and runtime products.
5. Manual corrections are versioned inputs with author/source context, never hidden mutable state.
6. Every expensive artifact is traceable to exact input artifacts, producer adapter/model/checkpoint, normalized configuration and producer version.
7. Reusing an artifact key with different content is an error, not an update.

## Identity and grouping

1. Observation identity, physical camera identity, scene identity, dynamic-entity identity and temporal-state identity are different domains.
2. Byte duplicates do not automatically collapse observation provenance.
3. Visual similarity proposes scene relationships; it does not prove scene identity.
4. Repeated/symmetric structures require stronger evidence before scene/fragment fusion.
5. A local coordinate frame does not imply metric scale, Earth alignment or compatibility with another local frame.
6. Merge and grouping decisions that can destroy alternatives must be reversible or preserve the competing hypothesis.

## Geometry and surface

1. Camera pose, depth, point maps, sparse points, dense geometry and explicit surface are separate products even when one solver emits several of them.
2. Geometry is evaluated independently from appearance quality.
3. Collision/measurement/navigation surfaces must come from an explicit physical representation suitable for that use.
4. Photorealistic appearance must not be substituted for physical geometry merely because it renders well.
5. Scale status and coordinate frame are explicit properties; unresolved scale is a valid state.
6. Coordinate-system conversions are centralized, explicit and tested.
7. Absolute anchors enter as uncertain provenance-bearing constraints; they do not destructively rewrite local geometry.

## Appearance, materials and environment

1. `AppearanceModel`, `SurfaceModel`, `MaterialModel` and `EnvironmentModel` have distinct ownership.
2. Sky/horizon/distant illumination should not be encoded as fake nearby solid geometry when an environment representation is appropriate.
3. Exposure, white balance, transfer functions and working color spaces are explicit artifacts/metadata when they affect appearance reconstruction.
4. Held-out render quality does not prove geometric fidelity.
5. Material/inverse-rendering output is inferred unless directly measured by a supported sensor/evidence contract.

## Time

1. Continuous event time and long-term chronological time are distinct regimes.
2. Historical structural change is represented as state/event change by default, not forced into one continuous deformation field.
3. Publication/upload time is not silently treated as capture time.
4. A later accepted state never destructively overwrites a retained earlier state.
5. Apparent change caused by registration error, occlusion or uncertainty must not be promoted to physical change without the relevant quality gate.
6. Dynamic entities must not contaminate stable scene geometry merely because they appear in source observations.
7. Object persistence through occlusion is an inferred state with uncertainty unless directly observed continuously.

## Routing

1. Routing chooses work, not truth.
2. `PREVIEW`, `FAST`, `QUALITY` and `MASTER` are compute/product modes, not provenance classes.
3. The router acts only on declared data-profile signals, hardware/resource budgets, existing artifacts and quality/failure outcomes.
4. A fallback must be explicitly registered for the triggering profile/failure class; arbitrary try-everything behavior is not allowed.
5. Escalation terminates at a bounded policy and may return `UNRESOLVED`.
6. New scientific models enter as candidates behind existing capability contracts unless they introduce a genuinely new responsibility.
7. Default changes require benchmark evidence appropriate to the affected profile and quality mode.

## Artifact graph and caching

1. Master artifacts form a versioned DAG; hidden mutable dependencies are prohibited.
2. Cache reuse is based on normalized, content-addressed identity rather than filenames or timestamps alone.
3. A configuration/model/checkpoint change invalidates only dependent artifacts, not unrelated project state.
4. Heavy jobs expose checkpoint/resume where the upstream implementation supports it safely.
5. Resume compatibility is explicit; a checkpoint produced by incompatible model/configuration state is rejected.
6. Source media remain independently addressable even after derived artifacts are deleted/rebuilt.

## External dependencies

1. Every production dependency/model/checkpoint has explicit source, version, license notes, integration mode and shipping status.
2. Floating `latest` versions/checkpoints are prohibited in reproducible production paths.
3. Code license does not imply checkpoint/model/dataset redistribution permission.
4. Native tools/parsers handling untrusted media are invoked without shell-string construction from user-controlled paths.
5. Resource limits and failure handling are explicit at untrusted/native boundaries.

## Runtime

1. `MasterScene`, exchange formats and `RuntimeScene` are different representations.
2. Runtime compilation may simplify/compress but preserves a link to the exact master-scene version.
3. Geometry LOD, appearance LOD and temporal LOD may differ but obey one target resource budget.
4. Collision/nav assets may be coarser than visual assets but remain linked to explicit physical geometry.
5. Runtime optimization never mutates or back-propagates into the master reconstruction silently.
6. Streaming/LOD behavior is evaluated with representative memory, loading and frame-performance metrics.

## Human review

1. AUTO, ASSISTED and EXPERT modes share the same underlying artifact/provenance semantics.
2. Expert overrides are explicit versioned inputs, not hidden flags inside adapter state.
3. Human approval cannot relabel generated content as observed reconstruction.
4. Comparing competing solver outputs does not delete losing evidence unless an explicit retention policy allows cleanup after decision/provenance recording.

## Fail-closed rules

When a required condition is unknown or unsupported, WRE prefers an explicit unresolved state over permissive inference.

Examples:

- unsupported CRS -> unresolved anchor, not guessed WGS84;
- unknown timezone -> local ambiguous time, not fabricated UTC;
- unsupported camera model -> route failure/specialist escalation, not pinhole coercion;
- weak scene identity -> separate clusters/hypotheses, not merge;
- weak geometry -> hole/unresolved region, not fake collision surface;
- missing historical date -> uncertain interval/group, not a fabricated epoch;
- model/checkpoint not approved -> unavailable route, not silent download/use.

These fail-closed rules are expected to reduce code as well as risk: the system models the valid domain directly instead of supporting arbitrary combinations and repairing them afterward.
