# Engineering execution protocol

This document defines how the WRE blueprint is converted into safe, reviewable implementation work. Product documents define **what** WRE must become; this document defines **how** an agent is allowed to change the repository while getting there.

## 1. Execution objective

WRE is developed as a sequence of small, explicit capability increments. A work item is considered well-sized only when one coding agent can understand its boundary, implement it, test it and review the resulting diff in one development run without needing to reinterpret the product architecture.

The preferred unit is:

```text
one work item
  -> one coherent contract/capability increment
  -> one short-lived branch
  -> one PR
  -> deterministic acceptance evidence
  -> one explicit state handoff
```

If that is not realistic, the item is too large and must be split before implementation.

Wall-clock duration is not a correctness contract because external downloads, CI queues and heavy native/model integrations vary, but the **reasoning and code-change scope** must remain one-run sized. An item that routinely requires several independent implementation phases, several unrelated external integrations or several separate review domains is too large even if an agent could eventually finish it in one conversation.

## 2. Deny-by-default scope rule

Implementation scope is **allowlisted**, not open-ended.

For an active work item, an agent may implement only behavior required by:

1. the work item's objective;
2. its explicit acceptance criteria;
3. the minimum supporting code/tests/docs needed for those criteria;
4. an already-accepted architectural invariant that the item necessarily touches.

Everything else is out of scope even when it appears useful, elegant or adjacent.

The absence of an explicit prohibition is **not** permission.

Examples:

- a retrieval item may emit candidates; it may not silently add geometric acceptance;
- a geometry adapter may import a `GeometrySolution`; it may not silently create a `MasterScene`;
- an appearance adapter may emit an `AppearanceModel`; it may not redefine collision geometry;
- a runtime compiler item may simplify assets; it may not mutate master-scene truth;
- a generated-completion item may synthesize pixels/geometry only through the declared generated provenance contract; it may not relabel synthesis as observed reconstruction.

This rule minimizes code and prevents later negative checks from having to undo permissive behavior.

## 3. Positive-domain logic

Prefer representing valid states directly instead of accepting arbitrary states and rejecting combinations later.

Preferred patterns:

- enums or tagged unions for closed state sets;
- constructors that reject invalid state before persistence;
- immutable/versioned artifacts instead of mutable bags of optional fields;
- explicit capability interfaces instead of `dict[str, Any]` solver payloads in the core;
- explicit allowlists for supported models/formats/routes;
- no implicit fallback to a different semantic mode;
- no truthy/falsy interpretation where a distinct `UNKNOWN`, `UNRESOLVED` or `NOT_APPLICABLE` state matters;
- no last-write-wins for identity-bearing or evidence-bearing state unless the owning contract explicitly defines it.

Do not add speculative fields, generalized extension hooks or plugin surfaces before an accepted work item requires them.

## 4. Work-item sizing rules

A work item should normally satisfy all of the following:

- one primary responsibility;
- one primary code ownership boundary;
- one independently testable output contract;
- no more than one major external integration decision;
- no hidden dependency on an unimplemented later capability;
- no requirement to change unrelated product semantics;
- a reviewable diff whose correctness can be reasoned about locally.

Split an item when it combines any of these:

- contract definition **and** several independent production adapters;
- ingestion **and** reconstruction;
- candidate generation **and** truth acceptance;
- geometry **and** appearance ownership;
- master representation **and** runtime optimization;
- dynamic event reconstruction **and** long-term chronology;
- implementation **and** benchmark-based default promotion when the benchmark itself does not yet exist;
- more than one independently failing heavy external solver/model integration;
- two specialist families that can fail independently, such as HDR and low-light, or material fusion and inverse rendering;
- several independent editing/export targets that would require unrelated UI, package or round-trip logic;
- a final-validation item that would need to create missing product behavior instead of only running/assembling already-owned validation evidence.

### Pre-activation complexity gate

A planned item must be split **before** it becomes `ready` when its executable contract would require any of the following:

1. more than one major external repository/model/checkpoint integration;
2. more than one independently shippable product responsibility;
3. several unrelated persistence/runtime/UI surfaces whose failures can be reviewed independently;
4. a dependency/license review, adapter implementation, normalization contract and benchmark/default promotion that cannot be reasoned about as one coherent change;
5. a diff whose correctness cannot be reviewed locally without understanding several future lots;
6. a validation item that discovers missing coverage requiring new implementation.

For heavy research integrations, it is acceptable—and often preferable—to split dependency/checkpoint approval, adapter normalization, real integration fixture and benchmark/default promotion into separate work items.

Prefer creating another lot over adding deep administrative nesting. Lots are capability-sized review groups; work items are run-sized implementation units.

## 5. Contract-first sequencing

For every new capability family, use this order unless an ADR justifies otherwise:

1. define/extend the solver-independent WRE contract;
2. add deterministic codec/persistence support when needed;
3. add the smallest baseline or legacy-donor adapter that cleanly satisfies the WRE contract;
4. add focused fixtures/tests;
5. add alternative specialist adapter(s);
6. benchmark under the same contract;
7. promote a default through routing/registry policy;
8. add expensive optimization only after correctness is measurable.

Do not build a router before its candidate routes expose comparable contracts and metrics. Do not build a benchmark around solver-private outputs when the product contract is still undefined.

## 6. Reuse-before-implementation

Before writing an algorithm, inspect the dependency/technology registries and maintained upstream implementations. Foundational algorithms should normally be reused behind adapters.

Custom implementation requires evidence that the existing implementations fail the relevant WRE contract and, for foundational algorithms, an accepted ADR.

WRE should own:

- stable domain contracts;
- orchestration;
- normalization;
- provenance;
- artifact lifecycle;
- quality policy;
- routing/fallback policy;
- user-visible state;
- integration tests and benchmarks.

It should not duplicate mature solver internals merely to own more code.

## 7. Legacy-donor migration

V2 is authoritative. Existing tested V1 components are optional donors and regression evidence, not compatibility requirements.

When a V2 contract supersedes older code:

1. inventory the existing code/tests/data it affects;
2. classify each part as reusable donor, generalize, temporary wrapper, deprecate or remove;
3. reuse or temporarily wrap legacy behavior only when that is the cheapest safe way to satisfy the **V2** contract;
4. add migration/regression tests before deleting behavior that V2 still promises;
5. remove obsolete code when the replacement has objective evidence and no required consumer remains.

Do **not** preserve a legacy API, data model or sequencing assumption merely for backward compatibility. If a cleaner V2 implementation better satisfies the frozen architecture, V2 wins. Historical Git/PR/review evidence may remain even after the corresponding implementation is removed.

No rewrite is authorized merely because a class/module name no longer matches the newest blueprint vocabulary; replacement still needs a concrete V2 benefit and evidence.

## 8. Candidate freshness and external-adapter gate

Before activating a work item that integrates or promotes an external solver/model, refresh the candidate landscape rather than trusting a shortlist written months earlier.

The activation/review must check, as applicable:

- `docs/14_TECHNOLOGY_SELECTION.md` and `docs/27_RESEARCH_CANDIDATE_COVERAGE.md`;
- current upstream releases and maintained implementations;
- major recent conference/research candidates relevant to that responsibility;
- exact code version/commit and model/checkpoint identity;
- code, model and dataset licensing/redistribution constraints separately;
- hardware/runtime support and reproducibility;
- whether a newer major dependency release materially improves WRE's required capability;
- a stable baseline plus the strongest currently viable primary/specialist candidates.

A candidate-refresh / technology gate may conclude **NO-GO / deferred optional**. This is the correct outcome when no reviewed external candidate satisfies a mandatory product constraint such as licensing, reproducibility, supported baseline hardware or exact checkpoint availability. Do not force an integration merely because the roadmap item was originally named after an expected candidate.

For mandatory product capabilities, CPU-only portability is part of the baseline hardware contract. Accelerator-only candidates may remain registered benchmark/specialist work, but their absence must not block later mandatory work when an accepted CPU route already satisfies the capability contract. A later explicit work item may revisit the deferred accelerator candidate when hardware and evidence become available.

A dependency is **not** upgraded merely because a larger version number exists. Keeping an older pinned version requires a concrete reproducibility, platform, licensing or integration rationale when a materially newer viable release exists.

An adapter must declare at least:

- input WRE contract;
- output WRE contract;
- exact dependency/model/checkpoint identity;
- supported input profile;
- resource requirements;
- normalized configuration identity;
- provenance/artifact identity;
- failure taxonomy mapping;
- determinism/reproducibility guarantees;
- checkpoint/resume behavior when applicable;
- exposed metrics;
- license/shipping status.

Solver-private structures may exist inside the adapter but must not become the stable product API accidentally.

## 9. Failure behavior

Failure is data, not an excuse for hidden behavior.

An adapter or route must never silently:

- substitute a different model/quality mode;
- reduce a correctness threshold;
- invent metadata or timestamps;
- merge observations/scenes because they look similar;
- convert generated content into reconstructed content;
- discard an older temporal state because a newer one exists;
- retry forever or consume unbounded resources.

Failures map to stable categories and may cause `RETRY`, `ESCALATE`, `ACCEPT_WITH_WARNINGS` or `UNRESOLVED` only through explicit policy.

## 10. Quality modes versus acceptance

`PREVIEW`, `FAST`, `QUALITY` and `MASTER` define compute budget and richness of attempted processing. They do not redefine provenance or turn weak evidence into truth.

Different modes may:

- choose different adapters;
- use different resolutions/iteration budgets;
- skip optional refinement;
- run more competing specialists;
- compile different runtime assets.

They must still expose metrics and provenance appropriate to the product they claim to produce.

## 11. Testing obligation by change type

Every behavior change needs the cheapest reliable evidence that can catch its likely regression.

- domain/invariant change -> unit tests including invalid states;
- persistence/codec change -> round-trip, version and conflict tests;
- adapter change -> mocked/fixture contract tests plus real external integration in the appropriate lane;
- geometry change -> independent numeric/structural checks with tolerances;
- appearance change -> held-out-view metrics plus artifact/geometry consistency checks;
- dynamic change -> temporal consistency/tracking/object-persistence checks;
- chronology change -> unchanged/change/misregistration/date-ambiguity cases;
- runtime change -> correctness plus memory/FPS/loading/LOD measurements as appropriate;
- bug fix -> regression test reproducing the failure whenever practical.

Tests must not weaken production invariants merely to make a fixture convenient.

## 12. Review obligation

Every PR review asks:

- Did the diff implement only the allowlisted scope?
- Is a mature dependency being needlessly reimplemented?
- Did a solver-private assumption leak into the core?
- Are new states represented positively and explicitly?
- Are provenance and artifact identities preserved?
- Can failure remain unresolved rather than being forced?
- Are generated/inferred/reconstructed classes preserved?
- Are coordinate/color/time conventions explicit?
- Are tests checking the real risk, including negative cases?
- Did the item introduce future work that belongs in another PR?
- For external research integrations, was the candidate/dependency landscape refreshed and were license/checkpoint/hardware constraints rechecked?
- Is the item still small enough that its behavior can be reviewed as one coherent run-sized change?

Lot reviews verify cross-item composition. Milestone reviews verify an end-to-end user capability rather than merely counting completed work items.

## 13. Minimal-change rule

Prefer the smallest coherent change that makes the target contract true.

Do not:

- refactor neighboring modules for style;
- introduce generic abstractions with only one speculative caller;
- add optional configuration knobs without a required use case;
- broaden accepted input/state space unless the work item requires it;
- duplicate validation already enforced by a lower-level immutable contract unless crossing an untrusted boundary requires revalidation.

Do add a small reusable primitive when it removes duplicated correctness logic across **existing** callers.

## 14. State handoff

A work-item PR is not complete until repository metadata makes the next session unambiguous:

- active work item/status agree;
- dependencies are complete;
- component registry paths/status are current;
- dependency/model decisions are recorded;
- review evidence is recorded when a lot/milestone closes;
- known limitations are assigned to a future item or explicitly accepted;
- CI required by the item is green on the final handoff head.

The next agent should never need conversation history to infer what code it is allowed to write.
