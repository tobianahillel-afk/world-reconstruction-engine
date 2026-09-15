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
- more than one independently failing heavy external solver/model integration.

Prefer creating another lot over adding deep administrative nesting. Lots are capability-sized review groups; work items are run-sized implementation units.

## 5. Contract-first sequencing

For every new capability family, use this order unless an ADR justifies otherwise:

1. define/extend the solver-independent WRE contract;
2. add deterministic codec/persistence support when needed;
3. add the smallest baseline or compatibility adapter;
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

## 7. Compatibility-first migration

Existing tested components are assets, not obstacles. When a new architecture supersedes an older contract:

1. inventory the existing code/tests/data it affects;
2. classify each part as `retain`, `generalize`, `wrap_compatibly`, `deprecate`, or `remove`;
3. preserve existing behavior behind compatibility adapters where practical;
4. add migration tests before deleting old representations;
5. remove code only when the replacement has objective evidence and no required consumer remains.

No rewrite is authorized merely because a class/module name no longer matches the newest blueprint vocabulary.

## 8. External-adapter boundary

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
