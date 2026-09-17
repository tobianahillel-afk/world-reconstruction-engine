from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly once, found {count}")
    return text.replace(old, new, 1)


work_items = Path("registry/work-items/v2m0.yaml")
text = work_items.read_text(encoding="utf-8")
text = replace_once(
    text,
    "  - id: V2L4.5\n    lot: V2L4\n    title: Deterministic quality policy evaluator\n    status: ready\n",
    "  - id: V2L4.5\n    lot: V2L4\n    title: Deterministic quality policy evaluator\n    status: done\n",
    "V2L4.5 completion",
)
old = "  - {id: V2L4.6, lot: V2L4, title: Unknown metric and failure negative tests, status: planned, depends_on: [V2L4.5]}"
new = """  - id: V2L4.6
    lot: V2L4
    title: Unknown metric and failure negative tests
    status: ready
    objective: Close V2L4 by stress-testing the fail-closed composition of the stable failure taxonomy, typed metric vector, quality-decision vocabulary, immutable benchmark records and deterministic quality-policy evaluator; prove unknown missing unmapped malformed or contradictory inputs never silently PASS, record the V2L4 lot review, and prepare V2L5.1 without adding new quality or routing functionality.
    allowed_scope:
      - tests/test_quality_policy.py for focused cross-contract negative and precedence regressions
      - tests/test_failure_taxonomy.py tests/test_metrics.py tests/test_quality_decisions.py and tests/test_benchmarks.py only when a focused cross-contract regression belongs at the owning boundary
      - src/wre/domain/quality_policy.py only for a narrowly evidenced fail-closed defect exposed by an acceptance test in this work item
      - docs/reviews/V2L4-quality-failure-benchmark-review.md
      - registry/reviews.yaml
      - registry/components.yaml
      - PROJECT_STATE.yaml
      - this work-item registry
      - registry/work-items/v2m1.yaml only to expand V2L5.1 into a complete executable contract after V2L4 review evidence is green and without implementing V2L5.1
    out_of_scope:
      - new failure categories metric semantics QualityDecision values QualityMode values benchmark fields policy capabilities or scoring algorithms
      - raw adapter failure-signal parsing automatic failure inference metric computation normalization weighted scoring benchmark ranking or default promotion
      - retry fallback escalation routing scheduling checkpoint execution resource allocation persistence dashboards or registry runtime mutation
      - profiler NVTX Nsight TensorRT decode backend ANN storage-path or other execution-profile integration
      - broad refactors of V2L4.1 through V2L4.5 contracts when no acceptance regression demonstrates a fail-closed defect
      - adding new external dependencies models adapters or research candidates
      - implementing V2L5.1 media profiling in this work item
    depends_on: [V2L4.5]
    read_before:
      - docs/01_PRODUCT.md
      - docs/02_ARCHITECTURE.md
      - docs/15_PRODUCTION_RUNTIME.md
      - docs/23_ENGINEERING_EXECUTION.md
      - docs/24_SYSTEM_INVARIANTS.md
      - docs/25_V2_ROADMAP.md
      - docs/28_PERFORMANCE_OPTIMIZATION_PLAYBOOK.md
      - docs/29_PERFORMANCE_INTEGRATION_MAP.md
      - registry/reviews.yaml
      - registry/components.yaml
      - src/wre/domain/failures.py
      - src/wre/domain/metrics.py
      - src/wre/domain/quality.py
      - src/wre/domain/benchmarks.py
      - src/wre/domain/quality_policy.py
      - tests/test_failure_taxonomy.py
      - tests/test_metrics.py
      - tests/test_quality_decisions.py
      - tests/test_benchmarks.py
      - tests/test_quality_policy.py
    reuse:
      - the exact stable FailureCategory MetricVector QualityDecision QualityMode BenchmarkRecord and QualityPolicy contracts already implemented by V2L4.1 through V2L4.5
      - the existing repository validator fast CI CodeQL and native COLMAP integration evidence model
      - the established lot-review format and registry review gate used by V2L0 through V2L3
      - fail-closed unknown-input and bounded-escalation invariants from docs/24_SYSTEM_INVARIANTS.md
    acceptance:
      - Every valid but policy-unknown MetricName supplied in MetricVector yields QualityDecision.UNRESOLVED with UNKNOWN_METRIC and can never silently PASS or be dropped from evaluation.
      - Every stable FailureCategory lacking an exact selected-policy mapping yields UNRESOLVED with UNMAPPED_FAILURE and can never silently PASS or be inferred into another category.
      - Missing required metrics direction mismatches unknown metrics and unmapped failures retain UNRESOLVED precedence over mapped ESCALATE RETRY or ACCEPT_WITH_WARNINGS conditions in mixed evidence.
      - Duplicate unsorted mutable raw-string or otherwise untyped policy rules failure inputs and metric-vector inputs continue to fail closed rather than being sorted deduplicated coerced or ignored silently.
      - Unknown QualityDecision tokens unknown FailureCategory tokens and malformed metric identities remain rejected by their owning closed or validated contracts; no generic unknown-to-success coercion exists across the V2L4 composition.
      - Optional absent metrics and explicitly admitted informational metrics remain deterministic valid cases and do not create false negative failures merely to make the negative suite pass.
      - BenchmarkRecord remains descriptive evidence only and still contains no quality decision threshold policy route winner rank default-promotion or execution behavior.
      - Adapter-local failure_signals and metric_names remain distinct declarations and are not automatically mapped into stable FailureCategory or MetricVector semantics by V2L4.
      - The deterministic evaluator remains pure non-mutating and side-effect free and returned RETRY or ESCALATE values still execute no routing fallback scheduler or retry action.
      - A V2L4 lot review audits V2L4.1 through V2L4.6 together for composition boundaries regressions exact CI evidence and absence of premature routing/default-selection behavior; registry/reviews.yaml may mark V2L4 passed only when that evidence supports PASS.
      - After a PASS lot review V2L5.1 is expanded into a complete one-run deny-by-default MediaProfile contract and becomes the sole active ready item without implementing it.
    tests:
      - parameterized unknown-policy MetricName cases proving UNRESOLVED and typed UNKNOWN_METRIC reasons
      - parameterized unmapped stable FailureCategory cases covering the complete current FailureCategory vocabulary
      - mixed missing unknown direction-mismatch mapped-failure and unmapped-failure precedence regressions
      - duplicate unsorted mutable raw-string and wrong-type negative regressions across QualityPolicy and evaluator inputs
      - closed-vocabulary regressions for QualityDecision and FailureCategory plus malformed MetricName rejection
      - optional and informational metric regressions proving valid absence/presence remains PASS when no failure condition exists
      - BenchmarkRecord and adapter-registry separation regressions proving no policy routing or automatic mapping leakage
      - repository validator and full fast CI
      - native COLMAP integration regression
      - CodeQL
      - V2L4 lot review checklist and lifecycle handoff validation
"""
text = replace_once(text, old, new.rstrip(), "V2L4.6 activation")
work_items.write_text(text, encoding="utf-8")

components = Path("registry/components.yaml")
text = components.read_text(encoding="utf-8")
text = replace_once(
    text,
    "      - src/wre/domain/benchmarks.py\n      - src/wre/domain/__init__.py\n",
    "      - src/wre/domain/benchmarks.py\n      - src/wre/domain/quality_policy.py\n      - src/wre/domain/__init__.py\n",
    "V2L4 quality-policy implementation registry",
)
text = replace_once(
    text,
    "      - tests/test_benchmarks.py\n    status: implementing\n",
    "      - tests/test_benchmarks.py\n      - tests/test_quality_policy.py\n    status: implementing\n",
    "V2L4 quality-policy test registry",
)
components.write_text(text, encoding="utf-8")

state = Path("PROJECT_STATE.yaml")
text = state.read_text(encoding="utf-8")
text = replace_once(text, "active_work_item: V2L4.5", "active_work_item: V2L4.6", "active work item")
text = replace_once(
    text,
    "last_completed: [V2L0.1, V2L0.2, V2L0.3, V2L1.1, V2L1.2, V2L1.3, V2L1.4, V2L1.5, V2L1.6, V2L2.1, V2L2.2, V2L2.3, V2L2.4, V2L2.5, V2L2.6, V2L3.1, V2L3.2, V2L3.3, V2L3.4, V2L3.5, V2L3.6, V2L4.1, V2L4.2, V2L4.3, V2L4.4]",
    "last_completed: [V2L0.1, V2L0.2, V2L0.3, V2L1.1, V2L1.2, V2L1.3, V2L1.4, V2L1.5, V2L1.6, V2L2.1, V2L2.2, V2L2.3, V2L2.4, V2L2.5, V2L2.6, V2L3.1, V2L3.2, V2L3.3, V2L3.4, V2L3.5, V2L3.6, V2L4.1, V2L4.2, V2L4.3, V2L4.4, V2L4.5]",
    "last completed",
)
text = replace_once(text, "last_merged_pr: 62", "last_merged_pr: 63", "last merged PR")
old_goal = """current_goal: >-
  Define only the deterministic fail-closed quality-policy evaluator over typed MetricVector evidence and explicitly
  supplied stable FailureCategory values, returning one auditable QualityDecision without executing routing or retry.
next_action: >-
  Complete only V2L4.5 from its executable contract in registry/work-items/v2m0.yaml: add immutable versioned policy
  rules, pure deterministic evaluation and focused tests for thresholds, mapped failures, missing/unknown inputs and
  decision precedence; do not start V2L4.6 lot-close negative-test work early.
"""
new_goal = """current_goal: >-
  Close V2L4 with fail-closed unknown metric/failure regressions and a composition review covering failure taxonomy,
  typed metrics, quality decisions, benchmark evidence and deterministic policy evaluation without adding routing.
next_action: >-
  Complete only V2L4.6 from its executable contract in registry/work-items/v2m0.yaml: add the focused negative
  regressions, produce and register the V2L4 lot review, then prepare V2L5.1 as the sole ready item without starting
  media-profile implementation early.
"""
text = replace_once(text, old_goal, new_goal, "project goal handoff")
state.write_text(text, encoding="utf-8")

Path(".github/v2l4_5_handoff.py").unlink()
Path(".github/workflows/v2l4-5-handoff.yml").unlink()
