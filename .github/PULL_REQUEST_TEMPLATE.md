## Work item

`<ID>` — `<title>`

## Objective

<!-- What this PR changes and why. Keep scope aligned to the active work item. -->

## Allowed scope

<!-- List the exact behavior/contracts this PR is allowed to add/change. Missing prohibitions are not permission. -->

## Reuse

<!-- External capabilities reused. State explicitly what was NOT reimplemented and the reviewed version/license assumptions when relevant. -->

## Invariants / ownership boundaries

<!-- Which system invariants and responsibility boundaries does this PR touch? State what must remain unchanged. -->

## Acceptance criteria

- [ ] Work-item acceptance criteria satisfied
- [ ] Relevant positive and negative tests added/updated and passing
- [ ] No unrelated or speculative scope added
- [ ] External model/dependency identity/license metadata updated when relevant
- [ ] Documentation/state/registry updated when required

## Test evidence

<!-- Exact commands/checks, integration/benchmark lanes and results. -->

## Review checklist

- [ ] Diff matches objective and the allowlisted scope
- [ ] Invalid/unsupported states fail closed or remain explicit
- [ ] Interfaces and data ownership remain coherent
- [ ] Solver/model-private structures did not leak into stable core contracts
- [ ] Reuse-before-implementation policy respected
- [ ] Provenance/artifact identity and reproducibility are preserved
- [ ] Regression/security/resource risks considered
- [ ] Generated/inferred/reconstructed semantics remain distinct where applicable
- [ ] Known limitations and future work recorded rather than bundled

## Known limitations / follow-up

<!-- Record follow-up work; do not silently implement the next work item in this PR. -->
