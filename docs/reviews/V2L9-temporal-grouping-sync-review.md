# V2L9 — Temporal grouping and synchronization foundations review

**Review status:** PASS  
**Reviewed:** 2026-09-20  
**Lot:** `V2L9 — Temporal grouping and synchronization foundations`

## Review objective

Verify that V2L9.1 through V2L9.5 compose into one conservative event-time synchronization foundation without conflating relative event alignment with absolute capture truth, long-term historical chronology, source fusion or global timeline construction.

The lot must retain exact evidence provenance, keep `SUPPORTED`, `UNRESOLVED` and explicit `CONTRADICTED` states distinct, permit metadata/audio/visual evidence to disagree without inventing a winner, and leave cross-source synchronization fusion to its later owning responsibility.

## V2L9 implementation sequence

V2L9 was intentionally split into five bounded work items:

1. PR #89 — `V2L9.1 TemporalGroup and SyncHypothesis contracts`: canonical event-time group identity and evidence-scoped pairwise relative-offset representation.
2. PR #90 — `V2L9.2 Metadata timecode synchronization evidence`: fail-closed adaptation of already-interpreted capture time plus exact same-source-video frame timing.
3. PR #91 — `V2L9.3 Audio correlation synchronization adapter`: bounded deterministic FFmpeg reference extraction plus explicit-policy audio-correlation evidence.
4. PR #92 — `V2L9.4 Visual event synchronization interface`: solver-independent pairwise visual-event candidate/policy interface with no visual execution or implicit fusion.
5. PR #93 — `V2L9.5 Synchronization ambiguity contradiction fixtures`: composition regressions proving agreement, ambiguity, source disagreement and explicit contradiction remain separate evidence-scoped hypotheses.

No V2L9 item performs multi-source consensus, clock-drift correction, global timeline optimization, dynamic 4D reconstruction or long-term historical chronology.

## Composition findings

**PASS.**

- `TemporalGroup` is an explicit event-time synchronization context tied to one existing `SceneClusterId`; it is not an absolute timestamp, historical epoch or implicit result of pairwise evidence.
- `SyncHypothesis` uses canonical observation endpoints, exact evidence refs and one signed microsecond offset only when `SUPPORTED`.
- The shared offset convention is stable across metadata, same-video, audio and visual paths: positive `offset_us` means event time on observation 2 equals event time on observation 1 plus the offset.
- `UNRESOLVED` and `CONTRADICTED` never carry a fake accepted offset.
- Capture-time synchronization consumes only already-interpreted `CaptureTimeInterpretation` values. It does not reparse raw EXIF, guess a timezone, substitute `received_at`, or equate publication/upload time with capture time.
- Two absent capture-time interpretations produce no hypothesis rather than fabricated unresolved evidence.
- Local-ambiguous, invalid or partially absent capture-time evidence remains `UNRESOLVED`.
- Same-video frame synchronization compares `frame_time_us` only when both frames have the exact same parent `video_asset`; unrelated source-relative clocks fail closed.
- V2L9.3 verifies exact source bytes before external execution and reuses the retained FFmpeg 6.1.1-3ubuntu5 identity rather than creating a second executable identity path.
- Audio extraction is bounded mono signed 16-bit PCM at an explicit sample rate and duration. Correlation work is bounded by policy and implemented with Python standard-library arithmetic only.
- Audio acceptance has no hidden threshold. Weak, ambiguous, silent, insufficient-overlap or unavailable audio remains unresolved and never becomes contradiction merely because it is poor evidence.
- The approved `ffmpeg.audio_correlation_sync` registry entry reuses the existing FFmpeg dependency/license posture and adds no new project dependency, model or checkpoint.
- The FFmpeg 8 compatibility lane remained green while retaining the intentional production pin to 6.1.1-3ubuntu5; V2L9 does not silently upgrade the production toolchain.
- V2L9.4 is interface-only. It accepts already-produced visual-event evidence with explicit candidate offset, residual, support and multiplicity; it does not decode media, run tracking, optical flow or a model.
- A visual candidate becomes supported only when it is `UNIQUE` and passes the explicit support/residual policy. Ambiguous or weak candidates remain unresolved.
- Ordinary metadata ambiguity, audio weakness, visual ambiguity and supported-source disagreement never manufacture `CONTRADICTED`.
- Explicit contradiction remains first-class only when separately supplied as explicit contradiction evidence.
- V2L9.5 proves two independently supported sources may expose the same offset and remain separate records.
- V2L9.5 also proves two independently supported sources may expose different offsets and still remain separate supported evidence-scoped records; V2L9 neither chooses a winner nor fabricates contradiction.
- Unresolved audio and ambiguous visual evidence preserve source-local provenance and do not inherit stronger evidence from another source.
- Explicit contradiction can coexist beside supported or unresolved hypotheses without silently overwriting any of them.
- No V2L9 package surface provides hypothesis fusion, source ranking, winner selection, `TemporalGroup` construction from evidence, route selection or global timeline construction.

## Ownership boundaries retained

**PASS.**

- Cross-source synchronization fusion is deferred to **V2L29**, where metadata/audio/visual evidence may be combined under an explicit multi-video event responsibility.
- Long-term historical chronology remains **V2L31+** and is distinct from V2L9 continuous event-time alignment.
- Dynamic reconstruction, persistent motion and runtime temporal products remain V2L24 through V2L30.
- Routing and quality-mode selection remain V2L10 responsibilities.
- V2L9 therefore establishes evidence and representation boundaries without prematurely deciding how later systems consume competing synchronization hypotheses.

## Cross-cutting invariant audit

**PASS.**

- Raw observation truth remains immutable.
- Publication/upload time is never silently treated as capture time.
- Missing timezone remains ambiguous rather than fabricated UTC.
- Routing does not enter synchronization truth.
- Evidence provenance remains exact per source.
- Competing hypotheses are retained rather than destructively collapsed.
- Weak or unsupported evidence fails closed to absence or `UNRESOLVED`.
- Explicit contradiction is distinct from disagreement or low confidence.
- Solver-private correlation/support/residual details remain outside stable `SyncHypothesis` fields.
- No generated content, geometry, scene identity, historical state or route decision is inferred by synchronization foundations.
- No hidden fallback or try-everything behavior is introduced.
- No new persistence schema, network path or mutable global state is introduced.

## Review findings resolved during the lot

**PASS after correction.**

- PR #89 required only canonical export ordering and Ruff formatting; temporal semantics did not change.
- PR #90 required Ruff cleanup of exact `timedelta` microsecond arithmetic, formatting-only normalization and one quoted YAML handoff scalar; no timezone or timing semantics changed.
- PR #91 required synchronization export sorting, Ruff formatting, one test fixture adjustment to remain under the production correlation-work bound, and an explicit work-item scope addition for the unavoidable approved-registry baseline assertion. No truth threshold was weakened.
- PR #91 also updated the historical approved-baseline regression from four to five adapters because V2L9.3 explicitly approved `ffmpeg.audio_correlation_sync`; no unrelated candidate was promoted.
- PR #92 required one Ruff formatting-only correction in the visual interface test.
- PR #93 required one Ruff formatting-only correction in the composition fixture.
- Final PR review relies on exact-head CI evidence and manual deny-by-default diff review rather than status counts alone.
- No unresolved review thread or known semantic regression remains at the V2L9 review boundary.

## Evidence

Exact retained evidence:

- V2L9.1 / PR #89 / final head `b1ccb4e5abf9ddde14a272fcd0edcbfa5f737601`: fast-ci #677, native PyCOLMAP #356, CodeQL #615 — **PASS**.
- V2L9.2 / PR #90 / functional head `0398fe55357270fe1e76c1642529905ed120569c`: fast-ci #681, CodeQL #619 — **PASS**. Final head `c6c6de76325e9cd19083590a0bfacc1db797d92a`: fast-ci #683, native PyCOLMAP #359, CodeQL #621 — **PASS**.
- V2L9.3 / PR #91 / functional head `26b265ed82ac7506aa797955095a1a81a82c8315`: fast-ci #688, CodeQL #626 — **PASS**. Registry head `821d2b9bd4156eba920c2f6e5b80d1facff4cf29`: fast-ci #690, CodeQL #628, FFmpeg8 compatibility #34 — **PASS**. Final head `51dc58ffbeeea9207152d0814c30911a5faa4711`: fast-ci #691, CodeQL #629, FFmpeg8 compatibility #35, native PyCOLMAP #361 — **PASS**.
- V2L9.4 / PR #92 / functional head `b54bc0a5b9801ad18cbda88a7fc924295d827d84`: fast-ci #694, CodeQL #632 — **PASS**. Final head `05e27dc9b49bf041325d9f3eb1853c21cf470eb9`: fast-ci #695, CodeQL #633, native PyCOLMAP #363 — **PASS**.
- V2L9.5 / PR #93 / fixture head `aa93ab60b6debcd7c59d8f219c73fe054c72eabe`: fast-ci #698, CodeQL #636 — **PASS**.

The final PR #93 lifecycle/review head changes review and machine-state metadata after this already-green fixture evidence. It must independently pass every triggered exact-head lane before merge.

## V2L10.1 handoff decision

After this PASS, `V2L10.1 — Adopt canonical QualityMode in routing contracts` may become the sole `ready` item.

V2L10.1 must reuse the exact `wre.domain.QualityMode` type established by V2L4.4, with the closed `PREVIEW`, `FAST`, `QUALITY` and `MASTER` vocabulary. Routing may consume that mode as a compute/product request but must not redefine it, coerce arbitrary strings into it, treat it as provenance/truth, or begin route-graph/fallback logic owned by later V2L10 items.

## Decision

**V2L9 lot review: PASS.**

WRE now has a conservative event-time synchronization foundation spanning explicit contracts, fail-closed capture-time and same-video timing, bounded audio correlation, a solver-independent visual-event evidence interface and composition fixtures that preserve ambiguity, disagreement and explicit contradiction without implicit fusion. Cross-source fusion remains deferred to V2L29 and long-term chronology to V2L31+.
