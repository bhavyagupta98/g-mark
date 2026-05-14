# G-MARK Ablation Interpretation

This note restructures the validation-only ablation results into smaller,
paper-facing tables. The goal is to report effects where the ablation signal is
visible, while being conservative about components that are represented in the
KG but do not yet produce a clear metric change under the current selectors and
heads.

## Ablation Protocol

These are construction-level graph ablations. For each ablation setting, G-MARK
rebuilds the scene graph under the specified constraint before the task handler
or trained head sees the sample.

Examples:

- `ego_only_graph` filters the input observations and visibility facts to the
  asking vehicle before graph construction, so cooperative perception is not
  available to the graph.
- `no_provenance` removes source-agent provenance/support traces from the graph
  representation before inference.
- `no_candidate_retention` drops retained candidate tracks during graph
  construction/readout.
- `no_uncertainty_conflict` neutralizes uncertainty/conflict signals before the
  downstream selector/head uses the graph.
- `flat_non_graph_readout` keeps a flattened object list while removing
  graph-specific readout signals.

The validation tables below use the already selected task heads and policies.
The heads are not retrained separately for each ablation. This isolates the
effect of changing the graph evidence available at validation time under a
fixed downstream decision pipeline.

This protocol is defensible because it asks:

```text
Given the same trained G-MARK task heads, how much do the validation predictions
change when specific KG evidence types are removed before graph construction or
graph readout?
```

It should not be described as a full retraining ablation. A retraining ablation
would answer a different question: whether a model trained from scratch without
that evidence can compensate through other features.

## Metric Directions

- Q1, Q2, Q3, Q4: `F1@0.5m`, higher is better.
- Q5, Q7: `L2 Avg 123 (m)`, lower is better.
- Q8: `Action L1 (edit_dist/8)`, lower is better.

## Table A: Cooperation And Provenance For Visibility-Aware Reasoning

This table focuses on the tasks where cooperative evidence and source
provenance are expected to matter most: occluding objects, invisible objects,
planning awareness, and control settings.

| Setting | Q2 Occluding F1 ↑ | Q3 Invisible F1 ↑ | Q4 Planning F1 ↑ | Q8 Control L1 ↓ | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `full` | 0.427921 | 0.493934 | 0.613774 | 0.076139 | Full cooperative G-MARK. |
| `ego_only_graph` | 0.346497 | 0.000000 | 0.638968 | 0.213037 | Removes shared perception before graph construction. |
| `no_provenance` | 0.446217 | 0.396008 | 0.608350 | 0.151843 | Removes source-agent provenance/support traces. |

Main reading:

- Cooperation is essential for Q3. The `ego_only_graph` setting drops invisible
  object F1 from `0.493934` to `0.000000`, which is the clearest ablation signal
  in the current results.
- Cooperation also helps Q2 and Q8: occluding-object F1 falls from `0.427921` to
  `0.346497`, and control error rises from `0.076139` to `0.213037`.
- Provenance is useful for Q3 and Q8. Removing provenance drops Q3 from
  `0.493934` to `0.396008` and worsens Q8 from `0.076139` to `0.151843`.
- Q4 does not show the same pattern: `ego_only_graph` is higher than `full`.
  This should not be framed as proof that cooperation is unnecessary for
  planning awareness. It suggests the current Q4 selector/acceptor can still
  rely heavily on ego-visible trajectory proximity.

Paper-facing claim:

```text
Cooperative evidence and provenance have the clearest impact on tasks requiring
reasoning about objects outside the ego vehicle's direct view, especially Q3 and
Q8.
```

## Table B: Candidate Retention And Flat Readout

This table separates candidate-track retention and graph-structured readout from
the full model. These ablations probe whether retained candidate hypotheses and
graph-specific signals are helping downstream prediction.

| Setting | Q3 Invisible F1 ↑ | Q5 Object Motion L2 ↓ | Q7 Object Motion L2 ↓ | Q8 Control L1 ↓ | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `full` | 0.493934 | 3.822136 | 3.822136 | 0.076139 | Full cooperative G-MARK. |
| `no_candidate_retention` | 0.497400 | 3.414674 | 3.414674 | 0.075776 | Drops retained candidate tracks. |
| `flat_non_graph_readout` | 0.443299 | 3.403723 | 3.403723 | 0.088654 | Uses a flat object list while removing graph-specific signals. |

Main reading:

- The Q3 and Q8 signals suggest graph-specific readout is useful: the flat
  readout worsens Q3 from `0.493934` to `0.443299` and Q8 from `0.076139` to
  `0.088654`.
- Candidate retention does not show a clean benefit in this checkpoint.
  Removing candidates slightly improves Q3/Q8 and improves Q5/Q7 L2. This is not
  a good headline ablation for claiming candidate retention improves metrics.
- The Q5/Q7 improvement under `no_candidate_retention` and
  `flat_non_graph_readout` suggests that the current motion head may prefer
  cleaner, lower-cardinality object sets. This is useful diagnostic evidence,
  not a positive candidate-retention result.

Paper-facing claim:

```text
Flat readout weakens invisible-object and control reasoning, suggesting that
structured graph signals are useful for safety-relevant QA. Candidate retention,
however, is not yet a clean positive contributor under the current motion
prediction head.
```

## Table C: Uncertainty And Conflict Signals

This table focuses on uncertainty/conflict metadata. These fields are present in
the KG and support interpretability, but the current validation results show
limited metric sensitivity.

| Setting | Q2 Occluding F1 ↑ | Q3 Invisible F1 ↑ | Q4 Planning F1 ↑ | Q8 Control L1 ↓ | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `full` | 0.427921 | 0.493934 | 0.613774 | 0.076139 | Full cooperative G-MARK. |
| `no_uncertainty_conflict` | 0.426642 | 0.489761 | 0.610333 | 0.077409 | Neutralizes uncertainty and conflict scores. |

Main reading:

- The metric differences are small across Q2, Q3, Q4, and Q8.
- This does not mean uncertainty/conflict metadata is useless. It means the
  current deterministic selectors and shallow heads are not strongly using those
  fields on this split.
- The features remain valuable for interpretability: they expose whether an
  object was uncertain, conflicting, weakly supported, or cooperatively
  confirmed. That interpretability can support qualitative analysis and future
  models even when the current metric delta is small.

Paper-facing claim:

```text
Uncertainty and conflict are represented explicitly in the KG, but current
validation metrics show only weak sensitivity to ablating these fields. We
therefore treat them primarily as interpretability and future-modeling signals,
not as a headline performance driver in this checkpoint.
```

## Recommended Paper Usage

Use Table A as the strongest ablation evidence. It supports the central
cooperative-reasoning argument: shared perception and provenance matter most for
invisible/occluded/safety-control tasks.

Use Table B as a diagnostic graph-structure table. It gives a moderate positive
story for graph readout on Q3/Q8, but it should also acknowledge that candidate
retention needs more refinement.

Use Table C as a conservative interpretability table. It shows that the KG
captures uncertainty/conflict, while honestly reporting that the current
checkpoint does not extract a large metric gain from those fields.

Suggested phrasing:

```text
Not every graph attribute produces a large validation delta under the current
deterministic selectors. We therefore separate performance-critical components
from interpretability-oriented components. Cooperative evidence and provenance
show clear effects on visibility-aware and safety-control tasks, while
uncertainty/conflict metadata currently contributes more to traceability than to
headline metric gains.
```
