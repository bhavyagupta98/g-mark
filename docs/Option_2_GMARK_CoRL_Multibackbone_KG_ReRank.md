# Option 2: Stronger Learned Baseline and Multi-Backbone KG Re-Ranking for G-MARK

## 1. Aim of the experiment

The aim of this experiment is to strengthen the CoRL paper by adding a stronger learned comparison around G-MARK. The current paper may be criticized as a hand-engineered KG pipeline. This experiment tests whether the explicit KG representation helps beyond a flat object list and whether a lightweight learned module can improve the final results.

There are two possible versions of this option:

1. **Learned baseline:** train a simple GNN or MLP baseline over the same object evidence and compare it with G-MARK.
2. **Lightweight learned re-ranker:** use G-MARK features as input to a learned candidate re-ranker for Q1 to Q4 and possibly Q8.

If time is short, prioritize the learned re-ranker because it is easier to implement and directly improves the headline table.

## 2. Why this strengthens the CoRL paper

The current G-MARK draft has a strong robotics motivation, but reviewers may ask whether the method is mostly a rule-based or benchmark-specific pipeline. A learned baseline or learned re-ranker makes the paper stronger in three ways:

1. It improves the robot-learning story.
2. It tests whether explicit provenance-aware KG features are better than flat object features.
3. It may improve the main results table, especially Q1 to Q4 and Q8.

The key claim becomes:

> Explicit cooperative evidence is useful not only for deterministic reasoning, but also as a compact feature representation for learned downstream decision modules.

This is much better aligned with CoRL than a purely handcrafted pipeline.

## 3. Dataset and input evidence

### Primary dataset

Use the same dataset as the current paper:

- V2V-GoT-QA,
- built on V2V4Real,
- same train, validation, and test split as the current evaluation.

### Optional additional dataset

If time permits, test a small transfer or sanity experiment on:

- OPV2V, using OpenCOOD-compatible processed detections,
- or V2X-Sim, using multi-agent object annotations and poses.

This is optional. The main deliverable should be a stronger learned baseline on V2V-GoT-QA.

### Input evidence

Use the same input evidence already used by G-MARK:

- object boxes or positions,
- object confidence scores,
- semantic class,
- agent pose,
- source-agent ID,
- support count,
- visibility status if available,
- candidate status,
- uncertainty score,
- disagreement score,
- planning relevance score,
- spatial relations,
- motion cues if available.

## 4. Experimental design

The experiment compares four families of methods.

### Method A: Full G-MARK

This is the current method.

Inputs:

- G-MARK KG,
- provenance,
- visibility,
- uncertainty,
- disagreement,
- planning relevance,
- task-specific heads.

Output:

- same outputs as current paper for Q1 to Q9.

Purpose:

- main method.

### Method B: Flat object-list baseline

This baseline removes the graph structure and provenance relations.

Inputs:

- object position,
- object class,
- object confidence,
- optionally velocity or motion cues.

Removed:

- provenance,
- source support,
- ego/partner visibility,
- candidate status,
- explicit graph edges,
- disagreement,
- KG relations.

Output:

- same task outputs.

Purpose:

- tests whether object geometry alone is enough.

### Method C: Flat learned MLP baseline

Train an MLP over per-candidate flat features.

Candidate-level features:

```text
[x, y, z or bbox center,
 object size if available,
 class one-hot,
 confidence,
 distance to ego,
 angle to ego,
 velocity if available,
 distance to planned path if available]
```

For Q1 to Q4:

- each object candidate gets a score,
- select top-k or thresholded candidates,
- train using binary cross entropy against whether candidate matches a ground-truth answer.

For Q8:

- aggregate object features using mean, max, and top-k risk pooling,
- predict speed and steering class or normalized action vector,
- train with cross entropy or L1 loss depending on label format.

Purpose:

- tests whether a simple learned model over flat object features can match G-MARK.

### Method D: KG feature re-ranker

Train a lightweight model using G-MARK-derived candidate features.

Candidate-level features:

```text
object position,
 semantic class,
 confidence,
 support-agent count,
 ego-visible flag,
 partner-only flag,
 candidate flag,
 uncertainty score,
 disagreement score,
 planning relevance score,
 distance to ego,
 distance to planned path,
 occluder relation count,
 hidden-candidate relation count,
 provenance strength,
 number of supporting observations
```

Model choices:

- logistic regression,
- small MLP,
- gradient boosted trees,
- or shallow GNN if implementation time allows.

Recommended first choice:

- small MLP with 2 hidden layers,
- hidden size 64 or 128,
- dropout 0.1,
- binary cross entropy for Q1 to Q4.

For Q1 to Q4:

- produce candidate score per object,
- rank candidates,
- output top-k or thresholded set.

For Q8:

- use pooled KG features,
- train speed and steering prediction,
- compare Action L1.

Purpose:

- tests whether explicit KG features improve a learned downstream model.

### Optional Method E: GNN over KG

If time allows, train a simple graph neural network over the G-MARK graph.

Nodes:

- object-hypothesis nodes,
- observation nodes,
- agent nodes.

Edges:

- observed-by,
- supports,
- visible-to,
- near,
- occludes,
- planning-relevant.

Node features:

- position,
- class,
- confidence,
- visibility,
- candidate flag,
- uncertainty,
- disagreement,
- support count.

Model:

- 2-layer GraphSAGE or GAT,
- node classifier for Q1 to Q4,
- graph pooling plus MLP for Q8.

Purpose:

- stronger ML baseline, but higher implementation risk.

## 5. Training setup

### Candidate label construction for Q1 to Q4

For each sample:

1. Build candidate object hypotheses.
2. Compare each candidate to the reference answer coordinates.
3. Mark candidate positive if it is within the benchmark matching threshold, such as 0.5 m.
4. Mark all other candidates negative.
5. Train binary classifier per task type or a shared classifier with task embedding.

### Q8 label construction

Use the reference control answer from V2V-GoT-QA.

Possible targets:

- speed label,
- steering label,
- normalized action vector,
- or action L1 target.

Train either:

- classification for speed and steering bins,
- or regression to normalized action values.

### Train / validation / test split

Use the official V2V-GoT-QA split.

If the official split is not directly exposed, use the same split as the current G-MARK evaluation scripts.

Important:

- Do not tune on the test set.
- Use validation for threshold selection.
- Report final numbers once on test.

### Hyperparameters

Recommended MLP setup:

```text
hidden layers: 2
hidden dimension: 64 or 128
activation: ReLU
optimizer: Adam
learning rate: 1e-3
batch size: 256 candidates or scene-level batches
epochs: 20 to 50
early stopping: validation F1 or validation loss
seeds: 3 if possible
```

Thresholding:

- tune score threshold on validation set,
- or output top-k where k is estimated from training answer cardinality.

## 6. Metrics to report

Use the same metrics as the current paper.

### Q1 to Q4

- Precision at 0.5 m,
- Recall at 0.5 m,
- F1 at 0.5 m.

Main focus:

- Q2 Occluding Objects F1,
- Q3 Invisible Objects F1,
- Q4 Planning Awareness F1.

### Q8

- Action L1 error,
- lower is better.

### Optional Q9

Only include Q9 if the learned model actually changes the trajectory module.

Metrics:

- average L2 error,
- near-horizon L2,
- far-horizon L2.

### Communication and runtime

For all learned models, report:

- parameter count,
- CPU inference time per sample,
- payload size if unchanged from G-MARK,
- whether the learned model changes communication or only local reasoning.

## 7. Baselines and prior papers to compare against

### Internal baselines

At minimum:

| Method | Purpose |
|---|---|
| V2V-GoT | closest external cooperative reasoning baseline |
| Full G-MARK | current method |
| Flat object list | tests whether KG structure matters |
| Flat MLP | tests whether simple learned flat features are enough |
| KG MLP re-ranker | tests whether learned model benefits from KG features |
| No provenance KG MLP | tests whether provenance matters in learned setting |
| No visibility KG MLP | tests whether visibility matters in learned setting |

### External prior papers

Use these as related baselines or context:

- V2V-GoT, for cooperative graph-of-thought reasoning on V2V-GoT-QA.
- V2V-LLM, for multimodal LLM cooperative driving QA.
- V2V4Real, for real-world cooperative perception benchmark context.
- OPV2V and OpenCOOD, for cooperative perception inputs and possible transfer.
- V2X-ViT and CoBEVT, for strong cooperative perception backbones.

The paper should be careful not to claim direct superiority over perception backbones unless you actually run them. The comparison should be framed as reasoning-layer evaluation.

## 8. Output tables

### Main learned-baseline table

| Method | Q1 F1 ↑ | Q2 F1 ↑ | Q3 F1 ↑ | Q4 F1 ↑ | Q8 L1 ↓ | Runtime ms ↓ |
|---|---:|---:|---:|---:|---:|---:|
| V2V-GoT | | | | | | |
| Flat object list | | | | | | |
| Flat MLP | | | | | | |
| Full G-MARK current | | | | | | |
| KG MLP re-ranker | | | | | | |
| KG MLP no provenance | | | | | | |
| KG MLP no visibility | | | | | | |

### Feature ablation table

| KG feature set | Q2 F1 ↑ | Q3 F1 ↑ | Q8 L1 ↓ |
|---|---:|---:|---:|
| geometry only | | | |
| + confidence | | | |
| + provenance | | | |
| + visibility | | | |
| + uncertainty | | | |
| + disagreement | | | |
| + planning relevance | | | |
| all KG features | | | |

This table is very valuable because it connects the performance improvements directly to the design choices in the paper.

## 9. Optional multi-backbone extension

If V2V-GoT-QA or processed artifacts include outputs from multiple perception backbones, such as PointPillars, AttFuse, V2X-ViT, or CoBEVT, use them as alternate inputs to G-MARK.

### Aim

Show that G-MARK is not tied to one upstream perception backbone.

### Design

Run the same G-MARK pipeline using detection outputs from different backbones:

| Upstream evidence | Reasoning layer | Output |
|---|---|---|
| PointPillars | G-MARK | Q1-Q4, Q8 |
| AttFuse | G-MARK | Q1-Q4, Q8 |
| V2X-ViT | G-MARK | Q1-Q4, Q8 |
| CoBEVT | G-MARK | Q1-Q4, Q8 |

### Metrics

- Q1 to Q4 F1,
- Q8 Action L1,
- duplicate rate,
- partner-only recall,
- false merge rate if available.

### Contribution

This would support the claim that G-MARK is a general reasoning layer above cooperative perception, not a one-off pipeline tied to one detector.

### Risk

This may be time-consuming if processed outputs are not already available. Do not prioritize this over the learned re-ranker unless the files are already present.

## 10. Expected outcomes and interpretation

### Best-case outcome

The KG MLP re-ranker improves Q2, Q3, Q4, and Q8 compared with the current G-MARK head and flat MLP.

Interpretation:

- explicit KG evidence provides useful features for learned downstream reasoning,
- provenance and visibility improve both deterministic and learned reasoning,
- G-MARK is a reusable representation, not just a handcrafted solver.

### Moderate outcome

The KG MLP improves Q1 to Q4 but does not improve Q8.

Interpretation:

- KG features are most useful for object-selection and visibility-sensitive tasks,
- control selection may need a stronger scene-level or temporal model.

### Weak outcome

Flat MLP matches or beats KG MLP.

Interpretation:

- current KG features may not add enough information beyond geometry,
- need stronger ablation to identify which KG fields matter,
- keep this result in appendix or use it to motivate future learned KG reasoning.

## 11. Risks and mitigation

### Risk 1: Learned model overfits

Mitigation:

- use small MLP,
- early stopping,
- validation threshold tuning,
- report 3 seeds if possible.

### Risk 2: Class imbalance in object candidates

Most candidates are negative.

Mitigation:

- use weighted BCE,
- use focal loss,
- sample hard negatives,
- tune threshold on validation F1.

### Risk 3: Learned re-ranker improves only slightly

Mitigation:

- still report feature ablations,
- emphasize that explicit KG features are useful and lightweight,
- keep deterministic G-MARK as main method if learned results are not better.

### Risk 4: GNN takes too long

Mitigation:

- skip GNN,
- use MLP re-ranker,
- describe GNN as future work.

### Risk 5: Results conflict with current narrative

For example, flat MLP may perform strongly.

Mitigation:

- frame G-MARK as providing interpretable features with similar or better accuracy,
- emphasize communication efficiency and traceability,
- use feature ablation to show what the MLP relies on.

## 12. Concrete contribution to the final paper

This experiment would add a strong learning-focused paragraph to the paper:

> To test whether the proposed KG is useful beyond deterministic rules, we train lightweight candidate re-rankers over flat object features and over G-MARK features. The KG-based re-ranker improves visibility-sensitive tasks compared with a flat MLP, showing that provenance, visibility, uncertainty, and planning relevance provide useful inductive bias for learned cooperative reasoning. This strengthens the interpretation of G-MARK as a reusable evidence representation rather than a benchmark-specific rule system.

This directly addresses one of the biggest potential CoRL reviewer concerns.

## 13. Minimal implementation checklist

- [ ] Export candidate-level features from G-MARK for Q1 to Q4.
- [ ] Construct candidate labels using 0.5 m matching to reference answers.
- [ ] Train flat MLP baseline.
- [ ] Train KG-feature MLP re-ranker.
- [ ] Train no-provenance and no-visibility variants.
- [ ] Tune thresholds on validation set.
- [ ] Evaluate on test split.
- [ ] Report F1, precision, recall, runtime, and parameter count.
- [ ] Add learned-baseline table to paper.

## 14. Recommended priority

If time is limited, implement only this sequence:

1. candidate feature export,
2. flat MLP for Q1 to Q4,
3. KG MLP re-ranker for Q1 to Q4,
4. no-provenance ablation,
5. no-visibility ablation.

This is the highest-value version of Option 2 because it directly addresses the concern that G-MARK is only handcrafted feature engineering.
