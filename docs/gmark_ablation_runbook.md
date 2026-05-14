# G-MARK Ablation Runbook

This runbook captures the isolated G-MARK ablation workflow. It is additive and
does not change the default e2e train or validation behavior.

## Why This Ablation Matters

The main benchmark table tells us whether G-MARK performs well. This ablation
suite is meant to explain why it performs well and which parts of the
architecture are actually carrying the result.

The paper-facing claim is not simply that we tuned task heads for Q1-Q9. The
claim is that an explicit cooperative scene graph is a useful reasoning
substrate. The graph stores multi-agent object evidence, provenance, retained
candidate hypotheses, uncertainty/conflict, and derived relations such as path
relevance and cooperative support. The ablation suite removes each of these
ingredients in isolation.

This gives a defensible mechanism study:

- `ego_only_graph` tests whether cooperative evidence matters at all.
- `no_candidate_retention` tests whether keeping weak/partner-only hypotheses is
  important for invisible-object and planning-aware reasoning.
- `no_provenance` tests whether remembering who saw an object and how much
  support it has matters beyond raw object coordinates.
- `no_uncertainty_conflict` tests whether quality/conflict scoring prevents noisy
  cooperative evidence from dominating selection.
- `no_graph_relations` tests whether derived graph edges add value beyond raw
  object attributes.
- `flat_non_graph_readout` tests the strongest skeptical baseline: maybe the
  system only needs a flat object list with geometry. If full G-MARK improves
  over this row, the graph representation is doing useful work.

The expected paper use is a table that reports task metrics for each ablation
mode, plus deltas from full G-MARK. Large drops identify which graph components
are important for which task families.

## Two Evaluation Designs

The runner supports two complementary designs.

Validation-only ablations:

- reuse an existing full G-MARK trained manifest;
- perturb the graph only at validation time;
- are fast diagnostic checks;
- answer: "how sensitive is the current trained system to removing this graph
  signal?"

Train+validation ablations:

- train ablation-specific artifacts from the train split;
- validate those artifacts on the validation split using the same ablated graph;
- are slower but more paper-defensible;
- answer: "if this architecture component is never available, how well can the
  method recover?"

For the paper, train+validation numbers are the stronger ablation. Validation-only
numbers are useful debugging and sanity evidence.

The train+validation runner now writes an isolated train-minus-validation QA
file before fitting ablation-specific models. The filter removes any train QA
record whose `scenario_index`, timestamp, `qa_type_id`, and exact question text
match a validation QA record. This avoids training on validation-equivalent QA
rows while leaving the original V2V-GoT folders untouched.

## Ablation Modes

| Mode | Baseline Mode | Graph Ablation | Meaning |
| --- | --- | --- | --- |
| `full` | `cooperative` | `full` | Full cooperative G-MARK |
| `no_provenance` | `cooperative` | `no_provenance` | Remove source-agent/support provenance |
| `no_candidate_retention` | `cooperative` | `no_candidate_retention` | Drop retained candidate tracks |
| `no_uncertainty_conflict` | `cooperative` | `no_uncertainty_conflict` | Neutralize uncertainty/conflict scores |
| `no_graph_relations` | `cooperative` | `no_graph_relations` | Remove derived relation edges |
| `ego_only_graph` | `ego_only` | `full` | Use only asking-agent evidence |
| `flat_non_graph_readout` | `cooperative` | `flat_non_graph_readout` | Keep a flat object list without graph-specific signals |

## Output Layout

All ablation outputs are isolated under:

```text
outputs/gmark_ablations/<run_name>/
  ablation_summary.md
  ablation_summary.json
  filtered_train/
  validation_only/<mode>/
  trained_validation/<mode>/
  trained_e2e_runs/<mode>_trained/
```

The normal `outputs/e2e_runs` and `outputs/phase8_*` paths are not used by the
ablation runner unless explicitly requested elsewhere.

## Local/Pod Command

Run both validation-only and train+validation ablations:

```bash
cd /workspace/repos/kg_coop_drive

python3 scripts/e2e/run_gmark_ablation_report.py \
  --run-name gmark_ablation_v1 \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --workers 32 \
  --progress-every 250
```

Run only validation-level diagnostics from an existing e2e manifest:

```bash
cd /workspace/repos/kg_coop_drive

python3 scripts/e2e/run_gmark_ablation_report.py \
  --run-name gmark_ablation_validation_only_v1 \
  --manifest-json outputs/e2e_runs/<run_name>/e2e_model_manifest.json \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --skip-trained-validation
```

Run a subset of modes:

```bash
python3 scripts/e2e/run_gmark_ablation_report.py \
  --run-name gmark_ablation_subset_v1 \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --mode full \
  --mode no_graph_relations \
  --mode ego_only_graph
```

## Kubernetes Job

Apply the dedicated job:

```bash
kubectl delete job gmark-ablation-report gmark-ablation-report-v2 -n seelab --ignore-not-found
kubectl apply -f k8s/gmark-ablation-job.yaml
kubectl get pods -n seelab -l app=gmark-ablation-report-v2 -o wide
kubectl describe job gmark-ablation-report-v2 -n seelab
kubectl describe pod -n seelab -l app=gmark-ablation-report-v2
kubectl logs job/gmark-ablation-report-v2 -n seelab --tail=200
```

To run only validation-level ablations, set `RUN_TRAINED_VALIDATION` to `0` in
the job env before applying. To run both validation-only and train+validation
ablations, set it to `1`.

If `kubectl logs -f` fails with a kubelet timeout such as
`dial tcp <node>:10250: i/o timeout`, treat that as a cluster log-streaming
problem rather than proof that the Python job failed. The job writes a persistent
copy of its bootstrap and Python output to:

```text
/workspace/gmark_ablation_job_logs/<run_name>.log
```

The job also uses `terminationMessagePolicy: FallbackToLogsOnError`, so
`kubectl describe pod ...` should include useful error text even when live log
streaming is flaky.

The job uses `quay.io/jupyter/scipy-notebook:latest` and runs as root because
the cluster has shown DNS failures during `pip install`. This image is expected
to provide `numpy`, `pyyaml`, and `scikit-learn` without runtime package
downloads. The preflight check exits early if any of those packages are missing.

## Notes

- Validation-only ablations reuse an existing full G-MARK model manifest and
  perturb the graph at evaluation time.
- Train+validation ablations train ablation-specific models from the train split
  and then evaluate them on the validation split.
- Some train+validation modes can legitimately fail. For example,
  `no_candidate_retention` may remove the candidate rows needed for Q3 acceptor
  training. The runner records failed modes as `ERR` unless `--fail-fast` is set.

## Wiring Audit Notes

After the first full job run, the ablation table showed weak or mixed deltas.
That was useful, but a code audit found a few ablation-wiring issues that should
be fixed before treating the table as paper-facing:

- `ego_only_graph` was filtering observations and visibility facts, but the
  prepared scene still kept the full GT object-track list. The stricter behavior
  now keeps only object tracks that are visible to the asking agent and removes
  non-asker visibility facts after graph enrichment. This makes `ego_only_graph`
  a cleaner "asking-agent evidence only" baseline.
- `no_provenance` now removes source-agent and observation provenance entirely,
  and also removes provenance-derived relation facts such as
  `cooperatively_supported`.
- `no_uncertainty_conflict` now removes direct uncertainty/conflict scores and
  drops conflict-derived relation traces such as `low_conflict`.
- `no_graph_relations` only removes relation facts. If a downstream task head
  does not read relation facts directly, this row can remain identical to full.
  That should be interpreted as "relations are not currently used by this
  readout," not as evidence that relation edges are useless.
- The ablation modes are now construction-level where possible, not only
  post-hoc answer filtering. For example, `no_candidate_retention` skips
  candidate promotion/resolution during scene construction, `no_provenance`
  avoids support/provenance enrichment and strips provenance before downstream
  stages, `no_uncertainty_conflict` skips quality scoring and removes conflict
  traces, `no_graph_relations` skips relation construction, and
  `flat_non_graph_readout` disables candidate retention, provenance,
  uncertainty/conflict, visibility inference, and relation construction.
- The default `full` path remains the normal e2e path. These construction
  changes are only activated when a non-full `graph_ablation` mode is explicitly
  requested by the ablation runner or diagnostic tools.
- For train+validation ablations, training scripts receive the filtered train
  QA JSON through the ablation-only
  `scripts/e2e/run_gmark_ablation_train_pipeline.py`. Validation still uses the
  original validation QA JSON. This keeps previous standalone train/eval flows
  reproducible while making the ablation job leakage-aware.
- Q6 and Q9 are expected to be mostly unchanged in the current implementation.
  Q6 trains from QA metadata/agent-path features rather than the prepared graph,
  and Q9 trains from raw V2V-GoT control/trajectory metadata. They should not be
  used as evidence for graph-component ablations unless those trainers are
  explicitly connected to graph-derived features.

The safest paper-facing interpretation is therefore task-family specific:
provenance/candidate/uncertainty ablations are meaningful for graph-readout
tasks such as Q1-Q5/Q7/Q8, while Q6/Q9 require separate work before they can
support graph-mechanism claims.

## Preflight Diagnostic Before Full Rerun

Before launching the full Kubernetes ablation job, run a small construction
diagnostic on the VM. This checks that cooperative and ego-only scenes differ,
and that each ablation mode actually changes graph construction:

```bash
cd /workspace/repos/kg_coop_drive

python3 scripts/inspect_gmark_ego_coop_graph_delta.py \
  --v2vgot-root /workspace/repos/V2V-GoT \
  --split val \
  --qa-type-id 12 \
  --qa-type-id 13 \
  --qa-type-id 14 \
  --limit-per-task 100 \
  --include-answers \
  --graph-ablation-mode no_provenance \
  --graph-ablation-mode no_candidate_retention \
  --graph-ablation-mode no_uncertainty_conflict \
  --graph-ablation-mode no_graph_relations \
  --graph-ablation-mode flat_non_graph_readout \
  --examples 5 \
  --output-json outputs/gmark_ablations/ego_coop_graph_delta_q2_q3_q4_construction_v2.json
```

Expected sanity checks:

- `ego_only` should have fewer objects than `cooperative`, especially on Q2-Q4.
- `no_candidate_retention` should reduce candidate counts.
- `no_graph_relations` and `flat_non_graph_readout` should report zero
  relations.
- `flat_non_graph_readout` should have no candidates/provenance-style support
  signals.
