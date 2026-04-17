# Kubernetes Bootstrap Notes

## Goal

Bootstrap a fresh Kubernetes workspace with:

- `kg_coop_drive`
- `auto_drive_copy`
- `V2V-GoT`
- `dataset_jsons.zip`
- `dataset_processed_features_and_gt.zip`
- `model_ckpt.zip`

using a persistent volume for downloaded artifacts and repositories.

## Files

- Pod manifest: [k8s/bootstrap-job.yaml](/Users/bhavya/Desktop/ms_projects/kg_coop_drive/k8s/bootstrap-job.yaml:1)

## Persistence Model

### 1. Repository and dataset persistence

Use a shared PVC mounted at `/workspace`.

That makes these survive pod death:

- cloned repos
- downloaded zip files
- extracted datasets

If the job is recreated or a new pod uses the same PVC, it can reuse the previous state.

### 2. SSH credential persistence

Use a Kubernetes `Secret` named `bhgupta-github-ssh`.

That makes SSH credentials survive pod death because the secret is a cluster object, not a pod-local file.

The job copies the mounted secret files into `/root/.ssh` and fixes permissions before cloning.

## Secret Setup

Create the secret once in the target namespace.

Example:

```bash
kubectl -n seelab create secret generic bhgupta-github-ssh \
  --from-file=id_ed25519=$HOME/.ssh/id_ed25519 \
  --from-file=known_hosts=$HOME/.ssh/known_hosts \
  --from-file=config=$HOME/.ssh/config
```

If you do not use a local SSH config file, omit the `config` entry and remove the optional copy logic only if needed.

## Why this is persistent enough

If the pod dies:

- the SSH key still exists in the `bhgupta-github-ssh` secret
- the cloned repos and downloaded data still exist on the PVC
- the next pod can remount both and continue

That is the standard persistent pattern for private Git access in Kubernetes.

## Recommended Workflow

1. Create or confirm the PVC exists.
2. Create the `bhgupta-github-ssh` secret in the same namespace.
3. Apply the pod manifest.
4. Watch the logs until bootstrap completes.
5. `exec` into the same pod and keep working there.

## Apply Commands

```bash
kubectl apply -f k8s/bootstrap-job.yaml
kubectl -n seelab logs -f pod/kg-coop-bootstrap
kubectl -n seelab exec -it kg-coop-bootstrap -- /bin/bash
```

## Notes on Private Repos

`kg_coop_drive` is cloned over SSH:

- `git@github.com:bhavyagupta98/kg_coop_drive.git`

The other two are currently cloned over HTTPS in the manifest because they appear public:

- `https://github.com/bhavyagupta98/auto_drive_copy.git`
- `https://github.com/eddyhkchiu/V2V-GoT.git`

If you want all three cloned over SSH for consistency, the job can be changed easily.

## Notes on Large Artifacts

`dataset_processed_features_and_gt.zip` and `model_ckpt.zip` are very large.

The job uses:

- `curl -L -C - --fail`

so downloads can resume if the target file already exists on the PVC.

## Config Flags

The manifest includes these environment variables:

- `DOWNLOAD_DATASET_JSONS`
- `DOWNLOAD_PROCESSED_FEATURES`
- `DOWNLOAD_MODEL_CKPT`

Set any of them to `"false"` if you want a lighter bootstrap run.

## Pod Behavior

This manifest now creates a long-lived pod instead of a batch job.

Behavior:

- it runs the bootstrap steps once at startup
- it runs validation checks automatically after bootstrap
- after bootstrap, it stays alive with `tail -f /dev/null`
- you can `kubectl exec` into it and keep using the same environment

If you update the manifest, it is usually easiest to recreate the pod:

```bash
kubectl -n seelab delete pod kg-coop-bootstrap
kubectl apply -f k8s/bootstrap-job.yaml
```

## Validation Output

The pod now prints a validation summary into its startup logs after bootstrap.

Watch it with:

```bash
kubectl -n seelab logs -f pod/kg-coop-bootstrap
```

You should see sections like:

- `Bootstrap Validation`
- `Repository Checks`
- `Artifact Checks`
- `Extracted Dataset Checks`
- `Summary`

The most important final line is one of:

- `[PASS] Bootstrap validation completed successfully.`
- `[FAIL] Bootstrap validation found missing files.`

There is also a copy of the validation script in the repo:

- [scripts/validate_bootstrap.sh](/Users/bhavya/Desktop/ms_projects/kg_coop_drive/scripts/validate_bootstrap.sh:1)

Once you `exec` into the pod, you can rerun the same checks manually:

```bash
/usr/local/bin/validate_bootstrap.sh
```
