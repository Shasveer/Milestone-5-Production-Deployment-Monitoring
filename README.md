# Khetha Milestone 5: Production Deployment and Monitoring

This repository packages the Khetha fraud-detection workflow as a versioned
production artifact. The root entrypoint is
[`khetha_production_deployment.py`](khetha_production_deployment.py). Running
it trains and validates the model from the local claims dataset, serializes
the complete preprocessing-and-model pipeline, writes deployment/audit
artifacts, exercises inference, and records a drift-monitoring result.

## Repository Contents

- `khetha_production_deployment.py`: deployer, production inference wrapper,
	PSI/performance monitoring, manifest generation, dashboard, and executable
	orchestration.
- `khetha_preprocessing.py`: importable sparse-to-dense transform used in the
	serialized inference pipeline.
- `claims_data.csv`: claim-level model features and `fraud_label` target.
- `fraud_labels.csv`, `vehicles.csv`, `policyholders.csv`: companion reference
	datasets retained with the submission.
- `geographic_risks.csv`: geographic risk reference table required by the
	submission checklist. `geographic_risk.csv` is retained as the supplied
	source filename; both names are supported by the upstream M2/M3 workflows.
- `config.json`: fraud threshold, retraining thresholds, artifact location,
	and South African domain configuration.

The upstream milestones document the integration contract:

- [Milestone 1 data pipeline](https://github.com/Shasveer/Milestone-1-Data-Structure-Problem-Framing-khetha_data_pipeline.py-)
	validates labels, excludes identifiers and post-outcome leakage, and creates
	a stratified train/test split.
- [Milestone 2 baseline/preprocessing](https://github.com/Shasveer/Milestone-2-Baseline-Modelling-Cross-Validation)
	provides the categorical one-hot/numeric passthrough preprocessor.
- [Milestone 3 ensemble optimization](https://github.com/Shasveer/Milestone-3-Ensemble-Optimisation-Nested-Tuning)
	provides the Random Forest ensemble and nested cross-validation interface.
- [Milestone 4 evaluation, interpretability, and drift](https://github.com/Shasveer/Milestone-4-Evaluation-Interpretability-Drift)
	provides classification evaluation, SHAP, and PSI/retraining interfaces.

When these milestone modules are available beside the script or on
`PYTHONPATH`, the runner imports and uses them. In a clean checkout containing
only the required Milestone 5 files, the script executes the same leakage-safe
split, compatible preprocessing, Random Forest training, and audit workflow
locally. Its metadata records which upstream modules were actually integrated.
The auxiliary tables are retained but are not blindly joined: the claims
records do not expose all companion-table keys needed for reliable row-level
merges.

## Setup and Execution

Use Python 3.10 or later. Install the core runtime packages:

```bash
python -m pip install numpy pandas scikit-learn scipy joblib
```

SHAP and the M4 plotting stack are optional for execution. When SHAP is
installed and the upstream evaluator is present, the M4 evaluator writes a
global summary and local case explanations; otherwise the deployment report
records the available model-importance fallback without presenting it as
SHAP output.

Run the complete deployment sequence from the repository root:

```bash
python khetha_production_deployment.py
```

The command is non-interactive and reads `claims_data.csv` and `config.json`
relative to the script. It uses a reproducible random seed, stratifies the
fraud labels, and preserves a holdout for evaluation. The primary target is
`fraud_label`.

## Deployment Artifacts

The run creates `khetha_artifacts/` and writes:

- `pipeline_v1.0.0.joblib`: a single scikit-learn `Pipeline` containing the
	fitted M2-compatible preprocessing, imputation, and classifier.
- `pipeline_v1.0.0.metadata.json`: version, UTC training/serialization times,
	feature schema, model metrics, validation CI/method, SHAP provenance, and
	environment references.
- `environment_lock.json`: exact installed Python and runtime package
	versions.
- `deployment_manifest.json`: service entrypoint, locked dependencies,
	required files, artifact version, and health artifacts.
- `deployment_report_v1.0.0.md`: validation CI bounds, attribution summary,
	monitoring schedule, retraining protocol, and rollback procedure.
- `health_dashboard.json`: latency, baseline/current F1 decay, active drift
	flags, and artifact version history.
- `drift_monitoring.jsonl`: timestamped per-feature PSI, performance decay,
	version, and retraining decision records.
- `shap_summary.png` and local SHAP explanations when the M4 evaluator and
	SHAP dependency are available.

The generated lock captures the actual deployment environment. Validate that
environment before promotion or reuse with:

```bash
python khetha_production_deployment.py --validate-lock
```

Deployment reruns also validate an existing lock and stop on Python or package
version mismatches. Promote the artifact and its matching metadata/lock together
so the preprocessing contract cannot drift from the classifier. Sparse preprocessor output is converted to dense arrays before
median imputation during both training and inference; the runner logs the
estimated dense memory size, so assess that estimate before using this fallback
with substantially larger datasets. This densification also supports sparse
output from the upstream M2 preprocessor. The `khetha_preprocessing.py`
utility must ship beside the script so deserialized pipelines can resolve the
transform.

## Inference Contract

Instantiate `KhethaProductionDeployer` with the fitted complete pipeline and
metadata. `metadata["features_expected"]` is the authoritative input schema.
The inference wrapper accepts exactly one JSON object with all and only those
keys, rejects nested values and schema mismatches, normalizes scalar
categorical/boolean values using the recorded training feature dtypes, loads
the cached joblib artifact, and returns a JSON-compatible record:

```json
{"flag": 0, "prob": 0.12, "version": "1.0.0"}
```

`flag` applies the configured fraud probability threshold (default `0.65`),
and `prob` is the finite positive-class probability in the range `[0, 1]`.
The wrapper rejects an absent positive class and records request latency for
the health dashboard.

## Drift and Retraining

`monitor_drift_and_trigger_retrain(current_batch_X, historical_X)` calculates
PSI independently for every expected predictive feature. Numeric features use
historical quantile bins; categorical and boolean features use the combined
observed category support, including unseen current categories. Constant
numeric historical features use a narrow center bin so shifted values are
visible. Config defaults are PSI `0.15` and absolute F1
decay `0.05`, sourced from `config.json`. Pass `current_f1` and optionally
`baseline_f1` to include performance decay in the same retraining decision.
Every check receives a UTC timestamp and is appended to the drift log; a
trigger also adds a retraining-log entry to the metadata.

The dashboard generator tracks request latency, baseline/current F1 and
absolute decay, most recent per-feature drift flags, retraining trigger state,
and discovered versioned model artifacts. Monitoring should run on each
incoming batch, with labeled performance reviewed at least weekly; the report
records the daily PSI and weekly performance schedule.

## Validation and Governance

The model metric CI is taken from M3 nested cross-validation when that module
is present. In the clean local fallback, a three-fold stratified ROC-AUC
interval is recorded and explicitly labeled as an approximation rather than
nested CV. The deployment report also includes holdout metrics, CI method,
attribution source, feature-schema count, dependency lock, monitoring
thresholds, retraining steps, and the previous-version rollback procedure.
This is an operational audit record aligned to FSCA-oriented model governance;
it is not a legal determination of regulatory compliance.

Quick checks after execution:

```bash
python -m py_compile khetha_production_deployment.py
python khetha_production_deployment.py
```

For a deployed artifact, verify the metadata and manifest alongside the
joblib file, then submit a known-good payload and a deliberately incomplete
payload to confirm prediction and strict schema rejection respectively.