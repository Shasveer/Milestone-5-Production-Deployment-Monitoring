# Khetha Production Deployment Audit Report

## Registration and version control
- Model version: `1.0.0`
- Training timestamp (UTC): `2026-10-27T13:13:42.830269+00:00`
- Artifact: `khetha_artifacts/pipeline_v1.0.0.joblib`
- Expected input features: 8
- Environment lock: `environment_lock.json`
- Deployment manifest: `deployment_manifest.json`
- Positive class / decision threshold: `1` / `0.65`

## Validation evidence
- Validation metric: ROC-AUC
- 95% confidence interval: **[0.6565, 0.7025]**
- CI method: `3-fold stratified CV approximation (M3 module unavailable)`
- Nested-CV evidence: `{"ci_95": [0.6564714416344581, 0.7024856953260084], "mean_score": 0.6794785684802332, "method": "3-fold stratified CV approximation (M3 module unavailable)", "outer_scores": [0.6619480876412538, 0.6747206119264775, 0.7017670058729686]}`
- Holdout metrics: F1 `0.17391304347826086`, precision `0.15789473684210525`, recall `0.1935483870967742`, ROC-AUC `0.6923868486786091`
- Feature preprocessing and classifier are serialized together in the versioned joblib artifact.

## SHAP and interpretability audit
- Attribution source: `Random Forest feature-importance fallback; SHAP not installed`
- Summary timestamp (UTC): `2026-10-27T13:13:43.002013+00:00`

| Feature | Mean absolute SHAP / attribution |
| --- | ---: |
| numeric__claim_amount | 0.269045 |
| numeric__policyholder_tenure_months | 0.168657 |
| numeric__vehicle_age_years | 0.129578 |
| categorical__claim_type_hijacking | 0.065424 |
| numeric__prior_claims_count | 0.039275 |
| categorical__province_Gauteng | 0.023193 |
| categorical__claim_type_theft | 0.021737 |
| categorical__province_Western Cape | 0.021369 |
| categorical__submission_channel_web_portal | 0.020296 |
| categorical__submission_channel_mobile_app | 0.020074 |

## Drift monitoring and retraining
- Schedule: run PSI checks on each incoming batch and at least daily; evaluate labeled performance weekly.
- PSI warning/retraining threshold: `0.15` per predictive feature.
- Performance decay trigger: absolute F1 decrease of `0.05` or greater from the approved baseline.
- Every check records UTC timestamp, feature PSI values, flags, model version, and retraining decision in `drift_monitoring.jsonl`.
- Retraining history:

| Timestamp (UTC) | Trigger | Model version |
| --- | --- | --- |
| No retraining event recorded | N/A | N/A |

## Retraining protocol
1. Freeze the triggering batch and its schema, record IDs, data quality checks, and monitoring evidence in the audit log.
2. Rebuild the leakage-safe M1 data split and training-only M2 preprocessing; rerun M3 tuning/nested validation and M4 evaluation/SHAP checks.
3. Require a reviewed validation report, documented 95% CI, threshold assessment, schema compatibility test, and approved model version before promotion.
4. Serialize preprocessing and estimator together, record the environment lock, run inference/drift smoke checks, and retain the prior approved artifact.

## Rollback procedure
1. Disable traffic to the new version and restore the last approved versioned `pipeline_v*.joblib` artifact and its matching metadata.
2. Verify the restored version, feature schema, positive-class mapping, and decision threshold with a known-good smoke payload.
3. Record rollback timestamp, initiating alert, affected versions, operator approval, and impact review in the retraining/audit log.

## Governance record
This report records model version, training/validation evidence, explainability artifacts, drift thresholds, retraining protocol, and rollback controls for FSCA-oriented model governance. It is an operational audit record, not a legal determination of regulatory compliance.
