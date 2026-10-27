"""Khetha Mutual production deployment, inference, and monitoring workflow."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("khetha.production")
RUNTIME_DISTRIBUTIONS = ("numpy", "pandas", "scikit-learn", "scipy", "joblib")
OPTIONAL_DISTRIBUTIONS = ("matplotlib", "shap")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (Path, datetime)):
        return value.isoformat() if isinstance(value, datetime) else str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _read_config(root: Path = ROOT) -> Dict[str, Any]:
    config_path = root / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object")
    return config


def _optional_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        LOGGER.info("Milestone module %s is unavailable: %s", module_name, error)
        return None


def _local_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    categorical = X_train.select_dtypes(include=["object", "string", "bool", "category"]).columns.tolist()
    numeric = [column for column in X_train.columns if column not in categorical]
    return ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ],
        remainder="drop",
    )


def _dense_for_median_imputation(encoded: Any) -> Any:
    if not hasattr(encoded, "toarray"):
        return encoded
    estimated_bytes = (
        int(encoded.shape[0])
        * int(encoded.shape[1])
        * np.dtype(encoded.dtype).itemsize
    )
    LOGGER.info(
        "Densifying sparse encoded data for median imputation (estimated %.2f MiB)",
        estimated_bytes / (1024 ** 2),
    )
    return encoded.toarray()


def _local_feature_split(claims: pd.DataFrame):
    target_column = "fraud_label"
    if target_column not in claims.columns:
        raise KeyError("claims_data.csv must contain fraud_label")
    labelled = claims.dropna(subset=[target_column]).copy()
    target = labelled[target_column].astype(int)
    excluded = {
        "claim_id", "claim_date", "fraud_label", "investigation_outcome",
        "claim_settlement_date", "investigation_date", "fraud_type",
        "recovered_amount_zar", "investigation_notes",
    }
    features = labelled.drop(columns=[column for column in excluded if column in labelled.columns])
    if features.empty or target.nunique() != 2:
        raise ValueError("Fraud training requires non-empty features and two target classes")
    return train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )


def _build_training_assets(root: Path = ROOT):
    """Use M1-M4 modules when present; otherwise execute their local workflow."""
    claims_path = root / "claims_data.csv"
    if not claims_path.is_file():
        raise FileNotFoundError("claims_data.csv must be beside this script")
    claims = pd.read_csv(claims_path)
    if claims.empty:
        raise ValueError("claims_data.csv contains no rows")

    m1 = _optional_module("khetha_data_pipeline")
    m2 = _optional_module("khetha_baseline_models")
    m3 = _optional_module("khetha_ensemble_optimizer")
    m4 = _optional_module("khetha_evaluation_interpretability")

    milestone1_used = False
    if m1 is not None and hasattr(m1, "KhethaDataPipeline"):
        preparation = m1.KhethaDataPipeline(claims)
        X_train, X_test, y_train, y_test = preparation.run(
            "fraud_label", ["province", "claim_type"]
        )
        milestone1_used = True
    else:
        X_train, X_test, y_train, y_test = _local_feature_split(claims)

    preprocessor = (
        m2._make_preprocessor(X_train)
        if m2 is not None and hasattr(m2, "_make_preprocessor")
        else _local_preprocessor(X_train)
    )
    encoded_train = preprocessor.fit_transform(X_train, y_train)
    encoded_test = preprocessor.transform(X_test)
    encoded_train = _dense_for_median_imputation(encoded_train)
    encoded_test = _dense_for_median_imputation(encoded_test)
    imputer = SimpleImputer(strategy="median")
    encoded_train = imputer.fit_transform(encoded_train)
    encoded_test = imputer.transform(encoded_test)

    optimizer = None
    nested_result = None
    if m3 is not None and hasattr(m3, "KhethaEnsembleOptimizer"):
        try:
            optimizer = m3.KhethaEnsembleOptimizer(
                encoded_train, np.asarray(y_train), encoded_test, np.asarray(y_test)
            )
            optimizer.build_ensembles("random_forest")
            optimizer.run_grid_search(
                {
                    "n_estimators": [100],
                    "max_features": ["sqrt"],
                    "min_samples_leaf": [1, 2],
                    "class_weight": ["balanced"],
                },
                cv=3,
            )
            try:
                nested_result = optimizer.nested_cross_validation(outer_cv=3, inner_cv=2)
            except (ValueError, TypeError, AttributeError) as error:
                LOGGER.warning("M3 nested CV unavailable for this data: %s", error)
            model = optimizer.tuned_model
        except (ValueError, TypeError, AttributeError) as error:
            LOGGER.warning("M3 optimizer could not complete; using local forest: %s", error)
            optimizer = None

    if optimizer is None:
        model = RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model, encoded_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        mean_score = float(np.mean(cv_scores))
        margin = float(1.96 * np.std(cv_scores, ddof=1) / np.sqrt(len(cv_scores)))
        nested_result = {
            "mean_score": mean_score,
            "ci_95": (max(0.0, mean_score - margin), min(1.0, mean_score + margin)),
            "outer_scores": cv_scores.tolist(),
            "method": "3-fold stratified CV approximation (M3 module unavailable)",
        }
        model.fit(encoded_train, y_train)

    if not hasattr(model, "classes_"):
        model.fit(encoded_train, y_train)

    complete_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("imputer", imputer),
            ("model", model),
        ]
    )
    encoded_predictions = model.predict(encoded_test)
    if hasattr(model, "predict_proba"):
        probability_matrix = model.predict_proba(encoded_test)
        positive_index = list(model.classes_).index(1) if 1 in model.classes_ else -1
        probabilities = probability_matrix[:, positive_index]
        roc_auc = float(roc_auc_score(y_test, probabilities))
    else:
        probabilities = np.zeros(len(encoded_test), dtype=float)
        roc_auc = None

    metrics = {
        "f1": float(f1_score(y_test, encoded_predictions, zero_division=0)),
        "precision": float(precision_score(y_test, encoded_predictions, zero_division=0)),
        "recall": float(recall_score(y_test, encoded_predictions, zero_division=0)),
        "roc_auc": roc_auc,
        "accuracy": float(np.mean(np.asarray(encoded_predictions) == np.asarray(y_test))),
    }
    evaluation_summary = None
    shap_summary = {}
    shap_source = "unavailable"
    if m4 is not None and hasattr(m4, "KhethaEvaluator"):
        try:
            evaluator = m4.KhethaEvaluator(
                model, encoded_train, y_train, encoded_test, y_test, _read_config(root)
            )
            evaluation_summary = evaluator.generate_classification_report()
            metrics.update(evaluation_summary.get("metrics", {}))
            try:
                shap_result = evaluator.apply_shap_explanations(
                    n_samples=50, output_dir=root / "khetha_artifacts"
                )
                shap_summary = {
                    row["feature"]: row["mean_absolute_shap"]
                    for row in shap_result.get("summary_data", [])
                }
                shap_source = "SHAP TreeExplainer"
            except (ImportError, ValueError, TypeError, AttributeError, OSError) as error:
                LOGGER.warning("M4 SHAP output unavailable: %s", error)
        except (ValueError, TypeError, AttributeError) as error:
            LOGGER.warning("M4 evaluator could not complete: %s", error)

    if not shap_summary and hasattr(model, "feature_importances_"):
        try:
            names = list(preprocessor.get_feature_names_out())
        except (AttributeError, ValueError):
            names = [f"feature_{index}" for index in range(len(model.feature_importances_))]
        shap_summary = {
            str(name): float(value)
            for name, value in zip(names, model.feature_importances_)
        }
        shap_source = "Random Forest feature-importance fallback; SHAP not installed"

    if nested_result is None:
        nested_result = {"ci_95": (0.0, 1.0), "method": "unavailable"}
    ci = nested_result.get("ci_95", (0.0, 1.0))
    if "method" not in nested_result:
        nested_result["method"] = "M3 nested cross-validation" if optimizer is not None else "cross-validation"

    metadata = {
        "training_date": _utc_now(),
        "metrics": metrics,
        "features_expected": list(X_train.columns),
        "positive_class": 1,
        "decision_threshold": float(_read_config(root).get("fraud_detection", {}).get("default_threshold", 0.65)),
        "performance_ci": [float(ci[0]), float(ci[1])],
        "ci_method": nested_result.get("method", "cross-validation"),
        "nested_cv": nested_result,
        "shap_summary": shap_summary,
        "shap_source": shap_source,
        "milestone_integration": {
            "milestone_1_data_pipeline": milestone1_used,
            "milestone_2_preprocessor": m2 is not None and hasattr(m2, "_make_preprocessor"),
            "milestone_3_ensemble_optimizer": optimizer is not None,
            "milestone_4_evaluator": evaluation_summary is not None,
        },
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    return complete_pipeline, metadata, X_train, X_test, y_test


class KhethaProductionDeployer:
    """Serialize complete pipelines, serve predictions, and monitor production health."""

    def __init__(self, pipeline, metadata, drift_thresholds=None):
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary")
        self.pipeline = pipeline
        self.metadata = metadata
        self.drift_thresholds = drift_thresholds or {"psi": 0.15, "f1_drop": 0.05}
        deployment_config = _read_config().get("deployment", {})
        self.artifact_dir = Path(deployment_config.get("artifact_dir", "khetha_artifacts"))
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.active_version = str(metadata.get("version", "1.0.0"))
        self.cached_pipeline_path = None
        self.latency_history = []
        self.performance_history = []
        self.drift_events = []
        self.metadata.setdefault("features_expected", list(getattr(pipeline, "feature_names_in_", [])))
        self.metadata.setdefault("metrics", {})
        self.metadata.setdefault("retraining_log", [])

    def _versioned_path(self, version: str, suffix: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", str(version)):
            raise ValueError("version must contain only letters, digits, dot, underscore, or hyphen")
        return self.artifact_dir / f"pipeline_v{version}.{suffix}"

    def generate_environment_manifest(self) -> Dict[str, Any]:
        """Write exact runtime version locks and a production dependency manifest."""
        versions = {}
        for distribution in RUNTIME_DISTRIBUTIONS + OPTIONAL_DISTRIBUTIONS:
            try:
                versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                continue
        lock = {
            "generated_at": _utc_now(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": versions,
        }
        manifest = {
            "service": "khetha-fraud-inference",
            "entrypoint": "khetha_production_deployment.py",
            "model_version": self.active_version,
            "python_version": platform.python_version(),
            "runtime_dependencies": versions,
            "required_files": [
                "claims_data.csv", "fraud_labels.csv", "vehicles.csv",
                "policyholders.csv", "geographic_risks.csv", "config.json",
            ],
            "health_artifacts": ["deployment_manifest.json", "environment_lock.json", "health_dashboard.json"],
        }
        self._write_json(self.artifact_dir / "environment_lock.json", lock)
        self._write_json(self.artifact_dir / "deployment_manifest.json", manifest)
        self.metadata["environment_lock"] = "environment_lock.json"
        self.metadata["deployment_manifest"] = "deployment_manifest.json"
        self.metadata["environment_versions"] = versions
        return manifest

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump(_json_safe(payload), output_file, indent=2, sort_keys=True, allow_nan=False)
            output_file.write("\n")
        os.replace(temporary_path, path)

    def serialize_pipeline(self, version: str) -> str:
        """Save the full preprocessing/model pipeline and versioned audit metadata."""
        pipeline_path = self._versioned_path(version, "joblib")
        metadata_path = self._versioned_path(version, "metadata.json")
        expected_features = self.metadata.get("features_expected")
        if not isinstance(expected_features, list) or not expected_features:
            raise ValueError("metadata['features_expected'] must be a non-empty list")
        self.active_version = str(version)
        self.metadata.update(
            {
                "version": self.active_version,
                "training_date": self.metadata.get("training_date", _utc_now()),
                "metrics": self.metadata.get("metrics", {}),
                "features_expected": expected_features,
                "serialized_at": _utc_now(),
                "artifact_path": str(pipeline_path),
            }
        )
        temporary_path = pipeline_path.with_suffix(".joblib.tmp")
        joblib.dump(self.pipeline, temporary_path, compress=3)
        os.replace(temporary_path, pipeline_path)
        self._write_json(metadata_path, self.metadata)
        self.cached_pipeline_path = pipeline_path
        return str(pipeline_path)

    def create_inference_wrapper(self, raw_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate one raw JSON record and return a versioned fraud prediction."""
        if not isinstance(raw_payload, dict):
            raise TypeError("raw_payload must be a JSON object")
        expected = self.metadata.get("features_expected", [])
        if not expected:
            raise ValueError("metadata['features_expected'] is required for inference")
        missing = sorted(set(expected) - set(raw_payload))
        unexpected = sorted(set(raw_payload) - set(expected))
        if missing or unexpected:
            raise ValueError(f"Payload schema mismatch; missing={missing}, unexpected={unexpected}")
        for key, value in raw_payload.items():
            if isinstance(value, (dict, list, tuple, set)):
                raise TypeError(f"Feature {key!r} must be a scalar JSON value")

        if self.cached_pipeline_path is None or not self.cached_pipeline_path.is_file():
            self.serialize_pipeline(self.active_version)
        start_time = time.perf_counter()
        cached_pipeline = joblib.load(self.cached_pipeline_path)
        row = pd.DataFrame([{feature: raw_payload[feature] for feature in expected}], columns=expected)
        if not hasattr(cached_pipeline, "predict_proba"):
            raise TypeError("The serialized pipeline must implement predict_proba()")
        probabilities = np.asarray(cached_pipeline.predict_proba(row))
        classes = list(getattr(cached_pipeline, "classes_", []))
        if probabilities.ndim != 2 or probabilities.shape != (1, len(classes)):
            raise ValueError("The serialized pipeline returned an invalid probability matrix")
        positive_class = self.metadata.get("positive_class", 1)
        if positive_class not in classes:
            raise ValueError(f"Positive class {positive_class!r} is not present in the model classes")
        positive_index = classes.index(positive_class)
        probability = float(probabilities[0, positive_index])
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("The serialized pipeline returned a non-finite or out-of-range probability")
        threshold = float(self.metadata.get("decision_threshold", 0.65))
        result = {"flag": int(probability >= threshold), "prob": probability, "version": self.active_version}
        self.latency_history.append((time.perf_counter() - start_time) * 1000.0)
        return result

    @staticmethod
    def _psi(expected: pd.Series, actual: pd.Series) -> float:
        expected = expected.dropna()
        actual = actual.dropna()
        if expected.empty or actual.empty:
            raise ValueError("PSI requires non-empty historical and current feature values")
        numeric = (
            pd.api.types.is_numeric_dtype(expected)
            and pd.api.types.is_numeric_dtype(actual)
            and not pd.api.types.is_bool_dtype(expected)
            and not pd.api.types.is_bool_dtype(actual)
        )
        if numeric:
            expected_values = expected.astype(float).to_numpy()
            actual_values = actual.astype(float).to_numpy()
            edges = np.unique(np.quantile(expected_values, np.linspace(0.0, 1.0, 11)))
            if len(edges) < 2:
                center = float(edges[0])
                margin = max(abs(center) * 1e-6, 1e-6)
                edges = np.array([-np.inf, center - margin, center + margin, np.inf])
            else:
                edges[0], edges[-1] = -np.inf, np.inf
            expected_counts = np.histogram(expected_values, bins=edges)[0].astype(float)
            actual_counts = np.histogram(actual_values, bins=edges)[0].astype(float)
        else:
            expected_values = expected.astype("string").fillna("<MISSING>")
            actual_values = actual.astype("string").fillna("<MISSING>")
            categories = expected_values.unique().tolist()
            categories.extend(item for item in actual_values.unique().tolist() if item not in categories)
            expected_counts = np.asarray([(expected_values == item).sum() for item in categories], dtype=float)
            actual_counts = np.asarray([(actual_values == item).sum() for item in categories], dtype=float)
        epsilon = 1e-6
        expected_share = np.clip(expected_counts / expected_counts.sum(), epsilon, None)
        actual_share = np.clip(actual_counts / actual_counts.sum(), epsilon, None)
        return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))

    def monitor_drift_and_trigger_retrain(
        self,
        current_batch_X: pd.DataFrame,
        historical_X: pd.DataFrame,
        current_f1: float | None = None,
        baseline_f1: float | None = None,
    ) -> bool:
        """Compute per-feature PSI and trigger on drift or absolute F1 decay."""
        if not isinstance(current_batch_X, pd.DataFrame) or not isinstance(historical_X, pd.DataFrame):
            raise TypeError("current_batch_X and historical_X must be pandas DataFrames")
        features = self.metadata.get("features_expected", [])
        absent = [feature for feature in features if feature not in current_batch_X or feature not in historical_X]
        if absent:
            raise ValueError(f"Drift data is missing predictive features: {absent}")

        psi_limit = float(self.drift_thresholds.get("psi", 0.15))
        feature_scores = {
            feature: self._psi(historical_X[feature], current_batch_X[feature])
            for feature in features
        }
        drift_flags = {
            feature: score >= float(self.drift_thresholds.get("psi_by_feature", {}).get(feature, psi_limit))
            for feature, score in feature_scores.items()
        }
        metrics = self.metadata.get("metrics", {})
        baseline = float(baseline_f1 if baseline_f1 is not None else metrics.get("f1", 0.0))
        latest = float(current_f1) if current_f1 is not None else baseline
        f1_drop = max(0.0, baseline - latest)
        f1_limit = float(self.drift_thresholds.get("f1_drop", 0.05))
        f1_threshold_reached = f1_drop >= f1_limit or np.isclose(
            f1_drop, f1_limit, rtol=1e-9, atol=1e-12
        )
        retrain = any(drift_flags.values()) or f1_threshold_reached
        event = {
            "timestamp": _utc_now(),
            "version": self.active_version,
            "psi": feature_scores,
            "drift_flags": drift_flags,
            "baseline_f1": baseline,
            "current_f1": latest,
            "f1_drop": f1_drop,
            "retrain_triggered": bool(retrain),
        }
        self.drift_events.append(event)
        if retrain:
            self.metadata.setdefault("retraining_log", []).append(
                {"timestamp": event["timestamp"], "reason": "PSI threshold or F1 decay exceeded", "version": self.active_version}
            )
        with (self.artifact_dir / "drift_monitoring.jsonl").open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(_json_safe(event), sort_keys=True) + "\n")
        return bool(retrain)

    def generate_deployment_report(
        self,
        performance_ci: Tuple[float, float],
        shap_summary: Dict,
    ) -> str:
        """Create an audit-oriented registration report with evidence and rollback steps."""
        lower, upper = (float(performance_ci[0]), float(performance_ci[1]))
        if lower > upper:
            raise ValueError("performance_ci lower bound must not exceed its upper bound")
        self.metadata["performance_ci"] = [lower, upper]
        self.metadata["shap_summary"] = _json_safe(shap_summary)
        top_features = sorted(shap_summary.items(), key=lambda item: abs(float(item[1])), reverse=True)[:10]
        feature_rows = "\n".join(
            f"| {feature} | {float(value):.6f} |" for feature, value in top_features
        ) or "| No attribution values supplied | N/A |"
        metrics = self.metadata.get("metrics", {})
        retraining_log = self.metadata.get("retraining_log", [])
        retraining_rows = "\n".join(
            f"| {event.get('timestamp', 'not recorded')} | {event.get('reason', 'not recorded')} | {event.get('version', self.active_version)} |"
            for event in retraining_log
        ) or "| No retraining event recorded | N/A | N/A |"
        return f"""# Khetha Production Deployment Audit Report

## Registration and version control
- Model version: `{self.active_version}`
- Training timestamp (UTC): `{self.metadata.get('training_date', 'not recorded')}`
- Artifact: `{self.metadata.get('artifact_path', self._versioned_path(self.active_version, 'joblib'))}`
- Expected input features: {len(self.metadata.get('features_expected', []))}
- Environment lock: `{self.metadata.get('environment_lock', 'environment_lock.json')}`
- Deployment manifest: `{self.metadata.get('deployment_manifest', 'deployment_manifest.json')}`
- Positive class / decision threshold: `{self.metadata.get('positive_class', 1)}` / `{self.metadata.get('decision_threshold', 0.65)}`

## Validation evidence
- Validation metric: ROC-AUC
- 95% confidence interval: **[{lower:.4f}, {upper:.4f}]**
- CI method: `{self.metadata.get('ci_method', 'nested cross-validation')}`
- Nested-CV evidence: `{json.dumps(_json_safe(self.metadata.get('nested_cv', {})), sort_keys=True)}`
- Holdout metrics: F1 `{metrics.get('f1', 'not recorded')}`, precision `{metrics.get('precision', 'not recorded')}`, recall `{metrics.get('recall', 'not recorded')}`, ROC-AUC `{metrics.get('roc_auc', 'not recorded')}`
- Feature preprocessing and classifier are serialized together in the versioned joblib artifact.

## SHAP and interpretability audit
- Attribution source: `{self.metadata.get('shap_source', 'SHAP summary supplied by evaluator')}`
- Summary timestamp (UTC): `{_utc_now()}`

| Feature | Mean absolute SHAP / attribution |
| --- | ---: |
{feature_rows}

## Drift monitoring and retraining
- Schedule: run PSI checks on each incoming batch and at least daily; evaluate labeled performance weekly.
- PSI warning/retraining threshold: `{self.drift_thresholds.get('psi', 0.15)}` per predictive feature.
- Performance decay trigger: absolute F1 decrease of `{self.drift_thresholds.get('f1_drop', 0.05)}` or greater from the approved baseline.
- Every check records UTC timestamp, feature PSI values, flags, model version, and retraining decision in `drift_monitoring.jsonl`.
- Retraining history:

| Timestamp (UTC) | Trigger | Model version |
| --- | --- | --- |
{retraining_rows}

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
"""

    def generate_health_dashboard(
        self,
        current_f1: float | None = None,
        current_accuracy: float | None = None,
    ) -> Dict[str, Any]:
        """Persist current latency, performance decay, drift flags, and artifact history."""
        metrics = self.metadata.get("metrics", {})
        baseline_f1 = float(metrics.get("f1", 0.0))
        latest_f1 = float(current_f1) if current_f1 is not None else baseline_f1
        baseline_accuracy = float(metrics.get("accuracy", 0.0))
        latest_accuracy = float(current_accuracy) if current_accuracy is not None else baseline_accuracy
        versions = sorted(path.stem.removeprefix("pipeline_") for path in self.artifact_dir.glob("pipeline_v*.joblib"))
        latest_event = self.drift_events[-1] if self.drift_events else {}
        dashboard = {
            "generated_at": _utc_now(),
            "model_version": self.active_version,
            "model_version_history": versions,
            "latency_ms": {
                "latest": float(self.latency_history[-1]) if self.latency_history else None,
                "mean": float(np.mean(self.latency_history)) if self.latency_history else None,
                "max": float(np.max(self.latency_history)) if self.latency_history else None,
                "observations": len(self.latency_history),
            },
            "accuracy_decay": {
                "baseline_accuracy": baseline_accuracy,
                "current_accuracy": latest_accuracy,
                "absolute_accuracy_drop": max(0.0, baseline_accuracy - latest_accuracy),
            },
            "f1_decay": {
                "baseline_f1": baseline_f1,
                "current_f1": latest_f1,
                "absolute_f1_drop": max(0.0, baseline_f1 - latest_f1),
                "threshold": float(self.drift_thresholds.get("f1_drop", 0.05)),
            },
            "active_drift_flags": latest_event.get("drift_flags", {}),
            "latest_retraining_trigger": bool(latest_event.get("retrain_triggered", False)),
        }
        self._write_json(self.artifact_dir / "health_dashboard.json", dashboard)
        return dashboard

    def deploy(self, version: str) -> Dict[str, Any]:
        """Write locks, model, report, and health dashboard as one deployment sequence."""
        self.active_version = str(version)
        manifest = self.generate_environment_manifest()
        artifact = self.serialize_pipeline(version)
        report = self.generate_deployment_report(
            tuple(self.metadata.get("performance_ci", (0.0, 1.0))),
            self.metadata.get("shap_summary", {}),
        )
        report_path = self.artifact_dir / f"deployment_report_v{version}.md"
        report_path.write_text(report, encoding="utf-8")
        dashboard = self.generate_health_dashboard()
        return {
            "status": "active",
            "version": str(version),
            "artifact": artifact,
            "metadata": str(self._versioned_path(version, "metadata.json")),
            "manifest": manifest,
            "report": report,
            "report_path": str(report_path),
            "dashboard": dashboard,
        }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    pipeline, metadata, X_train, X_test, _ = _build_training_assets(ROOT)
    config = _read_config(ROOT)
    thresholds = config.get("model_retraining", {})
    deployer = KhethaProductionDeployer(
        pipeline,
        metadata,
        drift_thresholds={
            "psi": float(thresholds.get("psi_drift_threshold", 0.15)),
            "f1_drop": float(thresholds.get("f1_decay_threshold", 0.05)),
        },
    )
    deployment = deployer.deploy("1.0.0")
    if len(X_test):
        deployer.create_inference_wrapper(X_test.iloc[0].to_dict())
    triggered = deployer.monitor_drift_and_trigger_retrain(
        X_test, X_train, current_f1=metadata["metrics"].get("f1")
    )
    dashboard = deployer.generate_health_dashboard(
        current_f1=metadata["metrics"].get("f1"),
        current_accuracy=metadata["metrics"].get("accuracy"),
    )
    print(
        f"Khetha Model v{deployment['version']} deployed; artifact={deployment['artifact']}; "
        f"retraining_alert={triggered}; dashboard={dashboard['model_version']}"
    )


if __name__ == "__main__":
    main()