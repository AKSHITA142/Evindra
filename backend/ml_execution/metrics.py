from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
)

from backend.schemas.experiment import MetricsResult


class MetricEngine:
    """Computes comprehensive evaluation metrics for classification and regression tasks."""

    @classmethod
    def compute_metrics(
        cls,
        y_true: Any,
        y_pred: Any,
        y_proba: Any = None,
        task_type: str = "classification",
        cv_scores: list = None,
        train_score: float = None,
        user_goal: str = "",
        target_column: str = "",
        column_names: list = None,
    ) -> MetricsResult:
        """Calculates evaluation metrics dictionary and wraps in MetricsResult."""
        cv_scores = cv_scores or []
        metrics: Dict[str, Any] = {}

        is_imbalanced = False
        if hasattr(y_true, "value_counts"):
            try:
                vc = y_true.value_counts(normalize=True)
                if len(vc) > 1 and float(vc.min()) < 0.20:
                    is_imbalanced = True
            except Exception:
                pass

        if task_type == "classification":
            acc = float(accuracy_score(y_true, y_pred))

            # Use 'binary' averaging for 2-class classification to accurately measure minority/positive class metrics
            # without majority class frequency inflation from 'weighted'
            unique_classes = np.unique(y_true)
            avg_mode = "binary" if len(unique_classes) <= 2 else "macro"

            prec = float(precision_score(y_true, y_pred, average=avg_mode, zero_division=0))
            rec = float(recall_score(y_true, y_pred, average=avg_mode, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, average=avg_mode, zero_division=0))
            bal_acc = float(balanced_accuracy_score(y_true, y_pred))

            if (prec + rec) > 0:
                f2 = float(5.0 * (prec * rec) / (4.0 * prec + rec))
                f05 = float(1.25 * (prec * rec) / (0.25 * prec + rec))
            else:
                f2, f05 = 0.0, 0.0

            metrics["accuracy"] = round(acc, 4)
            metrics["precision"] = round(prec, 4)
            metrics["recall"] = round(rec, 4)
            metrics["f1"] = round(f1, 4)
            metrics["f1_score"] = round(f1, 4)
            metrics["f2_score"] = round(f2, 4)
            metrics["f05_score"] = round(f05, 4)
            metrics["balanced_accuracy"] = round(bal_acc, 4)

            # ROC-AUC if proba available
            if y_proba is not None:
                try:
                    unique_classes = np.unique(y_true)
                    if len(unique_classes) == 2:
                        pos_idx = 1 if (y_proba.ndim > 1 and y_proba.shape[1] > 1) else 0
                        auc = float(roc_auc_score(y_true, y_proba[:, pos_idx]))
                    elif len(unique_classes) > 2:
                        all_labels = np.arange(y_proba.shape[1])
                        auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted", labels=all_labels))
                    metrics["roc_auc"] = round(auc, 4)
                except Exception:
                    pass

        else:
            mae = abs(float(mean_absolute_error(y_true, y_pred)))
            rmse = abs(float(np.sqrt(mean_squared_error(y_true, y_pred))))
            r2 = float(r2_score(y_true, y_pred))
            evs = float(explained_variance_score(y_true, y_pred))

            metrics["mae"] = round(mae, 4)
            metrics["rmse"] = round(rmse, 4)
            metrics["r2"] = round(r2, 4)
            metrics["explained_variance"] = round(evs, 4)

        from backend.evaluation.domain_metric_resolver import DomainMetricResolver
        metric_key, metric_name, rationale = DomainMetricResolver.resolve_primary_metric(
            task_type=task_type,
            user_goal=user_goal,
            target_column=target_column or (str(y_true.name) if hasattr(y_true, "name") else ""),
            column_names=column_names,
            is_imbalanced=is_imbalanced,
            has_proba=(y_proba is not None),
        )

        primary = float(metrics.get(metric_key, metrics.get("f1_score", metrics.get("rmse", 0.0))))

        if cv_scores:
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))
            metrics["cv_mean"] = round(cv_mean, 4)
            metrics["cv_std"] = round(cv_std, 4)
            metrics["test_score"] = round(primary, 4)

        # Honest train_test_gap: |train_score - test_score|
        if train_score is not None:
            metrics["train_score"] = round(train_score, 4)
            metrics["train_test_gap"] = round(abs(train_score - primary), 4)
        elif cv_scores:
            metrics["train_test_gap"] = round(abs(float(np.mean(cv_scores)) - primary), 4)

        return MetricsResult(
            primary_metric=round(primary, 4),
            primary_metric_name=str(metric_name),
            primary_metric_rationale=str(rationale),
            metrics=metrics,
            cv_scores=[round(s, 4) for s in cv_scores],
        )

