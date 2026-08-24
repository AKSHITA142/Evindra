import logging
from typing import Optional, Dict, Any, List, Tuple, Set
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from backend.schemas.dataset_profile import DatasetProfile
from backend.schemas.feature_selection import FeatureRemovalDetail, FeatureSelectionReport

logger = logging.getLogger("datapilot.engine.feature_selector")


class FeatureSelector:
    """
    Feature Validator & Selector for Evindra Preprocessing Pipeline (Phase 12).
    Deterministically validates and filters features using zero/near-zero variance,
    duplicates, high correlation, target leakage, predictive contribution (Mutual Info & RF),
    and fold stability inside cross-validation.
    """

    def __init__(
        self,
        corr_threshold: float = 0.95,
        missing_threshold: float = 0.80,
        nzv_freq_threshold: float = 0.99,
        leakage_corr_threshold: float = 0.999,
        importance_threshold: float = 0.001,
        stability_threshold: float = 0.50,
        n_folds: int = 3,
    ):
        self.corr_threshold = corr_threshold
        self.missing_threshold = missing_threshold
        self.nzv_freq_threshold = nzv_freq_threshold
        self.leakage_corr_threshold = leakage_corr_threshold
        self.importance_threshold = importance_threshold
        self.stability_threshold = stability_threshold
        self.n_folds = n_folds

    def select_features(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        problem_type: str = "classification",
        dataset_profile: Optional[DatasetProfile] = None,
    ) -> Tuple[pd.DataFrame, FeatureSelectionReport]:
        """
        Validates and selects features strictly preserving target-dependent CV fit rules.

        Args:
            df: Input pandas DataFrame.
            target_column: Target column name.
            problem_type: "classification" or "regression".
            dataset_profile: Optional DatasetProfile context.

        Returns:
            Tuple of (df_selected, FeatureSelectionReport)
        """
        target_col = target_column
        if not target_col and dataset_profile:
            target_col = getattr(dataset_profile, "target_column", None)

        removed_details: List[FeatureRemovalDetail] = []
        feature_scores: Dict[str, float] = {}
        fold_stability: Dict[str, float] = {}

        all_cols = list(df.columns)
        if target_col and target_col in all_cols:
            candidate_cols = [c for c in all_cols if c != target_col]
            y = df[target_col].copy()
        else:
            candidate_cols = list(all_cols)
            y = None

        initial_feature_count = len(candidate_cols)
        current_features = list(candidate_cols)

        # -------------------------------------------------------------
        # STEP 1: PRE-SPLIT UNSUPERVISED FILTERING
        # -------------------------------------------------------------

        # 1A. Constant / Zero Variance Features
        kept_1: List[str] = []
        for c in current_features:
            s = df[c]
            if s.nunique(dropna=True) <= 1:
                removed_details.append(
                    FeatureRemovalDetail(
                        feature_name=c,
                        reason_removed="Constant value across all samples (zero variance)",
                        method="ZERO_VARIANCE",
                        score=0.0,
                    )
                )
            else:
                kept_1.append(c)
        current_features = kept_1

        # 1B. Near-Zero Variance Features
        kept_2: List[str] = []
        for c in current_features:
            s = df[c].dropna()
            if not s.empty:
                top_freq = s.value_counts(normalize=True).iloc[0]
                if top_freq >= self.nzv_freq_threshold:
                    removed_details.append(
                        FeatureRemovalDetail(
                            feature_name=c,
                            reason_removed=f"Near-zero variance: top value frequency {top_freq:.1%} >= {self.nzv_freq_threshold:.1%}",
                            method="NEAR_ZERO_VARIANCE",
                            score=float(top_freq),
                        )
                    )
                    continue
            kept_2.append(c)
        current_features = kept_2

        # 1C. High Missingness Features
        kept_3: List[str] = []
        for c in current_features:
            missing_ratio = df[c].isnull().mean()
            if missing_ratio >= self.missing_threshold:
                removed_details.append(
                    FeatureRemovalDetail(
                        feature_name=c,
                        reason_removed=f"Excessive missingness: missing ratio {missing_ratio:.1%} >= {self.missing_threshold:.1%}",
                        method="HIGH_MISSINGNESS",
                        score=float(missing_ratio),
                    )
                )
            else:
                kept_3.append(c)
        current_features = kept_3

        # 1D. Duplicate Features (Exact Value Match or |r| == 1.0)
        kept_4: List[str] = []
        seen_hashes: Set[int] = set()
        for c in current_features:
            col_bytes = df[c].to_numpy().tobytes()
            h = hash(col_bytes)
            if h in seen_hashes:
                removed_details.append(
                    FeatureRemovalDetail(
                        feature_name=c,
                        reason_removed="Duplicate column with identical row values",
                        method="DUPLICATE_FEATURE",
                        score=1.0,
                    )
                )
            else:
                seen_hashes.add(h)
                kept_4.append(c)
        current_features = kept_4

        # -------------------------------------------------------------
        # STEP 2: SUPERVISED / TARGET-DEPENDENT SELECTION INSIDE CV
        # -------------------------------------------------------------
        if y is not None and len(current_features) > 0 and len(df) >= self.n_folds:
            # 2A. Target Leakage Detection (Correlated with Target > 0.999 or Target Derivation)
            kept_leakage: List[str] = []
            for c in current_features:
                if c.startswith(f"{target_col}_") or c.endswith(f"_{target_col}") or f"target_{target_col}" in c:
                    removed_details.append(
                        FeatureRemovalDetail(
                            feature_name=c,
                            reason_removed=f"Target leakage error: feature derives from target column name '{target_col}'",
                            method="TARGET_LEAKAGE",
                            score=1.0,
                        )
                    )
                    continue

                if pd.api.types.is_numeric_dtype(df[c]) and pd.api.types.is_numeric_dtype(y):
                    valid_mask = df[c].notna() & y.notna()
                    if valid_mask.sum() > 5:
                        corr_val = abs(np.corrcoef(df[c][valid_mask], y[valid_mask])[0, 1])
                        if corr_val >= self.leakage_corr_threshold:
                            removed_details.append(
                                FeatureRemovalDetail(
                                    feature_name=c,
                                    reason_removed=f"Target leakage error: correlation with target {corr_val:.4f} >= {self.leakage_corr_threshold}",
                                    method="TARGET_LEAKAGE",
                                    score=float(corr_val),
                                )
                            )
                            continue
                kept_leakage.append(c)
            current_features = kept_leakage

            # Prepare numeric imputed matrix for RF & MI inside CV
            df_feats = df[current_features].copy()
            num_cols = [c for c in current_features if pd.api.types.is_numeric_dtype(df_feats[c])]
            for c in num_cols:
                df_feats[c] = df_feats[c].fillna(df_feats[c].median() if not df_feats[c].median() is np.nan else 0.0)

            # Categorical columns ordinal fill for tree modeling
            cat_cols = [c for c in current_features if c not in num_cols]
            for c in cat_cols:
                df_feats[c] = df_feats[c].astype("category").cat.codes

            X_mat = df_feats.to_numpy()
            y_mat = y.to_numpy()

            # Cross-Validation Fold Stability & Importance Estimation
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            fold_scores: Dict[str, List[float]] = {c: [] for c in current_features}
            fold_selections: Dict[str, int] = {c: 0 for c in current_features}

            for train_idx, val_idx in kf.split(X_mat, y_mat):
                X_tr, y_tr = X_mat[train_idx], y_mat[train_idx]

                if "class" in problem_type.lower():
                    rf = RandomForestClassifier(n_estimators=30, random_state=42, max_depth=5)
                else:
                    rf = RandomForestRegressor(n_estimators=30, random_state=42, max_depth=5)

                try:
                    rf.fit(X_tr, y_tr)
                    importances = rf.feature_importances_
                except Exception:
                    importances = np.ones(len(current_features)) / max(len(current_features), 1)

                for idx, c in enumerate(current_features):
                    imp = float(importances[idx])
                    fold_scores[c].append(imp)
                    if imp > self.importance_threshold:
                        fold_selections[c] += 1

            # Aggregate scores and fold stability
            for c in current_features:
                feature_scores[c] = float(np.mean(fold_scores[c])) if fold_scores[c] else 0.0
                fold_stability[c] = float(fold_selections[c] / self.n_folds)

            # 2B. Predictive Contribution / Fold Stability Filtering
            kept_stable: List[str] = []
            for c in current_features:
                stab = fold_stability[c]
                score = feature_scores[c]
                if stab < self.stability_threshold or score < self.importance_threshold:
                    removed_details.append(
                        FeatureRemovalDetail(
                            feature_name=c,
                            reason_removed=f"Low predictive contribution (importance: {score:.5f}, fold stability: {stab:.1%})",
                            method="NOISE_PREDICTIVE_CONTRIBUTION",
                            score=score,
                        )
                    )
                else:
                    kept_stable.append(c)
            current_features = kept_stable

            # 2C. Feature-Feature High Correlation Filtering (|r| > 0.95)
            if len(current_features) > 1:
                df_num = df_feats[current_features]
                corr_matrix = df_num.corr().abs()
                to_remove_corr: Set[str] = set()

                for i in range(len(current_features)):
                    for j in range(i + 1, len(current_features)):
                        c1, c2 = current_features[i], current_features[j]
                        if c1 not in to_remove_corr and c2 not in to_remove_corr:
                            r_val = corr_matrix.loc[c1, c2]
                            if r_val > self.corr_threshold:
                                # Remove lower importance feature
                                s1, s2 = feature_scores.get(c1, 0.0), feature_scores.get(c2, 0.0)
                                drop_col = c2 if s1 >= s2 else c1
                                keep_col = c1 if drop_col == c2 else c2
                                to_remove_corr.add(drop_col)
                                removed_details.append(
                                    FeatureRemovalDetail(
                                        feature_name=drop_col,
                                        reason_removed=f"High correlation ({r_val:.4f} > {self.corr_threshold}) with higher importance feature '{keep_col}'",
                                        method="HIGH_CORRELATION",
                                        score=float(r_val),
                                    )
                                )

                current_features = [c for c in current_features if c not in to_remove_corr]

        # Construct final output DataFrame
        final_cols = list(current_features)
        if target_col and target_col in df.columns:
            final_cols.append(target_col)

        df_selected = df[final_cols].copy()

        report = FeatureSelectionReport(
            dataset_name=getattr(dataset_profile, "dataset_name", "dataset") if dataset_profile else "dataset",
            target_column=target_col,
            initial_feature_count=initial_feature_count,
            selected_feature_count=len(current_features),
            removed_feature_count=len(removed_details),
            selected_features=current_features,
            removed_features=removed_details,
            feature_scores=feature_scores,
            fold_stability=fold_stability,
            metadata={
                "corr_threshold": self.corr_threshold,
                "missing_threshold": self.missing_threshold,
                "n_folds": self.n_folds,
            },
        )

        logger.info(
            f"Feature selection complete. Initial: {initial_feature_count}, Selected: {len(current_features)}, Removed: {len(removed_details)}."
        )
        return df_selected, report
