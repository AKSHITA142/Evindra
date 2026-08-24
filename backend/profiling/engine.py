from typing import Optional, Tuple, Dict, Any
import pandas as pd

from backend.schemas.semantic_profile import SemanticProfile
from backend.schemas.dataset_profile import DatasetProfile
from backend.profiling.loader import DataLoader
from backend.profiling.schema_analyzer import SchemaAnalyzer
from backend.profiling.statistics_analyzer import StatisticsAnalyzer
from backend.profiling.quality_analyzer import QualityAnalyzer
from backend.profiling.distribution_analyzer import DistributionAnalyzer
from backend.profiling.relationship_analyzer import RelationshipAnalyzer
from backend.profiling.outlier_analyzer import OutlierAnalyzer
from backend.profiling.target_analyzer import TargetAnalyzer
from backend.profiling.resource_analyzer import ResourceAnalyzer
from backend.profiling.execution_hints import ExecutionHints
from backend.profiling.dataset_profiler import DatasetProfiler


class ProfilingEngine:
    """
    Main orchestrator for dataset profiling.
    Observes raw dataset files and generates structured DatasetProfile (and SemanticProfile) and ExecutionHints.
    """

    @classmethod
    def profile_file(
        cls,
        file_path: str,
        target_column: Optional[str] = None,
        user_mission: str = "",
        user_task_type: str = "general",
    ) -> Tuple[DatasetProfile, ExecutionHints]:
        """
        Profiles a dataset file (CSV/Parquet) and returns (DatasetProfile, ExecutionHints).
        Uses lazy sampling on large datasets to keep profiling fast and RAM-safe.
        """
        df, file_meta, is_sampled = DataLoader.load_lazy_sample(file_path)
        return cls.profile_dataframe(
            df, file_meta,
            target_column=target_column,
            user_mission=user_mission,
            user_task_type=user_task_type,
        )

    @classmethod
    def profile_bytes(
        cls,
        file_bytes: bytes,
        filename: str = "dataset.csv",
        target_column: Optional[str] = None,
        user_mission: str = "",
        user_task_type: str = "general",
    ) -> Tuple[DatasetProfile, ExecutionHints]:
        """
        Profiles in-memory dataset bytes (CSV/Parquet) directly without disk writes.
        """
        df, file_meta, is_sampled = DataLoader.load_lazy_sample_from_bytes(file_bytes, filename=filename)
        return cls.profile_dataframe(
            df, file_meta,
            target_column=target_column,
            user_mission=user_mission,
            user_task_type=user_task_type,
        )

    @classmethod
    def profile_dataframe(
        cls,
        df: pd.DataFrame,
        file_meta: Optional[Dict[str, Any]] = None,
        target_column: Optional[str] = None,
        user_mission: str = "",
        user_task_type: str = "general",
    ) -> Tuple[DatasetProfile, ExecutionHints]:
        """
        Profiles an in-memory DataFrame and returns (DatasetProfile, ExecutionHints).
        """
        file_meta = file_meta or {
            "filename": "in_memory_dataset",
            "file_size_bytes": int(df.memory_usage(deep=True).sum()),
            "row_count": len(df),
            "column_count": len(df.columns),
            "format": "dataframe",
        }

        # 1. Generate full unified DatasetProfile via DatasetProfiler
        dataset_profile = DatasetProfiler.profile_dataframe(
            df,
            target_column=target_column,
            user_mission=user_mission,
            user_task_type=user_task_type,
            file_meta=file_meta,
        )

        # 2. Extract resource hints for ExecutionHints return
        resource_prof, exec_hints = ResourceAnalyzer.analyze_resources(
            df, file_meta.get("file_size_bytes", 0)
        )

        # 3. Supplemental recommendation context
        dist_result = DistributionAnalyzer.analyze_distributions(df, dataset_profile.column_profiles)
        rel_result = RelationshipAnalyzer.analyze_relationships(df)
        outlier_result = OutlierAnalyzer.analyze_outliers(df)

        dataset_profile.recommendation_context = {
            "distributions": dist_result,
            "relationships": rel_result,
            "outliers": outlier_result,
        }

        return dataset_profile, exec_hints

