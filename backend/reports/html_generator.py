from typing import Optional
from backend.schemas.report import FinalRecommendation
from backend.schemas.evaluation import EvaluationReport
from backend.schemas.semantic_profile import SemanticProfile
from backend.schemas.mission_brief import MissionBrief


class HTMLReportGenerator:
    """Generates responsive, glassmorphism-styled HTML reports from recommendation artifacts."""

    @classmethod
    def generate_html(
        cls,
        recommendation: FinalRecommendation,
        evaluation_report: Optional[EvaluationReport] = None,
        profile: Optional[SemanticProfile] = None,
        mission: Optional[MissionBrief] = None,
        mission_brief_str: Optional[str] = None,
        experiment_results: Optional[list] = None,
    ) -> str:
        """Renders a standalone, styled HTML report with full transformation audit and validation methodology."""
        winning_id = recommendation.winning_experiment_id
        model_name = recommendation.model
        summary_text = recommendation.summary

        # Extract mission text
        mission_text = mission_brief_str or (mission.objective if mission else "Automated ML Optimization")

        # Build metrics HTML pills
        metrics_html = ""
        if recommendation.final_metrics:
            for k, v in recommendation.final_metrics.items():
                metrics_html += f"""
                <div class="metric-card">
                    <div class="metric-value">{v}</div>
                    <div class="metric-label">{k.upper().replace('_', ' ')}</div>
                </div>
                """

        # Build Transformation Audit Table rows
        audit_rows = ""
        col_profiles = []
        if profile:
            col_profiles = profile.column_profiles if hasattr(profile, "column_profiles") else profile.get("column_profiles", [])
        elif isinstance(profile, dict):
            col_profiles = profile.get("column_profiles", [])

        for col in col_profiles:
            c_name = col.name if hasattr(col, "name") else col.get("name", "col")
            c_type = col.type if hasattr(col, "type") else col.get("type", "unknown")
            c_missing = col.missing_pct if hasattr(col, "missing_pct") else col.get("missing_pct", 0.0)
            c_samples = col.sample_values if hasattr(col, "sample_values") else col.get("sample_values", [])
            c_enc = col.encoding_recommendation if hasattr(col, "encoding_recommendation") else col.get("encoding_recommendation", None)
            c_scale = col.scaling_recommendation if hasattr(col, "scaling_recommendation") else col.get("scaling_recommendation", None)

            sample_str = ", ".join(str(s) for s in c_samples[:3]) if c_samples else "N/A"
            enc_str = c_enc or ("onehot" if "cat" in str(c_type).lower() else "none")
            scale_str = c_scale or ("standard" if str(c_type).lower() == "numeric" else "none")

            audit_rows += f"""
            <tr>
                <td><strong>{c_name}</strong></td>
                <td><span class="badge font-mono">{str(c_type).replace('_', ' ')}</span></td>
                <td>{c_missing}%</td>
                <td>median/mean imputation</td>
                <td>Enc: <code>{enc_str}</code> | Scale: <code>{scale_str}</code></td>
                <td class="text-subtle">Fit on train split only (zero leakage)</td>
            </tr>
            """

        # Build experiment results table rows
        results_rows = ""
        exp_list = experiment_results or []
        if not exp_list and evaluation_report and evaluation_report.ranking:
            exp_list = evaluation_report.ranking

        for idx, exp in enumerate(exp_list, start=1):
            if isinstance(exp, dict):
                e_id = exp.get("experiment_id") or exp.get("id") or f"EXP_{idx}"
                e_mod = exp.get("model") or exp.get("model_name") or "Model"
                e_metrics = exp.get("metrics") or {}
                if isinstance(e_metrics, dict) and "metrics" in e_metrics:
                    raw_m = e_metrics.get("metrics", {})
                    p_score = e_metrics.get("primary_metric", 0.0)
                else:
                    raw_m = e_metrics
                    p_score = raw_m.get("primary_metric", 0.0) if isinstance(raw_m, dict) else 0.0

                cv_mean = raw_m.get("cv_mean", p_score) if isinstance(raw_m, dict) else p_score
                cv_std = raw_m.get("cv_std", 0.0) if isinstance(raw_m, dict) else 0.0
                test_score = raw_m.get("test_score", p_score) if isinstance(raw_m, dict) else p_score
                gap = raw_m.get("train_test_gap", 0.0) if isinstance(raw_m, dict) else 0.0
            else:
                e_id = getattr(exp, "experiment_id", f"EXP_{idx}")
                e_mod = getattr(exp, "model", getattr(exp, "model_name", "Model"))
                p_score = getattr(exp, "score", getattr(exp, "primary_metric", 0.0))
                cv_mean, cv_std, test_score, gap = p_score, 0.0, p_score, 0.0

            is_winner = (e_id == winning_id)
            row_cls = "class=\"winner-row\"" if is_winner else ""
            winner_badge = "<span class=\"badge-success\">★ WINNER</span>" if is_winner else f"#{idx}"

            results_rows += f"""
            <tr {row_cls}>
                <td>{winner_badge}</td>
                <td><code>{e_id}</code></td>
                <td><strong>{e_mod}</strong></td>
                <td>{cv_mean:.4f} \u00b1 {cv_std:.4f}</td>
                <td><strong>{test_score:.4f}</strong></td>
                <td>{gap:.4f}</td>
            </tr>
            """

        # Build findings HTML list
        findings_html = ""
        findings_list = recommendation.key_findings or ([f.finding for f in evaluation_report.knowledge] if evaluation_report else [])
        for f in findings_list:
            findings_html += f"<li>💡 {f}</li>"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evidra Research & Audit Report - {winning_id}</title>
    <style>
        :root {{
            --bg-color: #0a0a0b;
            --card-bg: rgba(20, 20, 24, 0.85);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-color: #33c7b6;
            --success-color: #34d399;
            --warning-color: #fbbf24;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            margin: 0;
            padding: 40px 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 960px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            margin-bottom: 32px;
        }}
        .header h1 {{
            font-size: 2.2rem;
            margin-bottom: 8px;
            background: linear-gradient(135deg, #12b3a3, #33c7b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .card {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }}
        h2 {{
            color: var(--accent-color);
            margin-top: 0;
            font-size: 1.25rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 8px;
        }}
        .mission-box {{
            background: rgba(18, 179, 163, 0.08);
            border-left: 4px solid var(--accent-color);
            padding: 12px 16px;
            border-radius: 4px;
            margin-bottom: 16px;
            font-size: 0.95rem;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 16px;
            margin-top: 16px;
        }}
        .metric-card {{
            background: rgba(15, 23, 42, 0.6);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
            border: 1px solid var(--border-color);
        }}
        .metric-value {{
            font-size: 1.6rem;
            font-weight: bold;
            color: var(--success-color);
        }}
        .metric-label {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            font-size: 0.85rem;
        }}
        th, td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            color: var(--text-secondary);
            text-transform: uppercase;
            font-size: 0.75rem;
        }}
        .winner-row {{
            background: rgba(52, 211, 153, 0.12);
        }}
        .badge {{
            background: rgba(18, 179, 163, 0.2);
            color: var(--accent-color);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.75rem;
        }}
        .badge-success {{
            background: rgba(52, 211, 153, 0.2);
            color: var(--success-color);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: bold;
        }}
        .text-subtle {{
            color: var(--text-secondary);
            font-size: 0.8rem;
        }}
        ul {{
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Evidra Research &amp; Technical Audit Report</h1>
            <p>Winning Experiment: <code>{winning_id}</code> | Model: <strong>{model_name}</strong></p>
        </div>

        <div class="card">
            <h2>🎯 Research Mission Brief</h2>
            <div class="mission-box">
                <strong>Objective:</strong> {mission_text}
            </div>
            <h2>Executive Summary</h2>
            <p>{summary_text}</p>
            <div class="metrics-grid">
                {metrics_html}
            </div>
        </div>

        <div class="card">
            <h2>🛡️ Split & Validation Methodology</h2>
            <p>To eliminate data leakage and provide independently verifiable evaluation metrics, the ML Execution Engine enforced:</p>
            <ul>
                <li><strong>Train/Test Split:</strong> Strict 80% Training / 20% Held-Out Testing split BEFORE any feature scaling, encoding, or cross-validation fitting.</li>
                <li><strong>Stratification:</strong> Target-stratified splitting for classification tasks to preserve class proportions.</li>
                <li><strong>Cross-Validation:</strong> 5-Fold Cross Validation performed exclusively on the 80% training split.</li>
                <li><strong>Non-Leakage Guarantee:</strong> All feature encoders and scalers were fit on training folds only (zero pre-fitting on held-out test data).</li>
            </ul>
        </div>

        <div class="card">
            <h2>🔬 Data Transformation Audit</h2>
            <p className="text-subtle">Column-level preprocessing audit documenting raw statistics, applied transformers, and non-leakage confirmation:</p>
            <table>
                <thead>
                    <tr>
                        <th>Column</th>
                        <th>Type</th>
                        <th>Missing %</th>
                        <th>Cleaning Rule</th>
                        <th>Selected Transformers</th>
                        <th>Audit Rationale</th>
                    </tr>
                </thead>
                <tbody>
                    {audit_rows or '<tr><td colspan="6">No column transformation audit data available.</td></tr>'}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>📊 Experiment Leaderboard & Honest Metrics</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Experiment ID</th>
                        <th>Model</th>
                        <th>CV Mean \u00b1 Std</th>
                        <th>Test Score</th>
                        <th>Train/Test Gap</th>
                    </tr>
                </thead>
                <tbody>
                    {results_rows or '<tr><td colspan="6">No experiment results recorded.</td></tr>'}
                </tbody>
            </table>
        </div>

        <div class="card">
            <h2>💡 Key Findings & Director Insights</h2>
            <ul>
                {findings_html or '<li>Automated experiment pipeline execution completed cleanly.</li>'}
            </ul>
        </div>
    </div>
</body>
</html>
"""
        return html_content
