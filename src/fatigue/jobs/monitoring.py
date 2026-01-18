"""Monitoring job (Plotly - Crash-Free & Interactive)."""

import typing as T
from pathlib import Path

import mlflow
import pandas as pd
import plotly.graph_objects as go
import pydantic as pdt
from scipy import stats
from sklearn.preprocessing import RobustScaler

# Internal Imports
from fatigue.io import datasets, services
from fatigue.jobs import base


class MonitoringJob(base.Job):
    """Job to compare reference data vs current data for drift using Plotly."""

    KIND: T.Literal["MonitoringJob"] = "MonitoringJob"
    model_version_used: str = "Unknown"

    model_config = {"arbitrary_types_allowed": True}

    # Inputs
    reference_data: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    current_data: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")

    # Configuration
    output_path: str = "data/monitor/drift_report.html"
    numerical_features: T.Optional[T.List[str]] = [
        "mean_hr_5min",
        "hr_volatility_5min",
        "hours_awake",
        "stress_cv",
    ]
    ignored_columns: T.List[str] = ["user_id", "timestamp", "event_id", "HRTIME"]

    # Services
    mlflow_service: services.MlflowService
    run_config: services.MlflowService.RunConfig = services.MlflowService.RunConfig(
        name="Feature Drift Monitoring V2"
    )

    def run(self) -> T.Dict[str, T.Any]:
        logger = self.logger_service.logger()
        with self.mlflow_service.run_context(self.run_config) as run:
            print(f"--- Starting Monitoring Job (Run ID: {run.info.run_id}) ---")

            # 1. Load Data
            ref_df = self.reference_data.read()
            curr_df = self.current_data.read()

            logger.info(f"Loaded Reference: {ref_df.shape}, Current: {curr_df.shape}")

            # Track Data Lineage in Azure
            logger.info("Logging data lineage to Azure...")
            mlflow.log_input(
                self.reference_data.lineage(name="reference_training_data", data=ref_df),
                context="reference",
            )
            mlflow.log_input(
                self.current_data.lineage(name="current_inference_data", data=curr_df),
                context="current",
            )

            # 2. Select Columns
            if self.numerical_features:
                monitor_cols = [
                    c
                    for c in self.numerical_features
                    if c in ref_df.columns and c in curr_df.columns
                ]
            else:
                monitor_cols = [
                    c
                    for c in ref_df.columns
                    if c not in self.ignored_columns and pd.api.types.is_numeric_dtype(ref_df[c])
                ]

            # 3. Clean Data
            for col in monitor_cols:
                ref_df[col] = pd.to_numeric(ref_df[col], errors="coerce")
                curr_df[col] = pd.to_numeric(curr_df[col], errors="coerce")

            ref_clean = ref_df[monitor_cols].dropna()
            curr_clean = curr_df[monitor_cols].dropna()

            if ref_df.empty or curr_df.empty:
                print("CRITICAL: Dataframe is empty.")
                return {"drift_passed": True}

            # 3.5. APPLY SCALING (NEW STEP)
            print(f"--- Scaling Data (RobustScaler) for {len(monitor_cols)} features ---")

            # Scale Reference to be centered at 0
            scaler_ref = RobustScaler()
            ref_scaled_data = scaler_ref.fit_transform(ref_clean)
            ref_scaled = pd.DataFrame(ref_scaled_data, columns=monitor_cols)

            # Scale Current to be centered at 0 (ignoring absolute shift)
            scaler_curr = RobustScaler()
            curr_scaled_data = scaler_curr.fit_transform(curr_clean)
            curr_scaled = pd.DataFrame(curr_scaled_data, columns=monitor_cols)

            # 4. Calculate Drift & Generate Plots
            print("--- Calculating Drift & Generating Plots (Plotly) ---")

            drift_results = {}
            drifted_cols = []
            plots_html = ""

            for col in monitor_cols:
                # KS-Test
                stat, p_value = stats.ks_2samp(ref_scaled[col], curr_scaled[col])
                is_drifted = p_value < 0.05

                if is_drifted:
                    drifted_cols.append(col)

                drift_results[col] = {
                    "p_value": float(p_value),
                    "statistic": float(stat),
                    "drift_detected": bool(is_drifted),
                }

                # --- PLOTLY VISUALIZATION (Safe) ---
                fig = go.Figure()

                # Reference Histogram
                fig.add_trace(
                    go.Histogram(
                        x=ref_scaled[col],
                        name="Reference",
                        opacity=0.6,
                        marker_color="blue",
                        histnorm="probability density",
                    )
                )

                # Current Histogram
                fig.add_trace(
                    go.Histogram(
                        x=curr_scaled[col],
                        name="Current",
                        opacity=0.6,
                        marker_color="red",
                        histnorm="probability density",
                    )
                )

                status_color = "red" if is_drifted else "green"
                status_text = "DRIFT DETECTED" if is_drifted else "Stable"

                fig.update_layout(
                    title=f"<b>{col}</b>: <span style='color:{status_color}'>{status_text}</span> (p={p_value:.4f})",
                    barmode="overlay",
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
                    plot_bgcolor="rgba(240,240,240,0.5)",
                )

                # Convert single plot to HTML div (not full page)
                plots_html += fig.to_html(full_html=False, include_plotlyjs="cdn")

            # 5. Build Final HTML Report
            n_features = len(monitor_cols)
            drift_share = len(drifted_cols) / n_features if n_features > 0 else 0.0
            test_passed = drift_share < 0.3

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Fatigue Drift Report</title>
                <style>
                    body {{ font-family: sans-serif; margin: 40px; background-color: #fafafa; }}
                    h1 {{ color: #333; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
                    .summary {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 30px; }}
                    .pass {{ color: green; font-weight: bold; }}
                    .fail {{ color: red; font-weight: bold; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; background: #fff; }}
                    th, td {{ padding: 12px; border: 1px solid #ddd; text-align: left; }}
                    th {{ background-color: #f4f4f4; }}
                    .drift-yes {{ background-color: #ffebee; color: #c62828; }}
                </style>
            </head>
            <body>
                <h1>Drift Report (Plotly)</h1>

                <div class="summary">
                    <p><strong>Status:</strong> <span class="{"pass" if test_passed else "fail"}">
                        {"PASSED" if test_passed else "FAILED"}
                    </span></p>
                    <p><strong>Drift Share:</strong> {drift_share:.1%} ({len(drifted_cols)}/{n_features} features)</p>
                </div>

                <h2>Metrics</h2>
                <table>
                    <thead>
                        <tr><th>Feature</th><th>Status</th><th>P-Value</th></tr>
                    </thead>
                    <tbody>
            """

            for col, res in drift_results.items():
                row_cls = "drift-yes" if res["drift_detected"] else ""
                status = "DRIFT" if res["drift_detected"] else "Stable"
                html_content += f"<tr class='{row_cls}'><td>{col}</td><td>{status}</td><td>{res['p_value']:.5f}</td></tr>"

            html_content += f"""
                    </tbody>
                </table>

                <h2>Distributions (Interactive)</h2>
                {plots_html}
            </body>
            </html>
            """

            # Save
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            print(f"Interactive Report saved: {self.output_path}")

            # Log to Azure
            mlflow.log_artifact(self.output_path)
            mlflow.log_metric("drift_share", drift_share)
            mlflow.log_param("drift_passed", str(test_passed))

            return {
                "drift_passed": test_passed,
                "report_path": self.output_path,
                "drift_share": drift_share,
                "run_id": run.info.run_id,
            }
