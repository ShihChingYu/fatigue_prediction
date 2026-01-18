import logging
import re
import typing as T
from pathlib import Path
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class ETLInputs(BaseModel):
    raw_path: str = "data/monitor"
    hr_pattern: str = "HR_*.csv"
    sleep_pattern: str = "sleep_*.csv"


class ETLOutputs(BaseModel):
    inputs_train: str = "data/processed/inputs_train.parquet"
    ids_train: str = "data/processed/ids_train.parquet"


class ETLSettings(BaseModel):
    target_col: str = "fatigue_score"
    hr_limits: Tuple[float, float] = (40.0, 180.0)
    sleep_limits: Tuple[float, float] = (0.25, 36.0)
    pvt_tolerance_minutes: int = 15
    window_size: str = "5min"
    min_periods: int = 30
    # Data Augmentation Offsets
    offsets: List[int] = [0, 1, 2, 3, 4]

    # PVT Settings
    pvt_artifact_threshold: float = 100.0  # Ignore taps < 100ms
    fatigue_threshold_low: float = 300.0  # Below this = 0.0 (Alert)
    fatigue_threshold_high: float = 400.0  # Above this = 1.0 (Fatigued)


class ETLSplit(BaseModel):
    test_size: float = 0.2
    random_state: int = 42


class ETLJob(BaseModel):
    KIND: Literal["ETLJob"] = "ETLJob"  # Set default here

    inputs: ETLInputs = Field(default_factory=ETLInputs)
    outputs: ETLOutputs = Field(default_factory=ETLOutputs)
    settings: ETLSettings = Field(default_factory=ETLSettings)
    split: ETLSplit = Field(default_factory=ETLSplit)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _get_user_ids(self, raw_path: Path):
        pattern_str = getattr(self.inputs, "pvt_pattern", "default_pattern.csv")
        files = list(raw_path.glob(pattern_str))
        regex_pattern = pattern_str.replace("*", r"(\d+)")
        ids = []
        for f in files:
            match = re.search(regex_pattern, f.name)
            if match:
                ids.append(match.group(1))
        unique_ids = sorted(list(set(ids)))
        log.info(f"Found {len(unique_ids)} users with PVT data.")
        return unique_ids

    def _clean_hr(self, file_path):
        df = pd.read_csv(file_path)
        df = df.rename(columns={c: c.upper() for c in df.columns})

        if "HRTIME" not in df.columns:
            raise KeyError(f"Column 'HRTIME' not found in {file_path.name}")

        df["HRTIME"] = pd.to_datetime(df["HRTIME"])
        df = df.sort_values("HRTIME")

        min_hr, max_hr = self.settings.hr_limits
        df = df.dropna(subset=["HR"])
        df = df[(df["HR"] > min_hr) & (df["HR"] < max_hr)]

        df = df.groupby("HRTIME", as_index=False)[["HR"]].mean()
        df = T.cast(pd.DataFrame, df)
        return df.sort_values("HRTIME")

    def _engineer_hr(self, df):
        df = df.set_index("HRTIME").sort_index()
        win = self.settings.window_size
        min_p = self.settings.min_periods

        df["mean_hr_5min"] = df["HR"].rolling(win, min_periods=min_p).mean()
        df["hr_volatility_5min"] = df["HR"].rolling(win, min_periods=min_p).std()

        df["hr_mean_total"] = df["HR"].expanding(min_periods=min_p).mean()
        df["hr_std_total"] = df["HR"].expanding(min_periods=min_p).std().fillna(1)
        df["hr_zscore"] = (df["HR"] - df["hr_mean_total"]) / (df["hr_std_total"] + 0.1)

        df["hr_diff"] = df["HR"].diff().abs()
        df["hr_jumpiness_5min"] = np.sqrt(
            (df["hr_diff"] ** 2).rolling(win, min_periods=min_p).mean()
        )

        df["stress_cv"] = df["hr_volatility_5min"] / (df["mean_hr_5min"] + 0.1)

        df = df.dropna(subset=["mean_hr_5min"])
        return df.reset_index() if not df.empty else None

    def _clean_sleep(self, file_path):
        df = pd.read_csv(file_path)
        df = df.rename(columns={c: c.upper() for c in df.columns})

        if "START" not in df.columns or "END" not in df.columns:
            raise KeyError(f"Columns 'START'/'END' not found in {file_path.name}")

        df["START"] = pd.to_datetime(df["START"])
        df["END"] = pd.to_datetime(df["END"])
        df = df.dropna(subset=["START", "END"]).drop_duplicates(subset=["START", "END"])

        df["duration"] = (df["END"] - df["START"]).dt.total_seconds() / 3600
        min_s, max_s = self.settings.sleep_limits
        df = df[(df["duration"] > min_s) & (df["duration"] < max_s)]
        return df.sort_values("END")

    def _engineer_sleep(self, df):
        df["sleep_debt"] = 8.0 - df["duration"]
        df["_grp"] = 1
        df["cum_sleep_debt"] = (
            df.groupby("_grp")["sleep_debt"]
            .rolling(3, min_periods=1)
            .sum()
            .reset_index(0, drop=True)
        )
        return df[["START", "END", "duration", "cum_sleep_debt"]]

    def _calculate_fatigue_score(self, rt):
        low = self.settings.fatigue_threshold_low
        high = self.settings.fatigue_threshold_high
        if rt < low:
            return 0.0
        elif rt < high:
            return (rt - low) / (high - low)
        else:
            return 1.0

    def _clean_pvt(self, file_path):
        df = pd.read_csv(file_path)
        cols = {c: c.upper() for c in df.columns}
        df = df.rename(columns=cols)

        if "TESTSTART" in df.columns:
            df = df.rename(columns={"TESTSTART": "timestamp"})
        elif "TIMESTAMP" in df.columns:
            df = df.rename(columns={"TIMESTAMP": "timestamp"})
        else:
            raise KeyError(f"Column 'TESTSTART' or 'TIMESTAMP' not found in {file_path.name}")

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        if "TAPTIME" in df.columns:
            df = df[df["TAPTIME"] > self.settings.pvt_artifact_threshold]

        group_cols = ["timestamp"]
        if "TESTID" in df.columns:
            group_cols.append("TESTID")

        if "TAPTIME" in df.columns:
            df = df.groupby(group_cols).agg({"TAPTIME": "mean"}).reset_index()
            df = df.rename(columns={"TAPTIME": "pvt_mean_rt"})
        else:
            found = False
            for c in ["RT_MEAN", "MEAN_RT", "PVT_MEAN_RT"]:
                if c in df.columns:
                    df = df.rename(columns={c: "pvt_mean_rt"})
                    found = True
                    break
            if not found:
                raise KeyError("PVT mean column not found")

        df["fatigue_score"] = df["pvt_mean_rt"].apply(self._calculate_fatigue_score)
        df = df.rename(columns={"timestamp": "TESTSTART"})
        return df.sort_values("TESTSTART")

    def _process_user(self, uid, raw_path):
        # ... (This is the original training logic, keeping it brief here) ...
        # You already have this logic in your previous working file.
        # It's okay to just declare the methods above and use them in InferenceETLJob
        pass

    def run(self):
        # Original training run logic
        pass


class InferenceETLJob(ETLJob):
    """
    Generates features for EVERY minute of available data,
    ignoring whether a PVT test exists.
    """

    KIND: T.Literal["InferenceETLJob"] = "InferenceETLJob"  # type: ignore[assignment]

    # Override Outputs to point to a new inference file
    outputs: ETLOutputs = Field(
        default_factory=lambda: ETLOutputs(
            inputs_train="data/processed/inputs_inference.parquet",
            ids_train="data/processed/ids_inference.parquet",
        )
    )

    def _get_user_ids(self, raw_path: Path):
        pattern_str = getattr(self.inputs, "hr_pattern", "default_pattern.csv")
        files = list(raw_path.glob(pattern_str))
        regex_pattern = pattern_str.replace("*", r"(\d+)")
        ids = []
        for f in files:
            match = re.search(regex_pattern, f.name)
            if match:
                ids.append(match.group(1))
        unique_ids = sorted(list(set(ids)))
        log.info(f"Found {len(unique_ids)} users with HR data for inference.")
        return unique_ids

    def _process_user(self, uid, raw_path):
        try:
            # Path definitions
            hr_name = self.inputs.hr_pattern.replace("*", uid)
            sleep_name = self.inputs.sleep_pattern.replace("*", uid)
            hr_path = raw_path / hr_name
            sleep_path = raw_path / sleep_name

            if not (hr_path.exists() and sleep_path.exists()):
                return None

            # Generate Continuous Grid (Features)
            df_hr = self._engineer_hr(self._clean_hr(hr_path))
            if df_hr is None:
                return None

            df_sleep = self._engineer_sleep(self._clean_sleep(sleep_path))
            df_sleep = df_sleep.rename(columns={"END": "last_sleep_end"})

            # Merge HR + Sleep
            df_continuous = pd.merge_asof(
                df_hr.sort_values("HRTIME"),
                df_sleep.sort_values("last_sleep_end"),
                left_on="HRTIME",
                right_on="last_sleep_end",
                direction="backward",
            )

            df_continuous["hours_awake"] = (
                df_continuous["HRTIME"] - df_continuous["last_sleep_end"]
            ).dt.total_seconds() / 3600

            # Impute Sleep
            if df_continuous["hours_awake"].isna().any():
                fill_val = df_continuous["hours_awake"].median()
                if pd.isna(fill_val):
                    fill_val = 14.0
                df_continuous["hours_awake"] = df_continuous["hours_awake"].fillna(fill_val)
                if "cum_sleep_debt" in df_continuous.columns:
                    df_continuous["cum_sleep_debt"] = df_continuous["cum_sleep_debt"].fillna(0.0)

            df_continuous["sleep_inertia_idx"] = 1 / (df_continuous["hours_awake"] + 0.1)
            hr_hour = df_continuous["HRTIME"].dt.hour + (df_continuous["HRTIME"].dt.minute / 60)
            df_continuous["circadian_sin"] = np.sin(2 * np.pi * hr_hour / 24)
            df_continuous["circadian_cos"] = np.cos(2 * np.pi * hr_hour / 24)

            # Resample Grid
            df_continuous = df_continuous.set_index("HRTIME").sort_index()
            df_continuous = df_continuous.resample("1min").mean(numeric_only=True).reset_index()

            # Clean up columns for the Schema
            required_cols = ["mean_hr_5min", "hours_awake"]

            # [CRITICAL] Drop forbidden columns (Same as training)
            drop_cols = [
                "TESTSTART",
                "timestamp",
                "pvt_mean_rt",
                "MATCH_TIME",
                "last_sleep_end",
                "TESTID",
                "START",
                "duration",
                "hr_diff",
                "HR",
                "fatigue_score",
            ]

            # Keep HRTIME for now so we know WHEN the prediction is for
            df_final_user = df_continuous.dropna(subset=required_cols).copy()
            df_final_user["user_id"] = uid

            # Filter columns
            cols_to_drop = [c for c in drop_cols if c in df_final_user.columns]
            df_final_user = df_final_user.drop(columns=cols_to_drop)

            return df_final_user

        except Exception as e:
            log.error(f"Error processing user {uid}: {e}")
            return None

    def run(self):
        """Main execution for Inference Generation"""
        raw_path = Path(self.inputs.raw_path)
        out_inputs = Path(self.outputs.inputs_train)
        out_ids = Path(self.outputs.ids_train)
        out_inputs.parent.mkdir(parents=True, exist_ok=True)

        user_ids = self._get_user_ids(raw_path)
        all_data = []

        for uid in user_ids:
            res = self._process_user(uid, raw_path)
            if res is not None:
                all_data.append(res)

        if not all_data:
            log.error("No valid data processed.")
            return

        df_full = pd.concat(all_data, ignore_index=True)
        log.info(f"Total inference samples generated: {df_full.shape}")

        ids_df = df_full[["user_id", "HRTIME"]]
        inputs_df = df_full.drop(columns=["user_id", "HRTIME"])

        ids_df.to_parquet(out_ids, index=False)
        inputs_df.to_parquet(out_inputs, index=False)

        log.info(f"Saved Features to {out_inputs}")
        log.info(f"Saved IDs to {out_ids}")
