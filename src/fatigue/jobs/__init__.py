"""Jobs module."""

import typing as T

from fatigue.jobs.etl import ETLJob
from fatigue.jobs.evaluation import EvaluationsJob
from fatigue.jobs.explanation import ExplanationsJob
from fatigue.jobs.inference import InferenceJob
from fatigue.jobs.inference_etl import InferenceETLJob
from fatigue.jobs.monitoring import MonitoringJob
from fatigue.jobs.promotion import PromotionJob
from fatigue.jobs.training import TrainingJob
from fatigue.jobs.tuning import TuningJob

JobKind = T.Union[
    TrainingJob,
    TuningJob,
    EvaluationsJob,
    InferenceJob,
    ExplanationsJob,
    PromotionJob,
    MonitoringJob,
    ETLJob,
    InferenceETLJob,
]
