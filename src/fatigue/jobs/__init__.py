"""Jobs module."""

import typing as T

from fatigue.jobs.evaluation import EvaluationsJob
from fatigue.jobs.explanation import ExplanationsJob
from fatigue.jobs.inference import InferenceJob
from fatigue.jobs.training import TrainingJob
from fatigue.jobs.tuning import TuningJob

from fatigue.jobs.promotion import PromotionJob

# Define a Union of all available jobs
# This is what allows the YAML parser to pick the right class based on "KIND"
JobKind = T.Union[
    TrainingJob, TuningJob, EvaluationsJob, InferenceJob, ExplanationsJob, PromotionJob
]
