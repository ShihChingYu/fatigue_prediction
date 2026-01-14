"""Jobs module."""

import typing as T

from fatigue.jobs.evaluation import EvaluationsJob
from fatigue.jobs.explanation import ExplanationsJob
from fatigue.jobs.inference import InferenceJob
from fatigue.jobs.training import TrainingJob
from fatigue.jobs.tuning import TuningJob

from fatigue.jobs.promotion import PromotionJob

JobKind = T.Union[
    TrainingJob, TuningJob, EvaluationsJob, InferenceJob, ExplanationsJob, PromotionJob
]
