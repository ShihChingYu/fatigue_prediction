"""Define settings for the application."""

# %% IMPORTS

import pydantic as pdt
import pydantic_settings as pdts

from fatigue import jobs

# %% SETTINGS


class Settings(pdts.BaseSettings, strict=True, frozen=True, extra="forbid"):
    """Base class for application settings.

    Use settings to provide high-level preferences.
    i.e., to separate settings from provider (e.g., CLI).
    """


class MainSettings(Settings):
    """Main settings of the application.

    This class corresponds to the structure of your YAML file.

    Parameters:
        job (jobs.JobKind): The specific job configuration to run.
    """

    job: jobs.JobKind = pdt.Field(..., discriminator="KIND")
