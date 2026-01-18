"""Define settings for the application."""

# %% IMPORTS
import pydantic_settings as pdts

# %% SETTINGS


class Settings(pdts.BaseSettings):
    """Base class for application settings.

    Use settings to provide high-level preferences.
    i.e., to separate settings from provider (e.g., CLI).
    """

    model_config = pdts.SettingsConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
    )
