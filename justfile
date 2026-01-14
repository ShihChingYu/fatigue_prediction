# REQUIRES
# Checks if these tools exist in your shell.
docker := require("docker")
git := require("git")
uv := require("uv")

# SETTINGS
set dotenv-load := true

# VARIABLES
# This MUST match src/fatigue
PACKAGE := "fatigue"
# GitHub repo name or Docker image name
REPOSITORY := "fatigue"
SOURCES := "src"
TESTS := "tests"

# DEFAULTS
default:
    @just --list

# IMPORTS
import 'tasks/check.just'
import 'tasks/clean.just'
import 'tasks/commit.just'
import 'tasks/doc.just'
import 'tasks/docker.just'
import 'tasks/format.just'
import 'tasks/install.just'
import 'tasks/mlflow.just'
import 'tasks/project.just'
import 'tasks/package.just'

project-check: check-format check-code check-type check-security
