# tasks/docker.just

GCP_PROJECT := "fatigue-advice"
LOCATION    := "us-central1"
REPO_NAME   := "cloud-run-source-deploy"
IMAGE_NAME  := "fatigue-advice"

# Artifact Registry format
COMMIT_HASH := `git rev-parse --short HEAD`
DOCKER_REPO := LOCATION + "-docker.pkg.dev/" + GCP_PROJECT + "/" + REPO_NAME + "/" + IMAGE_NAME

# run docker tasks
[group('docker')]
docker: docker-build docker-run

# build docker image (automatically builds the package first)
[group('docker')]
docker-build tag=COMMIT_HASH: package-build
    @echo "Building image for commit: {{COMMIT_HASH}} (Target: linux/amd64)"
    docker build --platform=linux/amd64 --tag={{DOCKER_REPO}}:{{tag}} --tag={{DOCKER_REPO}}:latest .

# push images to GCP Artifact Registry
[group('docker')]
docker-push tag=COMMIT_HASH:
    docker push {{DOCKER_REPO}}:{{tag}}
    docker push {{DOCKER_REPO}}:latest

# start docker compose
[group('docker')]
docker-compose:
    docker compose up

# run latest docker image
[group('docker')]
docker-run tag="latest":
    docker run --rm {{DOCKER_REPO}}:{{tag}}

[group('docker')]
deploy tag=COMMIT_HASH:
    gcloud run deploy fatigue-service \
      --image={{DOCKER_REPO}}:{{tag}} \
      --region={{LOCATION}} \
      --platform=managed \
      --allow-unauthenticated \
      --memory=4Gi \
      --timeout=900 \
      --cpu=2 \
      --no-cpu-throttling \
      --execution-environment=gen2 \
      --set-env-vars GROQ_API_KEY={{env_var('GROQ_API_KEY')}}
