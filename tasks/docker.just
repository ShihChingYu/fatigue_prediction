# tasks/docker.just

GCP_PROJECT := "fatigue-advice"
COMMIT_HASH := `git rev-parse --short HEAD`
DOCKER_REPO := "gcr.io/" + GCP_PROJECT + "/fatigue-advice"

# run docker tasks
[group('docker')]
docker: docker-build docker-run

# build docker image (automatically builds the package first)
[group('docker')]
docker-build tag=COMMIT_HASH: package-build
    @echo "Building image for commit: {{COMMIT_HASH}}"
    docker build --tag={{DOCKER_REPO}}:{{tag}} --tag={{DOCKER_REPO}}:latest .

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
