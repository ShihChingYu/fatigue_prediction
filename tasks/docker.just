# tasks/docker.just

GCP_PROJECT := "fatigue-advice"
COMMIT_HASH := `git rev-parse --short HEAD`
REPOSITORY := "gcr.io/" + GCP_PROJECT + "/fatigue-advice"

# run docker tasks
[group('docker')]
docker: docker-build docker-run

# build docker image (automatically builds the package first)
[group('docker')]
docker-build tag=COMMIT_HASH: package-build
    @echo "Building image for commit: {{COMMIT_HASH}}"
    docker build --tag={{REPOSITORY}}:{{tag}} --tag={{REPOSITORY}}:latest .

# push images to GCP Artifact Registry
[group('docker')]
docker-push tag=COMMIT_HASH:
    docker push {{REPOSITORY}}:{{tag}}
    docker push {{REPOSITORY}}:latest

# start docker compose
[group('docker')]
docker-compose:
    docker compose up

# run latest docker image
[group('docker')]
docker-run tag="latest":
    docker run --rm {{REPOSITORY}}:{{tag}}
