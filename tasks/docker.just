# tasks/docker.just

GCP_PROJECT := "fatigue-advice"
GCP_REPO := "gcr.io/" + GCP_PROJECT + "/fatigue-advice"

# run docker tasks
[group('docker')]
docker: docker-build docker-run

# build docker image (automatically builds the package first)
[group('docker')]
docker-build tag="latest": package-build
    docker build --tag={{REPOSITORY}}:{{tag}} .

# start docker compose
[group('docker')]
docker-compose:
    docker compose up

# run latest docker image
[group('docker')]
docker-run tag="latest":
    docker run --rm {{REPOSITORY}}:{{tag}}
