#!/usr/bin/env bash
set -ex

echo $NQ_GITHUB_TOKEN | docker login ghcr.io -u nordquant --password-stdin
docker buildx build --platform linux/amd64,linux/arm64 -t ghcr.io/nordquant/ai-agents-crash-course-codespace:latest-release . --push

