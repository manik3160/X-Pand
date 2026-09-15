#!/bin/bash

# X-Pand Production Build Script
# ==============================
# Builds and pushes Docker image for production

set -e

# Configuration
DOCKER_REGISTRY="${DOCKER_REGISTRY:-your-registry.azurecr.io}"
IMAGE_NAME="${IMAGE_NAME:-xpand}"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
GIT_SHA=$(git rev-parse --short HEAD)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
VERSION="${VERSION:-1.0.0}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}=== X-Pand Docker Build ===${NC}"
echo "Registry: $DOCKER_REGISTRY"
echo "Image: $IMAGE_NAME:$VERSION"
echo "Git SHA: $GIT_SHA"
echo "Build Date: $BUILD_DATE"
echo ""

# Validation
if [ -z "$DOCKER_REGISTRY" ]; then
    echo -e "${RED}Error: DOCKER_REGISTRY not set${NC}"
    exit 1
fi

# Build image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build \
    --tag "$DOCKER_REGISTRY/$IMAGE_NAME:$VERSION" \
    --tag "$DOCKER_REGISTRY/$IMAGE_NAME:latest" \
    --tag "$DOCKER_REGISTRY/$IMAGE_NAME:$GIT_SHA" \
    --build-arg BUILD_DATE="$BUILD_DATE" \
    --build-arg VCS_REF="$GIT_SHA" \
    --build-arg VERSION="$VERSION" \
    -f Dockerfile \
    . 2>&1 | tail -20

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build successful${NC}"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

# Image size
IMAGESIZE=$(docker images --format "{{.Size}}" "$DOCKER_REGISTRY/$IMAGE_NAME:$VERSION")
echo "Image Size: $IMAGESIZE"
echo ""

# Scan for vulnerabilities (if available)
if command -v docker scan &> /dev/null; then
    echo -e "${YELLOW}Scanning image for vulnerabilities...${NC}"
    docker scan "$DOCKER_REGISTRY/$IMAGE_NAME:$VERSION" || true
    echo ""
fi

# Push to registry
if [ "$1" == "--push" ] || [ "$1" == "-p" ]; then
    echo -e "${YELLOW}Pushing image to registry...${NC}"
    
    docker push "$DOCKER_REGISTRY/$IMAGE_NAME:$VERSION"
    docker push "$DOCKER_REGISTRY/$IMAGE_NAME:latest"
    docker push "$DOCKER_REGISTRY/$IMAGE_NAME:$GIT_SHA"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Push successful${NC}"
    else
        echo -e "${RED}✗ Push failed${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}=== Build Complete ===${NC}"
echo ""
echo "Tags:"
echo "  - $DOCKER_REGISTRY/$IMAGE_NAME:$VERSION"
echo "  - $DOCKER_REGISTRY/$IMAGE_NAME:latest"
echo "  - $DOCKER_REGISTRY/$IMAGE_NAME:$GIT_SHA"
echo ""
echo "To push to registry: $0 --push"
echo ""
