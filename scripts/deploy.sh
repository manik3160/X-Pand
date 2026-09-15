#!/bin/bash

# Production Deployment Script for X-Pand
# ========================================
# This script automates the deployment of X-Pand to production

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT="${1:-production}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-your-registry.azurecr.io}"
IMAGE_NAME="xpand"
IMAGE_TAG="${2:-latest}"
NAMESPACE="default"

echo -e "${YELLOW}=== X-Pand Production Deployment ===${NC}"
echo "Environment: $ENVIRONMENT"
echo "Image: $DOCKER_REGISTRY/$IMAGE_NAME:$IMAGE_TAG"
echo "Namespace: $NAMESPACE"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"
for cmd in docker kubectl git; do
    if ! command_exists "$cmd"; then
        echo -e "${RED}Error: $cmd is not installed${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✓ All prerequisites met${NC}"
echo ""

# Build Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build \
    --tag "$DOCKER_REGISTRY/$IMAGE_NAME:$IMAGE_TAG" \
    --tag "$DOCKER_REGISTRY/$IMAGE_NAME:latest" \
    --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
    --build-arg VCS_REF=$(git rev-parse --short HEAD) \
    -f Dockerfile \
    .
echo -e "${GREEN}✓ Docker image built${NC}"
echo ""

# Push to registry
echo -e "${YELLOW}Pushing image to registry...${NC}"
docker push "$DOCKER_REGISTRY/$IMAGE_NAME:$IMAGE_TAG"
docker push "$DOCKER_REGISTRY/$IMAGE_NAME:latest"
echo -e "${GREEN}✓ Image pushed to registry${NC}"
echo ""

# Validate Kubernetes manifests
echo -e "${YELLOW}Validating Kubernetes manifests...${NC}"
for manifest in k8s/*.yaml; do
    kubectl apply -f "$manifest" --dry-run=client -o yaml > /dev/null
    echo "✓ $manifest"
done
echo -e "${GREEN}✓ All manifests are valid${NC}"
echo ""

# Apply Kubernetes manifests
echo -e "${YELLOW}Applying Kubernetes manifests...${NC}"
kubectl apply -f k8s/
echo -e "${GREEN}✓ Manifests applied${NC}"
echo ""

# Update image in deployment
echo -e "${YELLOW}Updating deployment image...${NC}"
kubectl set image deployment/xpand-api \
    api="$DOCKER_REGISTRY/$IMAGE_NAME:$IMAGE_TAG" \
    -n "$NAMESPACE"
echo -e "${GREEN}✓ Deployment updated${NC}"
echo ""

# Wait for rollout
echo -e "${YELLOW}Waiting for rollout to complete...${NC}"
kubectl rollout status deployment/xpand-api -n "$NAMESPACE" --timeout=10m
echo -e "${GREEN}✓ Rollout complete${NC}"
echo ""

# Verify deployment
echo -e "${YELLOW}Verifying deployment...${NC}"
READY_REPLICAS=$(kubectl get deployment xpand-api -n "$NAMESPACE" -o jsonpath='{.status.readyReplicas}')
DESIRED_REPLICAS=$(kubectl get deployment xpand-api -n "$NAMESPACE" -o jsonpath='{.spec.replicas}')

if [ "$READY_REPLICAS" -eq "$DESIRED_REPLICAS" ]; then
    echo -e "${GREEN}✓ All replicas are ready ($READY_REPLICAS/$DESIRED_REPLICAS)${NC}"
else
    echo -e "${YELLOW}Warning: Not all replicas are ready ($READY_REPLICAS/$DESIRED_REPLICAS)${NC}"
fi
echo ""

# Display pod status
echo -e "${YELLOW}Pod Status:${NC}"
kubectl get pods -n "$NAMESPACE" -l app=xpand-api
echo ""

# Display service info
echo -e "${YELLOW}Service Info:${NC}"
kubectl get svc xpand-api -n "$NAMESPACE"
echo ""

echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Monitor logs: kubectl logs -f deployment/xpand-api -n $NAMESPACE"
echo "2. Watch pods: kubectl get pods -n $NAMESPACE -l app=xpand-api -w"
echo "3. Port forward: kubectl port-forward svc/xpand-api 8000:8000 -n $NAMESPACE"
echo ""
