#!/usr/bin/env bash

# X-Pand Complete Production Setup Script
# ========================================
# Automates the entire production setup process

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
VERSION="${1:-1.0.0}"
ENVIRONMENT="${2:-production}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-your-registry.azurecr.io}"

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     X-Pand Production Setup Automation Script      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Pre-flight checks
log_info "Running pre-flight checks..."
if [ -f "scripts/pre_flight_check.sh" ]; then
    chmod +x scripts/pre_flight_check.sh
    if ./scripts/pre_flight_check.sh; then
        log_success "Pre-flight checks passed"
    else
        log_warning "Pre-flight checks completed with warnings"
    fi
else
    log_warning "Pre-flight check script not found, skipping"
fi

echo ""

# Step 2: Install dependencies
log_info "Installing Python dependencies..."
if command -v pip &> /dev/null; then
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    log_success "Dependencies installed"
else
    log_error "pip not found. Please install Python 3.11+"
    exit 1
fi

echo ""

# Step 3: Environment configuration
log_info "Configuring production environment..."
if [ ! -f ".env.production" ]; then
    log_error ".env.production not found"
    exit 1
fi

# Validate critical variables
CRITICAL_VARS=("API_ENVIRONMENT" "SECRET_KEY" "CORS_ORIGINS")
for var in "${CRITICAL_VARS[@]}"; do
    if ! grep -q "$var" .env.production; then
        log_error "Missing critical variable: $var in .env.production"
        exit 1
    fi
done

log_success "Environment configuration validated"

echo ""

# Step 4: Build Docker image
log_info "Building Docker image..."
if command -v docker &> /dev/null; then
    chmod +x scripts/build.sh
    
    if ./scripts/build.sh; then
        log_success "Docker image built successfully"
        log_info "Docker image: $DOCKER_REGISTRY/xpand:$VERSION"
    else
        log_error "Docker build failed"
        exit 1
    fi
else
    log_warning "Docker not installed. Skipping Docker build."
    log_info "Install Docker to build and deploy containers"
fi

echo ""

# Step 5: Run tests
log_info "Running production tests..."
if command -v pytest &> /dev/null; then
    if pytest tests/ -v --tb=short; then
        log_success "All tests passed"
    else
        log_warning "Some tests failed, review results above"
    fi
else
    log_warning "pytest not installed. Skipping tests."
    log_info "Install pytest to run automated tests: pip install pytest"
fi

echo ""

# Step 6: Generate documentation
log_info "Generating documentation..."
if [ -f "PRODUCTION_README.md" ]; then
    log_success "Production documentation found"
else
    log_warning "Production documentation not found"
fi

echo ""

# Step 7: Setup checklist
log_info "Pre-deployment checklist:"
echo ""
echo "  ☐ Review and update .env.production with your settings"
echo "  ☐ Ensure SECRET_KEY is changed from default"
echo "  ☐ Configure CORS_ORIGINS for your domain"
echo "  ☐ Set SENTRY_DSN for error tracking (optional)"
echo "  ☐ Review resource limits in k8s/api-deployment.yaml"
echo "  ☐ Configure your Docker registry"
echo "  ☐ Push Docker image to your registry (--push flag)"
echo "  ☐ Set up Kubernetes cluster (if using K8s)"
echo "  ☐ Configure namespace and secrets"
echo "  ☐ Deploy to staging first for validation"
echo ""

# Step 8: Next steps
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
log_info "Next steps:"
echo ""
echo "1. Deploy with Docker Compose (for staging):"
echo "   docker-compose -f docker-compose.prod.yml up -d"
echo ""
echo "2. Deploy to Kubernetes (for production):"
echo "   ./scripts/deploy.sh production $VERSION"
echo ""
echo "3. Verify deployment:"
echo "   ./scripts/health_check.sh http://your-api-url"
echo ""
echo "4. Monitor logs:"
echo "   kubectl logs -f deployment/xpand-api"
echo ""
log_success "Production setup complete!"
echo ""
echo "📚 Documentation:"
echo "  - PRODUCTION_README.md - Production deployment guide"
echo "  - PRODUCTION_DEPLOYMENT.md - Detailed deployment instructions"
echo "  - API docs: http://api-url/docs (in development)"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
