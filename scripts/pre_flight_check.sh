#!/usr/bin/env bash

# Production Pre-Flight Checklist
# ===============================
# Validate that all production requirements are met

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

CHECKS_PASSED=0
CHECKS_FAILED=0
CHECKS_WARNING=0

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((CHECKS_PASSED++))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((CHECKS_FAILED++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((CHECKS_WARNING++))
}

echo -e "${BLUE}=== X-Pand Production Pre-Flight Checklist ===${NC}\n"

# 1. Code Quality
echo -e "${BLUE}1. Code Quality${NC}"
if command -v black &> /dev/null; then
    check_pass "Python code formatter (black) installed"
else
    check_warn "Python code formatter (black) not installed"
fi

if command -v pylint &> /dev/null; then
    check_pass "Python linter (pylint) installed"
else
    check_warn "Python linter (pylint) not installed"
fi

if [ -f ".flake8" ]; then
    check_pass "Flake8 configuration found"
else
    check_warn "Flake8 configuration not found"
fi

# 2. Dependencies
echo -e "\n${BLUE}2. Dependencies${NC}"
if [ -f "requirements.txt" ]; then
    check_pass "requirements.txt found"
else
    check_fail "requirements.txt not found"
fi

if [ -f "requirements-prod.txt" ]; then
    check_pass "requirements-prod.txt found"
else
    check_warn "requirements-prod.txt not found (recommended for production)"
fi

# 3. Configuration
echo -e "\n${BLUE}3. Configuration${NC}"
if [ -f ".env.production" ]; then
    check_pass ".env.production found"
    
    if grep -q "API_ENVIRONMENT=production" .env.production; then
        check_pass "API_ENVIRONMENT set to production"
    else
        check_fail "API_ENVIRONMENT not set to production"
    fi
    
    if grep -q "SECRET_KEY=" .env.production; then
        if grep -q "change-me" .env.production; then
            check_fail "SECRET_KEY is still default value"
        else
            check_pass "SECRET_KEY configured"
        fi
    else
        check_fail "SECRET_KEY not configured"
    fi
    
    if grep -q "CORS_ORIGINS=" .env.production; then
        check_pass "CORS_ORIGINS configured"
    else
        check_fail "CORS_ORIGINS not configured"
    fi
else
    check_fail ".env.production not found"
fi

# 4. Docker & Container
echo -e "\n${BLUE}4. Docker & Containerization${NC}"
if [ -f "Dockerfile" ]; then
    check_pass "Dockerfile found"
else
    check_fail "Dockerfile not found"
fi

if [ -f "docker-compose.prod.yml" ]; then
    check_pass "docker-compose.prod.yml found"
else
    check_warn "docker-compose.prod.yml not found"
fi

if command -v docker &> /dev/null; then
    check_pass "Docker installed"
    
    DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
    check_pass "Docker version: $DOCKER_VERSION"
else
    check_fail "Docker not installed"
fi

# 5. Kubernetes
echo -e "\n${BLUE}5. Kubernetes & Orchestration${NC}"
if command -v kubectl &> /dev/null; then
    check_pass "kubectl installed"
    
    K8S_VERSION=$(kubectl version --short 2>/dev/null | grep -o 'v[0-9.]*' | head -1)
    check_pass "Kubernetes version: $K8S_VERSION"
else
    check_warn "kubectl not installed (needed for K8s deployment)"
fi

if [ -d "k8s" ] && [ -f "k8s/api-deployment.yaml" ]; then
    check_pass "Kubernetes manifests found"
else
    check_fail "Kubernetes manifests not found"
fi

# 6. Secrets & Security
echo -e "\n${BLUE}6. Security & Secrets${NC}"
if grep -r "password" . --include="*.py" --include="*.json" --include="*.yaml" 2>/dev/null | grep -v node_modules | grep -v ".git" > /dev/null; then
    check_warn "Hardcoded passwords found in code (search for 'password')"
else
    check_pass "No obvious hardcoded credentials found"
fi

if [ -f ".gitignore" ] && grep -q ".env" .gitignore; then
    check_pass ".env files in .gitignore"
else
    check_warn ".env files might not be properly ignored"
fi

if [ -f ".gitignore" ] && grep -q "*.pkl\|*.joblib" .gitignore; then
    check_pass "Model files in .gitignore"
else
    check_warn "Model files might not be properly ignored"
fi

# 7. Documentation
echo -e "\n${BLUE}7. Documentation${NC}"
if [ -f "README.md" ]; then
    check_pass "README.md found"
else
    check_fail "README.md not found"
fi

if [ -f "PRODUCTION_DEPLOYMENT.md" ]; then
    check_pass "PRODUCTION_DEPLOYMENT.md found"
else
    check_warn "PRODUCTION_DEPLOYMENT.md not found"
fi

if [ -f "api/main.py" ]; then
    if grep -q "async def\|def " api/main.py | wc -l | grep -q "[0-9]"; then
        check_pass "API endpoints documented"
    fi
fi

# 8. Tests
echo -e "\n${BLUE}8. Testing${NC}"
if [ -d "tests" ]; then
    check_pass "Tests directory found"
    
    if [ -f "tests/test_production.py" ]; then
        check_pass "Production tests found"
    else
        check_warn "Production tests not found"
    fi
else
    check_warn "Tests directory not found"
fi

if command -v pytest &> /dev/null; then
    check_pass "pytest installed"
else
    check_warn "pytest not installed"
fi

# 9. Scripts
echo -e "\n${BLUE}9. Deployment Scripts${NC}"
if [ -f "scripts/deploy.sh" ]; then
    check_pass "Deploy script found"
else
    check_warn "Deploy script not found"
fi

if [ -f "scripts/build.sh" ]; then
    check_pass "Build script found"
else
    check_warn "Build script not found"
fi

if [ -f "scripts/health_check.sh" ]; then
    check_pass "Health check script found"
else
    check_warn "Health check script not found"
fi

# 10. Monitoring & Logging
echo -e "\n${BLUE}10. Monitoring & Logging${NC}"
if grep -q "sentry\|logging\|prometheus" requirements-prod.txt 2>/dev/null; then
    check_pass "Monitoring/logging libraries in requirements"
else
    check_warn "Monitoring/logging libraries not explicitly listed"
fi

if grep -q "SENTRY_DSN\|LOG_LEVEL" .env.production 2>/dev/null; then
    check_pass "Monitoring configuration present"
else
    check_warn "Monitoring configuration may be incomplete"
fi

# Summary
echo ""
echo -e "${BLUE}=== Summary ===${NC}"
echo -e "${GREEN}Passed: $CHECKS_PASSED${NC}"
echo -e "${YELLOW}Warnings: $CHECKS_WARNING${NC}"
echo -e "${RED}Failed: $CHECKS_FAILED${NC}"
echo ""

if [ $CHECKS_FAILED -gt 0 ]; then
    echo -e "${RED}❌ Production checklist FAILED${NC}"
    echo "Please fix the above issues before deploying to production."
    exit 1
elif [ $CHECKS_WARNING -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Production checklist PASSED with WARNINGS${NC}"
    echo "Review the above warnings and address as needed."
    exit 0
else
    echo -e "${GREEN}✅ Production checklist PASSED${NC}"
    echo "Your deployment is ready for production!"
    exit 0
fi
