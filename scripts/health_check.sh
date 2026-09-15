#!/bin/bash

# X-Pand Production Health Check Script
# =====================================
# Monitors production deployment health and performance

set -e

API_URL="${1:-http://localhost:8000}"
CHECK_INTERVAL="${2:-30}"
MAX_RETRIES="${3:-3}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Counters
CHECKS=0
FAILURES=0
RESPONSE_TIMES=()

echo "X-Pand Production Health Monitor"
echo "================================="
echo "API URL: $API_URL"
echo "Check Interval: ${CHECK_INTERVAL}s"
echo ""

# Function to check endpoint
check_endpoint() {
    local endpoint=$1
    local method=${2:-GET}
    local data=${3:-""}
    
    start_time=$(date +%s%N)
    
    if [ "$method" == "POST" ]; then
        response=$(curl -s -w "\n%{http_code}" -X POST \
            -H "Content-Type: application/json" \
            -d "$data" \
            "$API_URL$endpoint" 2>/dev/null || echo "\n500")
    else
        response=$(curl -s -w "\n%{http_code}" "$API_URL$endpoint" 2>/dev/null || echo "\n500")
    fi
    
    end_time=$(date +%s%N)
    response_time=$(( (end_time - start_time) / 1000000 ))  # Convert to ms
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    RESPONSE_TIMES+=($response_time)
    CHECKS=$((CHECKS + 1))
    
    if [ "$http_code" == "200" ]; then
        echo -e "${GREEN}✓${NC} $endpoint (${response_time}ms)"
        return 0
    else
        echo -e "${RED}✗${NC} $endpoint (HTTP $http_code, ${response_time}ms)"
        FAILURES=$((FAILURES + 1))
        return 1
    fi
}

# Function to display stats
show_stats() {
    if [ ${#RESPONSE_TIMES[@]} -gt 0 ]; then
        total=0
        for time in "${RESPONSE_TIMES[@]}"; do
            total=$((total + time))
        done
        avg=$((total / ${#RESPONSE_TIMES[@]}))
        echo -e "\nAverage Response Time: ${avg}ms"
    fi
    
    success=$((CHECKS - FAILURES))
    echo "Checks: $success/$CHECKS passed"
    
    if [ $FAILURES -gt 0 ]; then
        echo -e "${RED}Health Check FAILED${NC}"
        return 1
    else
        echo -e "${GREEN}Health Check PASSED${NC}"
        return 0
    fi
}

# Main loop
iteration=0
while true; do
    iteration=$((iteration + 1))
    CHECKS=0
    FAILURES=0
    RESPONSE_TIMES=()
    
    echo ""
    echo "===== Check #$iteration - $(date '+%Y-%m-%d %H:%M:%S') ====="
    
    # Perform health checks
    check_endpoint "/health" || true
    check_endpoint "/status" || true
    check_endpoint "/cities" || true
    
    # Test predict endpoint
    check_endpoint "/predict" "POST" \
        '{"city":"delhi","locations":[{"grid_id":"test","lat":28.7041,"lon":77.1025}]}' || true
    
    # Display stats
    show_stats
    
    # Exit on critical failure
    if [ $FAILURES -gt $MAX_RETRIES ]; then
        echo -e "\n${RED}Critical failures detected. Exiting.${NC}"
        exit 1
    fi
    
    echo -e "\nNext check in ${CHECK_INTERVAL}s... (Press Ctrl+C to stop)"
    sleep $CHECK_INTERVAL
done
