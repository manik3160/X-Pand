#!/usr/bin/env zsh
# src/data_fetchers/run_data_collection.sh
# ==========================================
# Master script to orchestrate the entire data collection pipeline
#
# Usage:
#   bash run_data_collection.sh                    # Uses defaults (Delhi, synthetic data)
#   bash run_data_collection.sh mumbai osm         # Mumbai with OSM data
#   bash run_data_collection.sh bangalore          # Bangalore with synthetic data

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REGION="${1:-delhi}"
METHOD="${2:-synthetic}"
BBOX_MAP=(
    ["delhi"]="76.8,28.4,77.4,28.9"
    ["mumbai"]="72.7,18.8,73.0,19.3"
    ["bangalore"]="77.4,12.8,77.7,13.1"
    ["hyderabad"]="78.2,17.2,78.6,17.5"
    ["kolkata"]="88.2,22.4,88.5,22.7"
)

BBOX="${BBOX_MAP[$REGION]:-76.8,28.4,77.4,28.9}"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          X-PAND DATA COLLECTION PIPELINE                   ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Region: $REGION"
echo "  Method: $METHOD"
echo "  Bounding Box: $BBOX"
echo ""

# Create directories
echo -e "${YELLOW}[1/5] Creating directories...${NC}"
mkdir -p data/raw
mkdir -p data/processed
mkdir -p logs
echo -e "${GREEN}✓ Directories created${NC}\n"

# Step 1: Build geospatial grid (if not exists)
if [ ! -f "data/processed/grid.geojson" ]; then
    echo -e "${YELLOW}[2/5] Building geospatial grid...${NC}"
    echo "Please run: jupyter notebook notebooks/01_grid_build.ipynb"
    echo "This creates the 500m × 500m hexagonal grid."
    exit 1
else
    echo -e "${YELLOW}[2/5] Grid already exists: data/processed/grid.geojson${NC}"
    echo -e "${GREEN}✓ Skipping grid creation${NC}\n"
fi

# Step 2: Fetch delivery zones
echo -e "${YELLOW}[3/5] Collecting delivery zones data...${NC}"
python src/data_fetchers/fetch_delivery_zones.py \
    --method "$METHOD" \
    --bbox "$BBOX" \
    --region "$REGION" \
    --grid-path "data/processed/grid.geojson" \
    --output "data/raw/delivery_zones.csv" \
    2>&1 | tee logs/delivery_zones_${REGION}_$(date +%Y%m%d_%H%M%S).log

echo -e "${GREEN}✓ Delivery zones collected${NC}\n"

# Step 3: Fetch demographics
echo -e "${YELLOW}[4/5] Collecting demographics data...${NC}"
python src/data_fetchers/fetch_census_demographics.py \
    --bbox "$BBOX" \
    --region "$REGION" \
    --grid-path "data/processed/grid.geojson" \
    --output "data/raw/demographics.csv" \
    2>&1 | tee logs/demographics_${REGION}_$(date +%Y%m%d_%H%M%S).log

echo -e "${GREEN}✓ Demographics collected${NC}\n"

# Step 4: Validate and merge
echo -e "${YELLOW}[5/5] Validating and merging datasets...${NC}"
python src/data_fetchers/validate_and_merge.py \
    --delivery-zones "data/raw/delivery_zones.csv" \
    --demographics "data/raw/demographics.csv" \
    --grid "data/processed/grid.geojson" \
    --output "data/processed/combined_features.geojson" \
    2>&1 | tee logs/merge_${REGION}_$(date +%Y%m%d_%H%M%S).log

echo -e "${GREEN}✓ Validation and merge complete${NC}\n"

# Summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    COLLECTION COMPLETE                     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Data files created:${NC}"
echo "  • data/raw/delivery_zones.csv"
echo "  • data/raw/demographics.csv"
echo "  • data/processed/combined_features.csv"
echo "  • data/processed/combined_features.geojson"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Review data quality in logs/"
echo "  2. Run: jupyter notebook notebooks/02_features.ipynb"
echo "  3. Train model: jupyter notebook notebooks/03_train.ipynb"
echo "  4. Evaluate: jupyter notebook notebooks/04_bip.ipynb"
echo ""
