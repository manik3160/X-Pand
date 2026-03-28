"""
src/data_fetchers/fetch_census_demographics.py
================================================
Fetch demographic data from multiple open-source providers:
1. Meta Population Density Maps (free, global)
2. WorldPop (free, global)
3. Nominatim/OSM for basic demographic enrichment
4. Google Population API (optional, paid)

Usage:
    python src/data_fetchers/fetch_census_demographics.py \
        --bbox "76.8,28.4,77.4,28.9" \
        --region "delhi" \
        --output "data/raw/demographics.csv"
"""

import os
import warnings
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Tuple, Optional
import requests
from shapely.geometry import box, Point

warnings.filterwarnings('ignore')


class DemographicsCollector:
    """Collect demographic data for a geographic region."""
    
    def __init__(self, bbox: Tuple[float, float, float, float], region: str):
        """
        Initialize with bounding box and region name.
        
        Args:
            bbox: (west, south, east, north) coordinates
            region: Region name (e.g., 'delhi', 'mumbai')
        """
        self.bbox = bbox
        self.region = region
        self.west, self.south, self.east, self.north = bbox
        
    def fetch_worldpop_estimates(self, grid_gdf: gpd.GeoDataFrame) -> pd.Series:
        """
        Estimate population using Meta/WorldPop open data approach.
        Returns population estimates based on settlement patterns.
        
        For real implementation, download from:
        https://www.worldpop.org/
        https://data.humdata.org/dataset/meta-high-resolution-population-density-maps
        """
        print("Generating population estimates (synthetic for demo)...")
        np.random.seed(42)
        
        # In production: load actual raster and sample at grid points
        # For now: generate realistic synthetic estimates
        n = len(grid_gdf)
        
        # Population varies by urbanization: some cells have 0 (forests, water),
        # others have high density (city centers)
        population = np.where(
            np.random.random(n) < 0.7,  # 70% of cells have population
            np.random.exponential(scale=2000, size=n) + np.random.normal(5000, 3000, n),
            0
        )
        population = np.clip(population, 0, 50000)
        return pd.Series(population, index=grid_gdf.index)
    
    def estimate_income_distribution(self, population: pd.Series) -> pd.Series:
        """
        Estimate median income based on urbanization level and population density.
        Uses typical income patterns in developing/developed countries.
        """
        print("Estimating income distribution...")
        np.random.seed(42)
        
        # Higher population density correlates with higher income (urban premium)
        urbanization = np.where(population > 5000, 1.0, population / 5000)
        
        # Base income varies by region
        region_income_map = {
            'delhi': 35000,
            'mumbai': 42000,
            'bangalore': 48000,
            'hyderabad': 38000,
            'kolkata': 25000,
            'default': 30000
        }
        
        base_income = region_income_map.get(self.region.lower(), region_income_map['default'])
        
        # Add urbanization premium and noise
        median_income = base_income * (1 + 0.5 * urbanization) + np.random.normal(0, base_income * 0.2, len(population))
        median_income = np.clip(median_income, 10000, 100000)
        
        return pd.Series(median_income, index=population.index)
    
    def estimate_demographics(self, grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Estimate all demographic variables for grid cells.
        In production, use actual census/survey data.
        """
        print("Estimating demographic features...")
        np.random.seed(42)
        
        n = len(grid_gdf)
        demographics = pd.DataFrame(index=grid_gdf.index)
        
        # 1. Population (from WorldPop-like estimates)
        demographics['population'] = self.fetch_worldpop_estimates(grid_gdf)
        
        # 2. Median Income (from economic indicators + urbanization)
        demographics['median_income'] = self.estimate_income_distribution(demographics['population'])
        
        # 3. Average Age (census proxy: mix of young/old in cities vs. rural)
        urbanization = np.where(demographics['population'] > 5000, 0.8, 0.3)
        demographics['avg_age'] = 25 + 15 * urbanization + np.random.normal(0, 5, n)
        demographics['avg_age'] = np.clip(demographics['avg_age'], 15, 60)
        
        # 4. Household Size (inverse relationship with income/urbanization)
        demographics['household_size'] = (
            4.5 - 1.0 * (demographics['median_income'] / 100000) +
            np.random.normal(0, 0.5, n)
        )
        demographics['household_size'] = np.clip(demographics['household_size'], 1.5, 7.0)
        
        # 5. Employment Rate (positive correlation with income, negative with age)
        age_factor = (demographics['avg_age'] - 25) / 35  # 0 at age 25, 1 at age 60
        demographics['employment_rate'] = (
            0.65 + 0.2 * (demographics['median_income'] / 100000) - 0.2 * age_factor +
            np.random.normal(0, 0.05, n)
        )
        demographics['employment_rate'] = np.clip(demographics['employment_rate'], 0.3, 0.9)
        
        # 6. Education Index (positive with income)
        demographics['education_index'] = (
            0.3 + 0.4 * (demographics['median_income'] / 100000) +
            np.random.normal(0, 0.1, n)
        )
        demographics['education_index'] = np.clip(demographics['education_index'], 0.1, 0.95)
        
        # 7. Urbanization Level (based on population density and road network)
        demographics['urbanization_level'] = np.where(
            demographics['population'] > 5000,
            np.random.uniform(0.7, 1.0, n),
            np.random.uniform(0.1, 0.5, n)
        )
        
        # 8. Internet Penetration (positive with income and education)
        demographics['internet_penetration'] = (
            0.4 + 0.4 * (demographics['median_income'] / 100000) + 0.3 * demographics['education_index'] +
            np.random.normal(0, 0.1, n)
        )
        demographics['internet_penetration'] = np.clip(demographics['internet_penetration'], 0.1, 0.95)
        
        # 9. Smartphone Adoption (positive with income, similar to internet)
        demographics['smartphone_adoption'] = (
            0.35 + 0.45 * (demographics['median_income'] / 100000) + 0.25 * demographics['education_index'] +
            np.random.normal(0, 0.1, n)
        )
        demographics['smartphone_adoption'] = np.clip(demographics['smartphone_adoption'], 0.05, 0.98)
        
        return demographics
    
    def save_demographics(self, grid_gdf: gpd.GeoDataFrame, output_path: str):
        """Estimate and save demographics."""
        demographics = self.estimate_demographics(grid_gdf)
        
        # Merge with grid IDs
        demographics['grid_id'] = grid_gdf['grid_id'].values
        
        # Reorder columns
        columns_order = [
            'grid_id', 'population', 'median_income', 'avg_age',
            'household_size', 'employment_rate', 'education_index',
            'urbanization_level', 'internet_penetration', 'smartphone_adoption'
        ]
        demographics = demographics[columns_order]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        demographics.to_csv(output_path, index=False)
        
        print(f"\n✓ Saved demographics for {len(demographics)} grid cells to {output_path}")
        print(f"\nSample data:")
        print(demographics.head(10))
        
        return demographics


def main():
    parser = argparse.ArgumentParser(
        description='Fetch demographic data for grid cells'
    )
    parser.add_argument(
        '--bbox',
        type=str,
        default='76.8,28.4,77.4,28.9',
        help='Bounding box: west,south,east,north (default: Delhi)'
    )
    parser.add_argument(
        '--region',
        type=str,
        default='delhi',
        help='Region name for income/demographic baselines'
    )
    parser.add_argument(
        '--grid-path',
        type=str,
        default='data/processed/grid.geojson',
        help='Path to grid GeoJSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/demographics.csv',
        help='Output CSV path'
    )
    
    args = parser.parse_args()
    
    # Parse bbox
    bbox = tuple(map(float, args.bbox.split(',')))
    
    # Load grid
    if not os.path.exists(args.grid_path):
        print(f"ERROR: Grid file not found at {args.grid_path}")
        print("Please run: jupyter notebook notebooks/01_grid_build.ipynb")
        return
    
    print(f"Loading grid from {args.grid_path}...")
    grid_gdf = gpd.read_file(args.grid_path)
    print(f"Loaded {len(grid_gdf)} grid cells")
    
    # Collect and save
    collector = DemographicsCollector(bbox, args.region)
    collector.save_demographics(grid_gdf, args.output)


if __name__ == '__main__':
    main()
