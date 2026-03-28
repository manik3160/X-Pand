"""
src/data_fetchers/validate_and_merge.py
========================================
Validate collected data and merge delivery zones + demographics into a single dataset.

Usage:
    python src/data_fetchers/validate_and_merge.py \
        --delivery-zones data/raw/delivery_zones.csv \
        --demographics data/raw/demographics.csv \
        --grid data/processed/grid.geojson \
        --output data/processed/combined_features.csv
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Tuple, Dict, List
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validate data quality and consistency."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.issues = []
    
    def log(self, msg: str):
        if self.verbose:
            logger.info(msg)
    
    def check_completeness(self, df: pd.DataFrame, name: str, max_missing_pct: float = 5.0) -> bool:
        """Check if dataframe has acceptable missing values."""
        missing_pct = (df.isnull().sum() / len(df)) * 100
        
        issues = missing_pct[missing_pct > max_missing_pct]
        if len(issues) > 0:
            self.log(f"⚠️ {name}: Columns with >{max_missing_pct}% missing:")
            for col, pct in issues.items():
                self.log(f"   {col}: {pct:.1f}%")
                self.issues.append(f"{name}.{col}: {pct:.1f}% missing")
            return False
        else:
            self.log(f"✓ {name}: Completeness check passed")
            return True
    
    def check_consistency(self, df: pd.DataFrame, name: str, rules: Dict) -> bool:
        """
        Check data consistency rules.
        
        Args:
            rules: dict of {column_name: (min, max)} for numeric checks
        """
        passed = True
        for col, (min_val, max_val) in rules.items():
            if col not in df.columns:
                continue
            
            out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
            if len(out_of_range) > 0:
                pct = (len(out_of_range) / len(df)) * 100
                self.log(f"⚠️ {name}.{col}: {pct:.1f}% of values out of range [{min_val}, {max_val}]")
                self.issues.append(f"{name}.{col}: {pct:.1f}% out of range")
                passed = False
        
        if passed:
            self.log(f"✓ {name}: Consistency check passed")
        
        return passed
    
    def check_duplicates(self, df: pd.DataFrame, name: str, key_col: str) -> bool:
        """Check for duplicate keys."""
        duplicates = df.duplicated(subset=[key_col], keep=False)
        if duplicates.any():
            n_dups = duplicates.sum()
            pct = (n_dups / len(df)) * 100
            self.log(f"⚠️ {name}: {n_dups} ({pct:.1f}%) duplicate {key_col} values")
            self.issues.append(f"{name}: {n_dups} duplicate {key_col} values")
            return False
        else:
            self.log(f"✓ {name}: No duplicate {key_col} values")
            return True
    
    def check_alignment(self, df1: pd.DataFrame, df2: pd.DataFrame, key: str, name1: str, name2: str) -> bool:
        """Check if two dataframes have matching keys."""
        keys1 = set(df1[key])
        keys2 = set(df2[key])
        
        missing_in_df2 = keys1 - keys2
        missing_in_df1 = keys2 - keys1
        
        if len(missing_in_df2) > 0:
            self.log(f"⚠️ {key} present in {name1} but missing in {name2}: {len(missing_in_df2)} keys")
            self.issues.append(f"{name1}-{name2} alignment: {len(missing_in_df2)} keys in {name1} only")
        
        if len(missing_in_df1) > 0:
            self.log(f"⚠️ {key} present in {name2} but missing in {name1}: {len(missing_in_df1)} keys")
            self.issues.append(f"{name1}-{name2} alignment: {len(missing_in_df1)} keys in {name2} only")
        
        if len(missing_in_df1) == 0 and len(missing_in_df2) == 0:
            self.log(f"✓ {name1} and {name2} are perfectly aligned on {key}")
            return True
        else:
            return False


class DataMerger:
    """Merge and combine multiple data sources."""
    
    def __init__(self):
        self.validator = DataValidator(verbose=True)
    
    def load_data(self, delivery_zones_path: str, demographics_path: str, grid_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame]:
        """Load all required datasets."""
        logger.info("Loading data files...")
        
        logger.info(f"  Loading delivery zones from {delivery_zones_path}...")
        delivery_zones = pd.read_csv(delivery_zones_path)
        logger.info(f"    ✓ Loaded {len(delivery_zones)} zones")
        
        logger.info(f"  Loading demographics from {demographics_path}...")
        demographics = pd.read_csv(demographics_path)
        logger.info(f"    ✓ Loaded {len(demographics)} cells")
        
        logger.info(f"  Loading grid from {grid_path}...")
        grid = gpd.read_file(grid_path)
        logger.info(f"    ✓ Loaded {len(grid)} grid cells")
        
        return delivery_zones, demographics, grid
    
    def validate_data(self, delivery_zones: pd.DataFrame, demographics: pd.DataFrame, grid: gpd.GeoDataFrame) -> bool:
        """Run all validation checks."""
        logger.info("\n" + "="*60)
        logger.info("DATA VALIDATION")
        logger.info("="*60)
        
        # Delivery zones validation
        logger.info("\nValidating delivery_zones...")
        dz_rules = {
            'total_orders': (0, 100000),
            'avg_delivery_time_min': (0, 120),
            'avg_order_value': (10, 1000),
            'active_restaurants': (0, 1000),
            'active_customers': (0, 10000),
            'population_density': (0, 20000),
            'competitor_count': (0, 50),
            'road_density_km': (0, 2),
            'is_profitable': (0, 1)
        }
        
        self.validator.check_completeness(delivery_zones, 'delivery_zones')
        self.validator.check_consistency(delivery_zones, 'delivery_zones', dz_rules)
        self.validator.check_duplicates(delivery_zones, 'delivery_zones', 'grid_id')
        
        # Demographics validation
        logger.info("\nValidating demographics...")
        dem_rules = {
            'population': (0, 100000),
            'median_income': (10000, 1000000),
            'avg_age': (10, 100),
            'household_size': (0.5, 15),
            'employment_rate': (0, 1),
            'education_index': (0, 1),
            'urbanization_level': (0, 1),
            'internet_penetration': (0, 1),
            'smartphone_adoption': (0, 1)
        }
        
        self.validator.check_completeness(demographics, 'demographics')
        self.validator.check_consistency(demographics, 'demographics', dem_rules)
        self.validator.check_duplicates(demographics, 'demographics', 'grid_id')
        
        # Cross-dataset alignment
        logger.info("\nValidating cross-dataset alignment...")
        self.validator.check_alignment(delivery_zones, demographics, 'grid_id', 'delivery_zones', 'demographics')
        self.validator.check_alignment(delivery_zones, grid, 'grid_id', 'delivery_zones', 'grid')
        
        # Print summary
        logger.info("\n" + "="*60)
        if len(self.validator.issues) == 0:
            logger.info("✅ All validation checks PASSED")
        else:
            logger.warning(f"⚠️ Found {len(self.validator.issues)} issues:")
            for issue in self.validator.issues:
                logger.warning(f"   - {issue}")
        logger.info("="*60)
        
        return len(self.validator.issues) == 0
    
    def merge_data(self, delivery_zones: pd.DataFrame, demographics: pd.DataFrame, grid: gpd.GeoDataFrame) -> pd.DataFrame:
        """Merge delivery zones and demographics on grid_id."""
        logger.info("\n" + "="*60)
        logger.info("DATA MERGING")
        logger.info("="*60)
        
        logger.info("Merging delivery zones + demographics on grid_id...")
        
        # Merge on grid_id
        combined = delivery_zones.merge(
            demographics,
            on='grid_id',
            how='outer'
        )
        
        # Add geometry from grid
        combined = combined.merge(
            grid[['grid_id', 'geometry']],
            on='grid_id',
            how='left'
        )
        
        logger.info(f"✓ Merged {len(combined)} records")
        
        # Handle any missing values (fill with reasonable defaults)
        logger.info("Handling missing values...")
        
        # Numeric columns: fill with median
        numeric_cols = combined.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if combined[col].isnull().any():
                median_val = combined[col].median()
                combined[col].fillna(median_val, inplace=True)
                logger.info(f"  - {col}: filled {combined[col].isnull().sum()} with median={median_val:.2f}")
        
        # String columns: fill with 'Unknown'
        str_cols = combined.select_dtypes(include=['object']).columns
        for col in str_cols:
            if combined[col].isnull().any():
                combined[col].fillna('Unknown', inplace=True)
                logger.info(f"  - {col}: filled {combined[col].isnull().sum()} with 'Unknown'")
        
        # Reorder columns for readability
        column_order = [
            'grid_id', 'latitude', 'longitude', 'zone_name',
            'population', 'population_density',
            'total_orders', 'avg_delivery_time_min', 'avg_order_value',
            'active_restaurants', 'active_customers',
            'competitor_count', 'road_density_km',
            'median_income', 'avg_age', 'household_size',
            'employment_rate', 'education_index',
            'urbanization_level', 'internet_penetration', 'smartphone_adoption',
            'is_profitable', 'geometry'
        ]
        
        # Keep only columns that exist
        column_order = [col for col in column_order if col in combined.columns]
        combined = combined[column_order]
        
        logger.info(f"\nFinal dataset: {len(combined)} rows × {len(combined.columns)} columns")
        logger.info(f"Columns: {', '.join(combined.columns.tolist())}")
        
        return combined
    
    def save_combined(self, combined: pd.DataFrame, output_path: str):
        """Save combined dataset."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as CSV (without geometry for easy access)
        csv_path = output_path.replace('.geojson', '.csv')
        combined_csv = combined.drop(columns=['geometry'], errors='ignore')
        combined_csv.to_csv(csv_path, index=False)
        logger.info(f"\n✓ Saved combined data (CSV) to {csv_path}")
        
        # Save as GeoJSON (with geometry for spatial analysis)
        combined_gdf = gpd.GeoDataFrame(combined, geometry='geometry', crs='EPSG:4326')
        combined_gdf.to_file(output_path, driver='GeoJSON')
        logger.info(f"✓ Saved combined data (GeoJSON) to {output_path}")
        
        # Print summary statistics
        logger.info("\n" + "="*60)
        logger.info("SUMMARY STATISTICS")
        logger.info("="*60)
        print(combined_csv.describe())
        
        return combined_csv


def main():
    parser = argparse.ArgumentParser(
        description='Validate and merge delivery zones and demographics data'
    )
    parser.add_argument(
        '--delivery-zones',
        type=str,
        default='data/raw/delivery_zones.csv',
        help='Path to delivery_zones.csv'
    )
    parser.add_argument(
        '--demographics',
        type=str,
        default='data/raw/demographics.csv',
        help='Path to demographics.csv'
    )
    parser.add_argument(
        '--grid',
        type=str,
        default='data/processed/grid.geojson',
        help='Path to grid.geojson'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed/combined_features.geojson',
        help='Output file path'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate, do not merge'
    )
    
    args = parser.parse_args()
    
    # Load data
    merger = DataMerger()
    try:
        delivery_zones, demographics, grid = merger.load_data(
            args.delivery_zones,
            args.demographics,
            args.grid
        )
    except FileNotFoundError as e:
        logger.error(f"ERROR: {e}")
        return
    
    # Validate
    validation_passed = merger.validate_data(delivery_zones, demographics, grid)
    
    if args.validate_only:
        return
    
    # Merge and save
    if validation_passed or not args.validate_only:
        combined = merger.merge_data(delivery_zones, demographics, grid)
        merger.save_combined(combined, args.output)
        logger.info("\n✅ Data collection and merging complete!")
    else:
        logger.warning("\n⚠️ Validation failed. Fix issues before merging.")


if __name__ == '__main__':
    main()
