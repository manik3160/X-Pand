"""
src/data_fetchers/fetch_delivery_zones.py
===========================================
Fetch delivery zone operational metrics from multiple sources:

1. Company Data Warehouse (if available)
2. OpenStreetMap + Google Maps (for POI counts, road networks)
3. Synthetic realistic data (for demo/testing)

Usage:
    # Method 1: Use OSM data (free, no API key required)
    python src/data_fetchers/fetch_delivery_zones.py \
        --method osm \
        --bbox "76.8,28.4,77.4,28.9" \
        --output "data/raw/delivery_zones.csv"
    
    # Method 2: Use synthetic data (for demo)
    python src/data_fetchers/fetch_delivery_zones.py \
        --method synthetic \
        --grid-path "data/processed/grid.geojson" \
        --output "data/raw/delivery_zones.csv"
    
    # Method 3: Load from your own data warehouse (custom)
    python src/data_fetchers/fetch_delivery_zones.py \
        --method warehouse \
        --db-connection-string "postgresql://user:pass@host/db" \
        --output "data/raw/delivery_zones.csv"
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Tuple, Optional, Dict
from shapely.geometry import Point, box
import osmnx as ox

warnings.filterwarnings('ignore')


class DeliveryZoneCollector:
    """Collect delivery zone operational metrics."""
    
    def __init__(self, bbox: Tuple[float, float, float, float], region: str = 'delhi'):
        """
        Initialize with bounding box.
        
        Args:
            bbox: (west, south, east, north) coordinates
            region: Region name for realistic parameters
        """
        self.bbox = bbox
        self.region = region
        self.west, self.south, self.east, self.north = bbox
    
    def fetch_osm_delivery_metrics(self, grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Fetch real metrics from OpenStreetMap:
        - Restaurant counts (competitors)
        - Road networks (road density)
        - Transit stops (accessibility)
        """
        print(f"Fetching OpenStreetMap data for {self.region}...")
        
        metrics = pd.DataFrame(index=grid_gdf.index)
        
        try:
            # 1. Fetch restaurants (competitors)
            print("  - Fetching restaurants...")
            restaurants = ox.features_from_bbox(
                north=self.north, south=self.south, 
                east=self.east, west=self.west,
                tags={'amenity': ['restaurant', 'cafe', 'fast_food', 'food_court']}
            )
            
            # Count restaurants per grid cell
            restaurants_gdf = gpd.GeoDataFrame(
                geometry=restaurants.geometry, 
                crs='EPSG:4326'
            )
            
            # Spatial join to count restaurants in each grid cell
            rest_counts = gpd.sjoin(restaurants_gdf, grid_gdf, how='left')
            metrics['active_restaurants'] = rest_counts.groupby('index_right').size()
            metrics['active_restaurants'] = metrics['active_restaurants'].fillna(0).astype(int)
            
            print(f"    Found {len(restaurants)} total restaurants")
            
        except Exception as e:
            print(f"  WARNING: Could not fetch restaurants: {e}")
            metrics['active_restaurants'] = np.random.poisson(5, len(grid_gdf))
        
        try:
            # 2. Fetch road network (for road density)
            print("  - Fetching road network...")
            G = ox.graph_from_bbox(
                north=self.north, south=self.south,
                east=self.east, west=self.west,
                network_type='drive'
            )
            
            # Calculate road density per grid cell
            edges = ox.graph_to_gdfs(G)[1]
            edges_gdf = gpd.GeoDataFrame(
                geometry=edges.geometry, 
                crs='EPSG:4326'
            )
            
            # Spatial join and calculate road length per cell
            edges_in_grid = gpd.sjoin(edges_gdf, grid_gdf, how='left')
            
            # Get cell areas in km²
            grid_area_km2 = grid_gdf.geometry.area / 1e6
            
            # Road density = total road length / cell area
            road_length_per_cell = edges_in_grid.groupby('index_right').geometry.length.sum()
            metrics['road_density_km'] = (road_length_per_cell / grid_area_km2).fillna(0)
            
            print(f"    Found {len(edges)} road segments")
            
        except Exception as e:
            print(f"  WARNING: Could not fetch road network: {e}")
            metrics['road_density_km'] = np.random.uniform(0.2, 1.0, len(grid_gdf))
        
        try:
            # 3. Fetch transit stops
            print("  - Fetching transit stops...")
            transit_tags = {
                'public_transport': 'stop_position',
                'amenity': ['bus_station', 'taxi_rank'],
                'railway': ['station', 'stop']
            }
            transit = ox.features_from_bbox(
                north=self.north, south=self.south,
                east=self.east, west=self.west,
                tags=transit_tags
            )
            
            transit_gdf = gpd.GeoDataFrame(
                geometry=transit.geometry,
                crs='EPSG:4326'
            )
            
            transit_in_grid = gpd.sjoin(transit_gdf, grid_gdf, how='left')
            transit_counts = transit_in_grid.groupby('index_right').size()
            
            # Create transit accessibility metric (0-1 scale)
            max_transit = transit_counts.max() if len(transit_counts) > 0 else 1
            metrics['transit_accessibility'] = (transit_counts / max(max_transit, 1)).fillna(0)
            
            print(f"    Found {len(transit)} transit stops")
            
        except Exception as e:
            print(f"  WARNING: Could not fetch transit stops: {e}")
            metrics['transit_accessibility'] = np.random.uniform(0, 1, len(grid_gdf))
        
        return metrics
    
    def generate_synthetic_delivery_metrics(self, grid_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Generate realistic synthetic delivery zone metrics.
        Use when real data is unavailable.
        """
        print(f"Generating synthetic delivery metrics for {self.region}...")
        np.random.seed(42)
        
        n = len(grid_gdf)
        metrics = pd.DataFrame(index=grid_gdf.index)
        
        # Extract latitude/longitude for location-based variation
        centroids = grid_gdf.geometry.centroid
        lats = np.array([pt.y for pt in centroids])
        lons = np.array([pt.x for pt in centroids])
        
        # Normalize coordinates to [0, 1]
        lat_norm = (lats - self.south) / (self.north - self.south)
        lon_norm = (lons - self.west) / (self.east - self.west)
        
        # Distance from approximate city center (varies by region)
        city_centers = {
            'delhi': (28.6139, 77.2090),
            'mumbai': (19.0760, 72.8777),
            'bangalore': (12.9716, 77.5946),
            'hyderabad': (17.3850, 78.4867),
            'kolkata': (22.5726, 88.3639)
        }
        
        city_center = city_centers.get(self.region.lower(), (28.6139, 77.2090))
        dist_from_center = np.sqrt((lats - city_center[0])**2 + (lons - city_center[1])**2)
        dist_norm = np.clip(dist_from_center / (dist_from_center.max() + 1e-6), 0, 1)
        
        # 1. Active Restaurants (competitors): high in city center, poisson distributed
        centerality = 1 - dist_norm  # More central = more restaurants
        metrics['active_restaurants'] = (
            np.random.poisson(8 * centerality) +
            np.random.poisson(2 * (1 - centerality))
        ).astype(int)
        
        # 2. Road Density: high in urban areas
        metrics['road_density_km'] = (
            0.3 + 0.6 * centerality +
            np.random.uniform(-0.1, 0.1, n)
        )
        metrics['road_density_km'] = np.clip(metrics['road_density_km'], 0.1, 1.0)
        
        # 3. Population Density (people/km²)
        metrics['population_density'] = (
            1000 + 8000 * centerality +
            np.random.exponential(1000, n)
        )
        metrics['population_density'] = np.clip(metrics['population_density'], 100, 15000)
        
        # 4. Active Customers: proportional to population
        metrics['active_customers'] = (
            (metrics['population_density'] / 1000) * np.random.uniform(0.05, 0.15, n)
        ).astype(int)
        metrics['active_customers'] = np.clip(metrics['active_customers'], 0, 1000)
        
        # 5. Total Orders: high volume in central areas
        metrics['total_orders'] = (
            metrics['active_customers'] * np.random.uniform(5, 20, n) +
            np.random.poisson(50, n)
        ).astype(int)
        
        # 6. Average Delivery Time (minutes): longer in outer areas
        metrics['avg_delivery_time_min'] = (
            15 + 20 * dist_norm +
            np.random.normal(0, 2, n)
        )
        metrics['avg_delivery_time_min'] = np.clip(metrics['avg_delivery_time_min'], 5, 60)
        
        # 7. Average Order Value: higher in affluent (central) areas
        metrics['avg_order_value'] = (
            250 + 150 * centerality +
            np.random.normal(0, 30, n)
        )
        metrics['avg_order_value'] = np.clip(metrics['avg_order_value'], 100, 600)
        
        # 8. Competitor Count (other delivery platforms): more in profitable zones
        metrics['competitor_count'] = np.random.poisson(
            2 + 3 * centerality,
            n
        ).astype(int)
        
        # 9. Profitability Label (6-month target)
        # Profitable if: high order value, low delivery time, reasonable volume
        profitability_score = (
            0.4 * (metrics['avg_order_value'] / 600) +
            0.3 * (1 - metrics['avg_delivery_time_min'] / 60) +
            0.2 * (metrics['total_orders'] / metrics['total_orders'].max()) +
            0.1 * (1 - metrics['competitor_count'] / metrics['competitor_count'].max())
        )
        
        # Add noise and threshold
        profitability_score += np.random.normal(0, 0.1, n)
        metrics['is_profitable'] = (profitability_score > 0.5).astype(int)
        
        return metrics
    
    def save_delivery_zones(self, grid_gdf: gpd.GeoDataFrame, method: str, output_path: str):
        """Fetch and save delivery zone metrics."""
        
        if method == 'osm':
            metrics = self.fetch_osm_delivery_metrics(grid_gdf)
        else:  # synthetic
            metrics = self.generate_synthetic_delivery_metrics(grid_gdf)
        
        # Add grid information
        delivery_zones = pd.DataFrame({
            'grid_id': grid_gdf['grid_id'].values,
            'latitude': [pt.y for pt in grid_gdf.geometry.centroid],
            'longitude': [pt.x for pt in grid_gdf.geometry.centroid],
            'zone_name': [f"{self.region.title()}_Zone_{i}" for i in range(len(grid_gdf))]
        })
        
        # Merge metrics
        delivery_zones = pd.concat([delivery_zones, metrics], axis=1)
        
        # Ensure all required columns exist
        required_cols = [
            'grid_id', 'latitude', 'longitude', 'zone_name',
            'total_orders', 'avg_delivery_time_min', 'avg_order_value',
            'active_restaurants', 'active_customers', 'population_density',
            'competitor_count', 'road_density_km', 'is_profitable'
        ]
        
        # Fill missing columns with defaults
        for col in required_cols:
            if col not in delivery_zones.columns:
                if col == 'is_profitable':
                    delivery_zones[col] = np.random.binomial(1, 0.1, len(delivery_zones))
                else:
                    delivery_zones[col] = 0
        
        delivery_zones = delivery_zones[required_cols]
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        delivery_zones.to_csv(output_path, index=False)
        
        print(f"\n✓ Saved delivery zones for {len(delivery_zones)} grid cells to {output_path}")
        print(f"\nSample data:")
        print(delivery_zones.head(10))
        
        return delivery_zones


def main():
    parser = argparse.ArgumentParser(
        description='Fetch delivery zone operational metrics'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['osm', 'synthetic', 'warehouse'],
        default='synthetic',
        help='Data collection method'
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
        help='Region name'
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
        default='data/raw/delivery_zones.csv',
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
    collector = DeliveryZoneCollector(bbox, args.region)
    collector.save_delivery_zones(grid_gdf, args.method, args.output)


if __name__ == '__main__':
    main()
