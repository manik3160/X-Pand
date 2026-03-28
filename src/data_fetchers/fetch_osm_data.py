import os
import osmnx as ox
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

def fetch_poi_data(bbox, tags=None):
    """
    Fetch POIs from OpenStreetMap within a bounding box.
    bbox: (west, south, east, north)
    tags: dictionary of OSM tags to fetch (e.g., {'amenity': 'restaurant'})
    """
    if tags is None:
        tags = {'amenity': ['restaurant', 'cafe', 'fast_food', 'food_court']}
    
    print(f"Fetching POIs for bbox {bbox}...")
    try:
        # ox.features_from_bbox uses (north, south, east, west) order in some versions
        # but modern osmnx uses (west, south, east, north) or named params
        pois = ox.features_from_bbox(north=bbox[3], south=bbox[1], east=bbox[2], west=bbox[0], tags=tags)
        return pois
    except Exception as e:
        print(f"Error fetching POIs: {e}")
        return None

def main():
    # Delhi Bounding Box
    bbox = (76.8, 28.4, 77.4, 28.9) # (west, south, east, north)
    
    # 1. Fetch Restaurants (Competitors)
    print("Fetching restaurants...")
    restaurants = fetch_poi_data(bbox, tags={'amenity': ['restaurant', 'cafe', 'fast_food']})
    
    if restaurants is not None:
        cols = ['name', 'amenity', 'geometry']
        present_cols = [c for c in cols if c in restaurants.columns]
        restaurants_clean = restaurants[present_cols].copy()
        os.makedirs('data/raw', exist_ok=True)
        restaurants_clean.to_csv('data/raw/osm_restaurants.csv', index=False)
        print(f"Saved {len(restaurants_clean)} restaurants to data/raw/osm_restaurants.csv")

    # 2. Fetch Transit Stops (NEW)
    print("Fetching transit stops...")
    transit_tags = {'public_transport': 'stop_position', 'amenity': ['bus_station', 'taxi_rank'], 'railway': ['station', 'stop']}
    transit = fetch_poi_data(bbox, tags=transit_tags)
    
    if transit is not None:
        os.makedirs('data/raw', exist_ok=True)
        transit.to_csv('data/raw/osm_transit.csv', index=False)
        print(f"Saved {len(transit)} transit points to data/raw/osm_transit.csv")

    # 3. Fetch Road Network for Density
    print("Fetching road network...")
    G = ox.graph_from_bbox(north=bbox[3], south=bbox[1], east=bbox[2], west=bbox[0], network_type='drive')
    nodes, edges = ox.graph_to_gdfs(G)
    
    edges.to_csv('data/raw/osm_roads.csv', index=False)
    print(f"Saved {len(edges)} road segments to data/raw/osm_roads.csv")

if __name__ == "__main__":
    main()
