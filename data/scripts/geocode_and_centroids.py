"""
Geocodes and Centroids Generator

This script computes geographic coordinates for parcel-level data
by generating centroid points from parcel polygon geometries.

For each parcel, the script extracts:
- Latitude
- Longitude
- Centroid geometry

The script receives the city as an argument and processes parcel
datasets for the following cities:
- Boston
- New York City (NYC)
- San Francisco (SF)

Output datasets contain parcel identifiers along with centroid-based
coordinates.
"""

import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from config import *