"""
Parcel Loader

This script loads parcel-level data for the following cities: 
- Boston
- New York City (NYC)
- San Francisco (SF)

The script receives the city as a command-line argument and handles
differences in parcel identifiers and column naming across datasets.
"""



import geopandas as gpd
import pandas as pd
from pathlib import Path
from config import *