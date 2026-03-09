"""
Parcel Cleaning and Standardization

This script cleans and standardizes parcel-level attribute data after it
has been loaded from raw city datasets.

The script cleans data across cities by:
- Renaming columns
- Standardizing parcel identifiers
- Converting data types
- Removing duplicates or invalid records

The script receives the city as a command-line argument, processes
parcel datasets, and writes standardized versions to the cleaned data 
directory for later use.
"""



import pandas as pd
import numpy as np
from config import *