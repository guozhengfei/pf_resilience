import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import os

YEAR_MAX = 2023
MIN_OBS_YEARS = 5

def check_points_in_polygons(csv_path, shp_path, lat_col='lat', lon_col='lon'):
    """
    Check which points from a CSV are located within polygons from a shapefile.

    Args:
        csv_path (str): Path to the CSV file containing points
        shp_path (str): Path to the shapefile containing polygons
        lat_col (str): Name of the latitude column in CSV (default 'lat')
        lon_col (str): Name of the longitude column in CSV (default 'lon')

    Returns:
        GeoDataFrame: Original points with an additional column 'in_polygon' indicating
                     whether each point is inside any polygon (True/False)
    """
    # Read the shapefile
    polygons_gdf = gpd.read_file(shp_path)

    # Read the CSV file with points
    points_df = pd.read_csv(csv_path)

    # Convert points to GeoDataFrame
    geometry = [Point(xy) for xy in zip(points_df[lon_col], points_df[lat_col])]
    points_gdf = gpd.GeoDataFrame(points_df, geometry=geometry, crs=polygons_gdf.crs)

    # Perform spatial join to find points within polygons
    points_in_polygons = gpd.sjoin(points_gdf, polygons_gdf, how='left', predicate='within')

    # Add a column indicating whether the point is in any polygon
    points_gdf['in_polygon'] = ~points_in_polygons['index_right'].isna()

    return points_gdf


# Example usage
if __name__ == "__main__":
    # Replace these paths with your actual file paths

    shapefile_dir = os.path.join('..', '..', 'Project 2 pf resilience/1_Input/pf_shp', 'pf_shape.shp')

    csv_file = '/Volumes/Zhengfei_02/Fluxnet_2025/fluxnet_shuttle.csv' # CSV file with latitude/longitude columns

    shapefile = shapefile_dir  # Shapefile with polygons

    # Check points against polygons
    result1 = check_points_in_polygons(csv_file, shapefile)
    result1 = result1.loc[result1['in_polygon']]
    result1['last_year_use'] = result1['last_year'].clip(upper=YEAR_MAX)
    result1['obs_years'] = result1['last_year_use'] - result1['first_year'] + 1
    result1 = result1.loc[result1['obs_years'] >= MIN_OBS_YEARS].copy()
    # result1.to_csv('/Volumes/Zhengfei_02/Fluxnet_2025/pf_fluxnet_sites_5yr.csv', index=False)
    pf_siteNames = list(result1['site_id'].values)
    print(f'Sites kept: {len(result1)}')

    # pf_siteNames = ['CA-ARB', 'CA-ARF', 'CA-KLP', 'CA-Man', 'CA-PB1', 'CA-PB2', 'CA-Qfo', 'CA-SCB', 'CA-SCC', 'CH-Aws', 'CH-Dav', 'CN-Aro', 'CN-DaW', 'CN-Dan', 'CN-Dsh', 'CN-Jng', 'CN-You', 'FI-Ken', 'FI-Var', 'FR-CLt', 'GL-NuF', 'IT-Niv', 'MN-Hst', 'MN-Kbu', 'MN-Nkh', 'NO-Fns', 'RU-Ch2', 'RU-Che', 'RU-Ege', 'RU-NeC', 'RU-NeF', 'RU-Sk2', 'RU-SkP', 'SE-St1', 'US-A10', 'US-BZB', 'US-BZF', 'US-BZo', 'US-Cms', 'US-EML', 'US-Fo1', 'US-GLE', 'US-ICh', 'US-ICs', 'US-ICt', 'US-NGB', 'US-NGC', 'US-NR3', 'US-Prr', 'US-Rpf', 'US-TKs', 'US-Uaf', 'US-YK1', 'US-YK2', 'US-xBA', 'US-xBN', 'US-xDJ', 'US-xHE', 'US-xNW', 'US-xTL']
