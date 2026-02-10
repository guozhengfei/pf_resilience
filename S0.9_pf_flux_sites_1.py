import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import os

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
    AMF_site_dir = os.path.join('..', '..', 'Fluxnet2025', 'AMF', 'AmeriFlux-site-info.csv')
    ICOS_site_dir = os.path.join('..', '..', 'Fluxnet2025', 'ICOS', 'ICOS_stations.csv')


    shapefile_dir = os.path.join('..', '..', 'Project 2 pf resilience/1_Input/pf_shp', 'pf_shape.shp')

    csv_file = AMF_site_dir # CSV file with latitude/longitude columns
    csv_file2 = ICOS_site_dir
    shapefile = shapefile_dir  # Shapefile with polygons

    # Check points against polygons
    result1 = check_points_in_polygons(csv_file, shapefile)
    result1 = result1.loc[result1['in_polygon']]

    result2 = check_points_in_polygons(csv_file2, shapefile)
    result2 = result2.loc[result2['in_polygon']]

    # # Save results to a new CSV
    # result.drop(columns=['geometry']).to_csv('points_with_polygon_check.csv', index=False)
    # print("Results saved to 'points_with_polygon_check.csv'")
    #
    # urban_site = result['Site ID'].values.tolist()

    pf_siteNames = ['CA-ARB', 'CA-ARF', 'CA-CF1', 'CA-HPC', 'CA-Man', 'CA-Mtk',
       'CA-NS1', 'CA-NS2', 'CA-NS3', 'CA-NS4', 'CA-NS5', 'CA-NS6',
       'CA-NS7', 'CA-Qc2', 'CA-Qfo', 'CA-SCB', 'CA-SCC', 'CA-SF1',
       'CA-SMC', 'US-BZB', 'US-BZF', 'US-BZo', 'US-BZS', 'US-CAK',
       'US-Cms', 'US-EML', 'US-Fcr', 'US-GLE', 'US-HVs', 'US-ICh',
       'US-ICs', 'US-ICt', 'US-NGB', 'US-NGC', 'US-Prr', 'US-Rpf',
       'US-Sag', 'US-xBA', 'US-xBN', 'US-xDJ', 'US-xHE', 'US-xNW',
       'US-xTL', 'US-YK2', 'FI-Ouk', 'FI-Sod', 'GL-NuF', 'GL-ZaF', 'SE-Sto', 'FR-CLt',
       'FI-Var', 'CH-Dav', 'GL-ZaH', 'FI-Lom', 'FI-Ken', 'IT-Niv']
