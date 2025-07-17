import pandas as pd
# import pycountry
# from gadm import GADMDownloader
import geopandas as gpd
import requests
from itertools import combinations


def search_country_by_keyword(keyword: str):
    return pycountry.countries.search_fuzzy(keyword)


def get_country_admin_boundaries(country_name: str, ad_level: int) -> gpd.GeoDataFrame:
    """Wrapper of GADM function to get admin boundaries of a country.

    Can use country ISO code or country name.
    Can use `search_country_by_keyword` to get the country name or ISO code."""
    downloader = GADMDownloader(version="4.0")
    gdf = downloader.get_shape_data_by_country_name(country_name=country_name, ad_level=ad_level)
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.crs == "EPSG:4326"
    return gdf


def get_country_admin_boundaries_from_gadm(iso: str, ad_level: int) -> gpd.GeoDataFrame:
    """Wrapper of GADM function to get admin boundaries of a country."""
    url = f"https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{iso}_{ad_level}.json"
    return gpd.read_file(url)
    # gdf.to_file("gadm41_UZB_0.geojson", driver="GeoJSON")
    # print("GeoDataFrame saved as gadm41_UZB_0.geojson")


def get_region_adjacency(gdf, region_label: str) -> dict:
    # Ensure geometry is valid
    gdf['geometry'] = gdf['geometry'].to_crs(3857).buffer(10)

    # Build spatial index for fast lookup
    adjacency = {}

    for idx, row in gdf.iterrows():
        name = row[region_label]
        geom = row['geometry']

        # Find all provinces that touch this one (excluding self)
        neighbors = gdf[gdf.geometry.overlaps(geom)]
        neighbor_names = neighbors[region_label].tolist()

        adjacency[name] = neighbor_names

    return adjacency


def find_combinations(data, threshold, max_size=3, adjacency=None):
    results = []
    used_regions = set()

    for size in range(1, max_size + 1):
        # Exclude already used regions
        remaining_items = {k: v for k, v in data.items() if k not in used_regions}

        for combo in combinations(remaining_items.items(), size):
            regions = [k for k, _ in combo]
            total = sum(v for _, v in combo)

            if total > threshold:
                if adjacency is None or are_all_adjacent(regions, adjacency):
                    results.append(regions)
                    used_regions.update(regions)

def are_all_adjacent(combo, adjacency):
    # Build a set of all items reachable within the combo
    visited = set()
    stack = [combo[0]]

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend([n for n in adjacency.get(node, []) if n in combo and n not in visited])

    return visited == set(combo)


countries = ['KAZ', 'KGZ', 'TJK', 'TKM', 'UZB', 'GEO', 'ARM', 'AZE']
gdf_dict = {}
for country in countries:
    print(f"Downloading {country}")
    gdf_dict[country] = get_country_admin_boundaries_from_gadm(country, 0)
    assert isinstance(gdf_dict[country], gpd.GeoDataFrame)
    assert gdf_dict[country].crs == "EPSG:4326"

country_boundaries = pd.concat(list(gdf_dict.values()), axis=0, ignore_index=True)
country_boundaries.to_file("gadm41_country_boundaries.geojson", driver="GeoJSON")