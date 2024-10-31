import pycountry
from gadm import GADMDownloader
import geopandas as gpd


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
