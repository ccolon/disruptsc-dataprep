# What is it?

This repository contains tools to facilitate the data preparation for the DisruptSC model https://github.com/ccolon/disruptsc. 
The data preparation is done in Python.

# Installation

To install the package, run the following command:

```bash
pip install disruptsc-dataprep
```

It is advised to install the package in a virtual environment, especially if you have other packages that 
might conflict with the dependencies of this package (e.g, geopandas)

# Usage

Submodule admin_boundaries contains functions to download and prepare administrative boundaries data.

```python
from disruptsc-dataprep.admin_boundaries import search_country_by_keyword, get_country_admin_boundaries
```

There are two functions.

```python
search_country_by_keyword(keyword: str)
```

- This is a wrapper of the `pycountry.countries.search_fuzzy` function.
- It returns a list of countries that match the keyword.

```python
get_country_admin_boundaries(country_name: str, ad_level: int)
```
- This is a wrapper of the `gadm.GADMDownloader` class.
- It returns a geopandas.DataFrame of the administrative boundaries of the country specified 
by the country_name at the administrative level specified.
- The search_country_by_keyword function can be used to check the country name beforehand
- It can then be saved to a file using the `to_file` method of the geopandas.DataFrame, 
- ex. `gdf.to_file('path/to/file.geojson', driver="GeoJSON", index=False)`.
