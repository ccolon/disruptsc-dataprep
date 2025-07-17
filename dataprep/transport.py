import logging
import os
import re
from pathlib import Path

from shapely.geometry import Point, LineString
from shapely import wkt
import geopandas as gpd
import pandas as pd
from shapely.ops import split, snap, linemerge
from shapely.geometry import MultiPoint
from tqdm import tqdm


def update_linestring(linestring, new_end1, new_end2):
    return LineString([new_end1.coords[0]] + linestring.coords[1:-1] + [new_end2.coords[0]])


def get_merge_nodes(edges):
    return get_degree_n_nodes(edges, 2)


def get_degree_n_nodes(edges, n: int):
    end_points_connectivity = pd.concat([edges['end1'], edges['end2']]).value_counts()
    return end_points_connectivity[end_points_connectivity == n].index.sort_values().to_list()


def get_edges_from_endpoints(edges, endpoints: list):
    merged_df_end1 = pd.merge(pd.DataFrame({'endpoint': endpoints}),
                              edges[['id', 'end1']].rename(columns={'id': 'edge_id', 'end1': 'endpoint'}),
                              on='endpoint', how='left').dropna().astype(int)
    merged_df_end2 = pd.merge(pd.DataFrame({'endpoint': endpoints}),
                              edges[['id', 'end2']].rename(columns={'id': 'edge_id', 'end2': 'endpoint'}),
                              on='endpoint', how='left').dropna().astype(int)
    merged_df = pd.concat([merged_df_end1, merged_df_end2])
    return merged_df.groupby("endpoint")['edge_id'].apply(list).to_dict()


def check_degree2(node_id_to_edges_ids: dict):
    return (pd.Series(node_id_to_edges_ids).apply(len) == 2).all()


def merge_lines_with_tolerance(lines):
    """Snap lines to each other within a tolerance and then merge them."""
    snapped_lines = [snap(line, lines[0], TOLERANCE) for line in lines]  # Snap all lines to the first one
    return linemerge(snapped_lines)  # Merge the snapped lines


def merge_edges_attributes(gdf):
    new_data = {'geometry': linemerge(gdf.geometry.to_list())}
    if new_data['geometry'].geom_type != "LineString":
        print(gdf.geometry)
        print(new_data['geometry'])
        raise ValueError(f"Merged geometry is: {new_data['geometry'].geom_type}")

    def string_to_list(s):
        return list(map(int, re.findall(r'\d+', s)))

    def merge_or_unique(column):
        unique_vals = gdf[column].dropna().unique()
        return unique_vals[0] if len(unique_vals) == 1 else ', '.join(unique_vals)

    # Aggregate columns based on given rules
    if 'km' in gdf.columns:
        new_data['km'] = gdf['km'].sum()
    if 'osmids' in gdf.columns:
        new_data['osmids'] = str(string_to_list(', '.join(map(str, gdf['osmids'].fillna('')))))
    if 'name' in gdf.columns:
        new_data['name'] = ', '.join(filter(None, gdf['name'].astype(str)))  # Ignore None values
    if 'capacity' in gdf.columns:
        new_data['capacity'] = gdf['capacity'].min()
    for column_name in ['end1', 'end2']:
        if column_name in gdf.columns:
            new_data[column_name] = None
    for column_name in ['special', 'class', 'surface', 'disruption']:
        if column_name in gdf.columns:
            new_data[column_name] = merge_or_unique(column_name)

    # Create a new row with the merged data
    return new_data


def update_gdf(gdf, new_data, old_ids, new_id):
    for column_name, value in new_data.items():
        gdf.at[new_id, column_name] = value
    gdf.loc[list(set(old_ids) - {new_id}), 'to_keep'] = False


def update_dict(my_dict, old_value, new_value):
    for key, value_list in my_dict.items():
        if old_value in value_list:
            my_dict[key] = [new_value if v == old_value else v for v in value_list]
    return my_dict


def remove_degree_2_nodes(edges):
    merge_nodes = get_merge_nodes(edges)
    print(f"Nb degree 2 nodes: {len(merge_nodes)}")
    merge_nodes_with_edges_ids = get_edges_from_endpoints(edges, merge_nodes)
    print(f"Check that they are actually 2 transport_edges associated: {check_degree2(merge_nodes_with_edges_ids)}")

    edges['to_keep'] = True
    edges = edges.set_index('id')
    for merged_node in tqdm(list(merge_nodes_with_edges_ids.keys()), total=len(merge_nodes_with_edges_ids),
                            desc="merging transport_edges"):
        edge_ids = merge_nodes_with_edges_ids.pop(merged_node)
        if edge_ids[0] == edge_ids[1]:
            print(f'One self loop generated {edge_ids[0]}')
            continue
        merged_attributes = merge_edges_attributes(edges.loc[edge_ids])
        old_id = max(edge_ids)
        new_id = min(edge_ids)
        update_gdf(edges, merged_attributes, edge_ids, new_id)
        merge_nodes_with_edges_ids = update_dict(merge_nodes_with_edges_ids, old_id, new_id)

    print(
        f"Check that all resulting geometries are LineString: {edges['geometry'].apply(lambda geom: geom.geom_type == 'LineString').all()}")

    edges = edges[edges['to_keep']]
    edges = edges.drop(columns=['to_keep'])
    edges = edges.reset_index()
    return edges


# s = remove_degree_2_nodes(transport_edges)


def geometry_to_list_of_single_geom(geom, target_type: str):
    if (geom.geom_type == "Linestring") and (target_type == "LineString"):
        return [geom]
    if (geom.geom_type == "Point") and (target_type == "Point"):
        return [geom]
    if (geom.geom_type == "MultiLineString") and (target_type == "LineString"):
        return [single_geom for single_geom in geom.geoms]
    if (geom.geom_type == "MultiPoint") and (target_type == "Point"):
        return [single_geom for single_geom in geom.geoms]
    if geom.geom_type == "GeometryCollection":
        geoms = [single_geom for single_geom in geom.geoms]
        list_of_single_geoms = [geometry_to_list_of_single_geom(single_geom, target_type) for single_geom in geoms]
        return [item for sublist in list_of_single_geoms for item in sublist]
    return []


def treat_overlapping_linestrings(line0, line1):
    if line0.within(line1):
        return None, [line1]
    elif line1.within(line0):
        return [line0], None
    elif line0.overlaps(line1):
        overlapping_parts = geometry_to_list_of_single_geom(line0.intersection(line1), "LineString")
        new_line0 = overlapping_parts
        remaining_geom = line0.difference(line1)
        if remaining_geom.geom_type == "LineString":
            new_line1 = [remaining_geom]
        elif remaining_geom.geom_type == "MultiLineString":
            new_line1 = [geom for geom in remaining_geom.geoms]
        return new_line0, new_line1
    else:
        return False


def extract_endpoints(line):
    return [Point(line.coords[0]), Point(line.coords[-1])]


def get_intersection_without_endpoints(line0, line1):
    intersection = line0.intersection(line1)
    if intersection.is_empty:
        return False
    intersection_points = []
    if intersection.geom_type == "Point":  # Single intersection point
        intersection_points += [intersection]
    elif intersection.geom_type == "MultiPoint":  # Multiple intersection points
        intersection_points = list(intersection.geoms)
    elif intersection.geom_type == "GeometryCollection":  # super rare
        intersection_points += geometry_to_list_of_single_geom(intersection, "Point")
        intersection_points += geometry_to_list_of_single_geom(intersection, "MultiPoint")
        linestrings = geometry_to_list_of_single_geom(intersection, "LineString")
        intersection_points += [Point(coord) for linestring in linestrings for coord in linestring.coords]
    elif intersection.geom_type == "LineString":  # super rare
        intersection_points = [Point(coord) for coord in intersection.coords]
    elif intersection.geom_type == "MultiLineString":  # super rare
        linestrings = geometry_to_list_of_single_geom(intersection, "LineString")
        intersection_points = [Point(coord) for linestring in linestrings for coord in linestring.coords]
    else:
        print(line0, line1, intersection)
        raise ValueError(str(intersection.geom_type))
    endpoints = extract_endpoints(line0) + extract_endpoints(line1)
    intersection_points = [point for point in intersection_points if not is_close_to_any(point, endpoints)]
    if len(intersection_points) == 0:
        return False
    else:
        return list(set(intersection_points))


def is_close_to_any(point, point_list):
    """Check if a point is close to any endpoint using shapely's snap function."""
    return any(snap(point, p, TOLERANCE).equals(p) for p in point_list)


def find_new_geometries_for_overlapping_edges(edges, specific_ids=False):
    # Build a spatial index
    spatial_index = edges.sindex

    # Identify LineStrings whose coordinates are fully within another LineString
    # cond_fully_within = pd.Series(False, index=transport_edges.index)
    new_geometries = {}  # index: new geom
    chunck_to_process = edges
    if isinstance(specific_ids, list):
        chunck_to_process = edges.loc[specific_ids]
    for i, row1 in tqdm(chunck_to_process.iterrows(), total=len(chunck_to_process),
                        desc="Processing overlapping transport_edges"):
        if i in new_geometries.keys():
            continue
        # Use the spatial index to find potential candidates
        possible_matches_index = list(spatial_index.intersection(row1.geometry.bounds))
        possible_matches = edges.iloc[possible_matches_index]

        # Compare only with potential candidates
        for j, row2 in possible_matches.iterrows():
            if (i != j) and (j not in new_geometries.keys()):
                new_geoms = treat_overlapping_linestrings(row1.geometry, row2.geometry)
                if new_geoms:
                    new_geometries[i], new_geometries[j] = new_geoms
    return new_geometries


def format_new_edges(edges, new_geometries, border=False):
    new_edges_list = []
    for i, new_geom in new_geometries.items():
        if new_geom:
            new_edges = gpd.GeoDataFrame({'geometry': new_geom}, crs=edges.crs)
            row_data = edges.loc[[i]].drop(columns=["geometry"])  # Keep original structure
            row_data_repeated = row_data.loc[row_data.index.repeat(len(new_geom))]  # Repeat for each part
            new_edges = pd.concat([new_edges, row_data_repeated.reset_index(drop=True)], axis=1)
            if border:
                new_edges.loc[new_edges.index[1], 'special'] = 'border'  # the middle one is the border
            new_edges_list += [new_edges]
    all_new_edges = pd.concat(new_edges_list)
    all_new_edges = all_new_edges[~all_new_edges['geometry'].isnull()]
    return all_new_edges


def add_new_edges_to_gdf(edges, new_geometries, all_new_edges):
    print(f'Removing {len(new_geometries.keys())} transport_edges due to the removal of overlapping parts')
    edges = edges.drop(index=list(new_geometries.keys()))
    print(f'Adding {all_new_edges.shape[0]} transport_edges')
    edges = pd.concat([edges, all_new_edges], ignore_index=True)

    edges['id'] = list(range(edges.shape[0]))
    edges.index = edges['id']
    new_edge_ids = edges['id'].iloc[-all_new_edges.shape[0]:].to_list()

    return edges, new_edge_ids


def treat_overlapping_edges(edges, specific_ids=False):
    new_geometries = find_new_geometries_for_overlapping_edges(edges, specific_ids)
    if len(new_geometries) > 0:
        all_new_edges = format_new_edges(edges, new_geometries)
        return add_new_edges_to_gdf(edges, new_geometries, all_new_edges)
    else:
        return edges, []


def treat_intersecting_edges(edges, specific_ids=False):
    new_geometries = find_splitted_lines(edges, specific_ids)
    if len(new_geometries) > 0:
        all_new_edges = format_new_edges(edges, new_geometries)
        return add_new_edges_to_gdf(edges, new_geometries, all_new_edges)
    else:
        return edges, []


def find_splitted_lines(edges, specific_ids=False):
    # Build a spatial index
    spatial_index = edges.sindex

    # Identify LineStrings whose coordinates are fully within another LineString
    new_geometries = {}  # index: new geom
    chunck_to_process = edges['geometry'].to_dict()
    if isinstance(specific_ids, list):
        chunck_to_process = edges.loc[specific_ids, 'geometry'].to_dict()
    for i, row1_geom in tqdm(chunck_to_process.items(), total=len(chunck_to_process),
                             desc="Processing intersecting transport_edges"):
        if i in new_geometries.keys():
            continue
        # Use the spatial index to find potential candidates
        possible_matches_index = list(spatial_index.intersection(row1_geom.bounds))
        possible_matches = edges.iloc[possible_matches_index]['geometry'].to_dict()

        # Compare only with potential candidates
        for j, row2_geom in possible_matches.items():
            if (i != j) and (j not in new_geometries.keys()):
                # print(i, j)
                new_intersection_points = get_intersection_without_endpoints(row1_geom, row2_geom)
                if new_intersection_points:
                    # print(i, j, get_intersection_without_endpoints(row1_geom, row2_geom))
                    # print(chunck_to_process[i], chunck_to_process[j])
                    split_geometry_collection1 = split(row1_geom, MultiPoint(new_intersection_points))
                    split_geometry_collection2 = split(row2_geom, MultiPoint(new_intersection_points))
                    new_geometries[i] = [geom for geom in split_geometry_collection1.geoms]
                    new_geometries[j] = [geom for geom in split_geometry_collection2.geoms]

    return new_geometries


def remove_self_loop(edges):
    # remove selfloop
    cond = edges['end1'] == edges['end2']
    if cond.sum() > 0:
        print(f"Removing {cond.sum()} self-loops")
        edges = edges[~cond]
    return edges


def remove_duplicated_geometry(edges):
    cond = edges['geometry'].to_wkt().duplicated()
    if cond.sum() > 0:
        print(f"Removing {cond.sum()} duplicated transport_edges")
        edges = edges[~cond]
    return edges


def keep_one_edge_same_endpoints(edges):
    # if several transport_edges have the same start and end points, keep only one
    edges['end_set'] = edges.apply(lambda row: frozenset([row['end1'], row['end2']]),
                                   axis=1)  # Create a set representation of each row's end1 and end2
    cond = edges['end_set'].duplicated()
    if cond.sum() > 0:
        print(f"Removing {cond.sum()} transport_edges that have the same endpoints")
        edges = edges[~cond]
    edges = edges.drop(columns=['end_set'])  # Drop the temporary column
    edges['id'] = list(range(edges.shape[0]))
    edges.index = edges['id']
    return edges


def load_geojson(geojson_filename):
    edges = gpd.read_file(geojson_filename)
    edges = edges[~edges['geometry'].isnull()]
    if 'index' in edges.columns:
        edges = edges.drop('index', axis=1)
    edges['id'] = range(edges.shape[0])
    edges['end1'] = None
    edges['end2'] = None
    if "capacity" not in edges.columns:
        edges['capacity'] = None
    edges.index = edges['id']
    edges.index.name = "index"
    print("There are", edges.shape[0], "transport_edges")
    print(edges.crs)
    return edges


def create_nodes_and_update_edges(edges, update=True):
    """Create unique nodes from endpoints using a spatial tolerance and update edge indices."""
    print("Assigning nodes")
    endpoints = gpd.GeoDataFrame({
        "end1": edges.geometry.apply(lambda line: Point(line.coords[0])),
        "end2": edges.geometry.apply(lambda line: Point(line.coords[-1]))
    })

    # Combine all endpoints into one GeoDataFrame
    all_endpoints = gpd.GeoDataFrame(pd.concat([endpoints['end1'], endpoints['end2']]), columns=["geometry"],
                                     crs=edges.crs)

    all_endpoints['geometry_wkt'] = all_endpoints['geometry'].apply(lambda geom: wkt.dumps(geom, rounding_precision=5))
    nodes = all_endpoints.drop_duplicates('geometry_wkt').copy()
    nodes['id'] = range(nodes.shape[0])
    nodes.index = nodes['id']
    nodes['long'] = nodes['geometry'].x
    nodes['lat'] = nodes['geometry'].y

    # add nodes_id into end1 and end2 columns of transport_edges
    end1_wkt = endpoints['end1'].apply(lambda geom: wkt.dumps(geom, rounding_precision=5))
    end2_wkt = endpoints['end2'].apply(lambda geom: wkt.dumps(geom, rounding_precision=5))
    edges['end1'] = end1_wkt.map(nodes.set_index('geometry_wkt')['id'])
    edges['end2'] = end2_wkt.map(nodes.set_index('geometry_wkt')['id'])

    # Update transport_edges' geometry to ensure they start at end1 and end at end2
    if update:
        print('Update geometries')
        edges['geometry'] = edges.apply(
            lambda row: update_linestring(row.geometry, nodes.loc[row['end1']].geometry, nodes.loc[row['end2']].geometry),
            axis=1
        )

    return nodes, edges


def add_km(edges, crs):
    # Project the layer. Watch out, the CRS should be adapted to the country
    edges['km'] = edges.to_crs(crs).length / 1000
    return edges


def get_isolated_edges(edges):
    degree_1_nodes = get_degree_n_nodes(edges, 1)
    degree_1_nodes_with_edge_id = get_edges_from_endpoints(edges, degree_1_nodes)
    nb_degree_1_nodes_per_edge_id = pd.Series(degree_1_nodes_with_edge_id).apply(lambda l: l[0]).value_counts()
    edge_ids_with_2_degree_1_endpoints = nb_degree_1_nodes_per_edge_id[nb_degree_1_nodes_per_edge_id == 2]
    return edge_ids_with_2_degree_1_endpoints.index.to_list()

def remove_isolated_edges(edges):
    isolated_edges = get_isolated_edges(edges)
    print(f"Nb isolated edge: {len(isolated_edges)}")
    edges = edges[~edges['id'].isin(isolated_edges)].copy()
    edges['id'] = range(edges.shape[0])
    edges.index = edges['id']
    return edges


def find_anchor_points(nodes, mode):
    if mode in ['roads', 'pipelines']:
        return nodes
    elif mode in ['maritime', 'airways', 'waterways', 'railways']:
        return nodes[nodes['multimodal_point']]
    else:
        raise ValueError("Wrong mode chosen")


def build_multimodal_links(from_nodes, from_mode, to_nodes, to_mode, max_distance_km=20):
    anchor_from_nodes = find_anchor_points(from_nodes, from_mode)
    anchor_to_nodes = find_anchor_points(to_nodes, to_mode)

    # Find the closest road for each port
    links = []
    projected_anchor_from_nodes = anchor_from_nodes.to_crs(epsg=3857)
    projected_anchor_to_nodes = anchor_to_nodes.to_crs(epsg=3857)
    for _, target_node in projected_anchor_to_nodes.iterrows():
        # Find the closest road point to the current port
        closest_from_nodes = projected_anchor_from_nodes.distance(target_node.geometry).idxmin()
        distance_to_closest_node = projected_anchor_from_nodes.loc[[closest_from_nodes]].distance(target_node.geometry).iloc[0]
        if distance_to_closest_node <= max_distance_km * 1000:
            closest_point = projected_anchor_from_nodes.loc[closest_from_nodes].geometry

            # Create a LineString from port to the closest road point
            link = LineString([target_node.geometry, closest_point])
            links.append({"geometry": link})

    # Create a new GeoDataFrame for the links
    links_gdf = gpd.GeoDataFrame(links, crs=projected_anchor_to_nodes.crs)
    links_gdf = links_gdf.to_crs(epsg=4326)
    links_gdf['multimodes'] = from_mode + '-' + to_mode
    return links_gdf


def cut_line_at_distance(line, distance):
    """Cut a LineString at a specified distance from its starting point.
    Returns a tuple: (segment_before, segment_after)."""
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]

    coords = list(line.coords)
    for i, p in enumerate(coords):
        current_distance = line.project(Point(p))
        if current_distance == distance:
            return [LineString(coords[:i + 1]), LineString(coords[i:])]
        if current_distance > distance:
            # Interpolate the cut point
            cp = line.interpolate(distance)
            seg1 = LineString(coords[:i] + [(cp.x, cp.y)])
            seg2 = LineString([(cp.x, cp.y)] + coords[i:])
            return [seg1, seg2]


def split_line_by_two_distances(line, d1, d2):
    """Split a line into three segments at distances d1 and d2 along the line.
    Assumes 0 < d1 < d2 < line.length."""
    seg1, remainder = cut_line_at_distance(line, d1)
    seg2, seg3 = cut_line_at_distance(remainder, d2 - d1)
    return seg1, seg2, seg3


def split_line_with_intersection(line, border, middle_length):
    """
    Splits a line that intersects the polygon's boundary into three segments.
    The middle segment will be approximately middle_length (in the same unit as the line)
    and have the intersection point as its midpoint.
    """
    # Get the polygon's boundary
    if not line.intersects(border):
        return []  # No intersection, return the original line.

    # Find the intersection point(s)
    intersection = line.intersection(border)
    # If multiple points, pick the first one (or refine as needed)
    if intersection.geom_type == 'MultiPoint':
        intersections = list(intersection.geoms)
        if len(intersections) % 2:
            return []
        else:
            intersection = intersections[0]
    elif intersection.geom_type != 'Point':
        # In case of unexpected geometry type, skip splitting
        return []

    # Project the intersection onto the line to get the distance along the line.
    d = line.project(intersection)
    half = middle_length / 2.0
    # Define distances for splitting, ensuring they remain within the line's extent.
    d1 = max(0, d - half)
    d2 = min(line.length, d + half)

    # If the calculated d1 or d2 hit the endpoints, then splitting as desired isn’t possible.
    if d1 == 0 or d2 == line.length:
        return []

    # Split the line at the two distances.
    return split_line_by_two_distances(line, d1, d2)


def create_borders(edges, countries_gdf: gpd.GeoDataFrame):
    print("Creating border crossing edges")
    borders = countries_gdf.union_all().boundary
    border_crossing_length = 0.0005
    new_geometries = {}
    for i, edge in transport_edges.iterrows():
        if edge['geometry'].intersects(borders):
            cut_lines = split_line_with_intersection(edge['geometry'], borders, middle_length=border_crossing_length)
            if len(cut_lines) == 3:
                new_geometries[i] = cut_lines

    if len(new_geometries) > 0:
        all_new_edges = format_new_edges(edges, new_geometries, border=True)
        return add_new_edges_to_gdf(edges, new_geometries, all_new_edges)[0]
    else:
        return edges


def identify_multimodal_points(nodes, edges):
    multimodal_points = get_degree_n_nodes(edges, n=1)
    print(f"There are {len(multimodal_points)} multimodal_points")
    nodes['multimodal_point'] = False
    nodes.loc[multimodal_points, 'multimodal_point'] = True
    return nodes


logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

TOLERANCE = 0.0001

region = "Cambodia"
do_multimodes = True
transport_mode = 'roads'


special_railway_suffix = ""
present_modes = {"ECA": ['railways', 'pipelines', 'maritime', 'roads'],
                 "Cambodia": ['railways', 'waterways', 'maritime', 'roads']}

if region == "Italia":
    input_folder = os.path.join('..', '..', '..', '..', 'Research', 'Elisa', "disruptsc-ita", "input", "Italy",
                                "Transport")
else:
    input_folder = os.path.join('..', '..', '..', region, 'Data', 'Structured', "Transport")

output_folder = os.path.join('..', "..", "disruptsc", 'input', region, 'Transport')

projected_crs = {
    'Cambodia': 3857,
    'Ecuador': 31986,
    'ECA': 3857,
    'Italia': 32633
}
projected_crs = projected_crs[region]


filename = os.path.join(input_folder, transport_mode.capitalize(),
                        f"raw_{transport_mode}_edges{special_railway_suffix}.geojson")
logging.info(f'Reading {filename}')
transport_edges = load_geojson(filename)

#country_boundaries_filename = os.path.join('..', '..', '..', region, 'Data', 'Structured', "Admin", 'gadm41_country_boundaries.geojson')
country_boundaries = gpd.read_file(Path("data/countries.geojson"))

transport_nodes, transport_edges = create_nodes_and_update_edges(transport_edges)

if transport_mode not in ['waterways', "railways"]:  # keep river-crossing ferry boats
    transport_edges = remove_isolated_edges(transport_edges)
transport_edges = remove_self_loop(transport_edges)
transport_edges = remove_duplicated_geometry(transport_edges)
transport_edges = keep_one_edge_same_endpoints(transport_edges)

transport_edges, new_ids = treat_overlapping_edges(transport_edges)
transport_edges, new_ids = treat_overlapping_edges(transport_edges, new_ids)
print(f"Check that all resulting geometries are LineString: "
      f"{transport_edges['geometry'].apply(lambda geom: geom.geom_type == 'LineString').all()}")

transport_edges, new_ids = treat_intersecting_edges(transport_edges)
transport_edges, new_ids = treat_intersecting_edges(transport_edges, new_ids)
print(f"Check that all resulting geometries are LineString: "
      f"{transport_edges['geometry'].apply(lambda geom: geom.geom_type == 'LineString').all()}")

# add columns
if transport_mode == 'roads':
    transport_edges['surface'] = 'paved'
    for col in ['surface', 'class', 'disruption', 'name', 'special']:
        if col not in transport_edges.columns:
            transport_edges[col] = None

transport_nodes, transport_edges = create_nodes_and_update_edges(transport_edges, update=True)
if transport_mode in ["roads", "maritime"]:
    transport_edges = remove_degree_2_nodes(transport_edges)
    transport_nodes, transport_edges = create_nodes_and_update_edges(transport_edges, update=False)
transport_edges = remove_self_loop(transport_edges)
transport_edges = keep_one_edge_same_endpoints(transport_edges)
transport_nodes, transport_edges = create_nodes_and_update_edges(transport_edges, update=False)
if transport_mode in ["roads"]:
    transport_edges = remove_isolated_edges(transport_edges)

if transport_mode in ["maritime", "railways", "airways", "waterways"]:
    transport_nodes = identify_multimodal_points(transport_nodes, transport_edges)

transport_edges = create_borders(transport_edges, country_boundaries)
transport_edges = add_km(transport_edges, projected_crs)

filename = os.path.join(input_folder, f"{transport_mode}_edges{special_railway_suffix}.geojson")
transport_edges.to_file(filename, driver="GeoJSON", index=False)
filename = os.path.join(input_folder, f"{transport_mode}_nodes{special_railway_suffix}.geojson")
transport_nodes.to_file(filename, driver="GeoJSON", index=False)

print(f"There are {transport_edges.shape[0]} transport_edges")
print(transport_edges.head())

# Exports
# export(nodes, transport_edges, input_folder, output_folder, transport_mode, special_suffix)


# multimode = ["roads", "maritime"]
# suffix0 = "_osmsimp"
# suffix1 = "_mc"

if do_multimodes:
    roads_nodes = gpd.read_file(os.path.join(input_folder, "roads_nodes.geojson"))
    if "maritime" in present_modes[region]:
        maritime_nodes = gpd.read_file(os.path.join(input_folder, "maritime_nodes.geojson"))
        maritime_nodes['multimodal_point'] = maritime_nodes['multimodal_point'].map(lambda x: bool(x) if pd.notna(x) else False)
    if "railways" in present_modes[region]:
        railways_nodes = gpd.read_file(os.path.join(input_folder, f"railways_nodes{special_railway_suffix}.geojson"))
        railways_nodes['multimodal_point'] = railways_nodes['multimodal_point'].map(lambda x: bool(x) if pd.notna(x) else False)
    if "airways" in present_modes[region]:
        airways_nodes = gpd.read_file(os.path.join(input_folder, "airways_nodes.geojson"))
        airways_nodes['multimodal_point'] = airways_nodes['multimodal_point'].map(lambda x: bool(x) if pd.notna(x) else False)
    if "waterways" in present_modes[region]:
        waterways_nodes = gpd.read_file(os.path.join(input_folder, "waterways_nodes.geojson"))
    if "pipelines" in present_modes[region]:
        pipelines_nodes = gpd.read_file(os.path.join(input_folder, "pipelines_nodes.geojson"))

    multimodal_edges_list = []
    if "maritime" in present_modes[region]:
        multimodal_edges_list += [build_multimodal_links(roads_nodes, "roads", maritime_nodes, "maritime")]
    if "railways" in present_modes[region]:
        multimodal_edges_list += [build_multimodal_links(roads_nodes, "roads", railways_nodes, "railways")]
    if "waterways" in present_modes[region]:
        multimodal_edges_list += [build_multimodal_links(roads_nodes, "roads", waterways_nodes, "waterways")]
    if "pipelines" in present_modes[region]:
        multimodal_edges_list += [build_multimodal_links(roads_nodes, "roads", pipelines_nodes, "pipelines")]
    if "airways" in present_modes[region]:
        multimodal_edges_list += [build_multimodal_links(roads_nodes, "roads", airways_nodes, "airways")]
    if ("railways" in present_modes[region]) and ("waterways" in present_modes[region]):
        multimodal_edges_list += [build_multimodal_links(railways_nodes, "railways", waterways_nodes, "waterways")]
    if ("railways" in present_modes[region]) and ("maritime" in present_modes[region]):
        multimodal_edges_list += [build_multimodal_links(railways_nodes, "railways", maritime_nodes, "maritime")]
    if ("waterways" in present_modes[region]) and ("maritime" in present_modes[region]):
        multimodal_edges_list += [build_multimodal_links(waterways_nodes, "waterways", maritime_nodes, "maritime")]
    if ("pipelines" in present_modes[region]) and ("maritime" in present_modes[region]):
        multimodal_edges_list += [build_multimodal_links(pipelines_nodes, "pipelines", maritime_nodes, "maritime")]
    multimodal_edges = pd.concat(multimodal_edges_list)
    multimodal_edges['km'] = 0.1  # no impact
    multimodal_edges['id'] = range(multimodal_edges.shape[0])
    multimodal_edges['capacity'] = None
    multimodal_edges.to_file(os.path.join(input_folder, "multimodal_edges.geojson"), driver="GeoJSON",
                             index=False)
