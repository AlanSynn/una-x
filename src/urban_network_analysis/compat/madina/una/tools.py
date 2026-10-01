"""Madina una.tools parity: interim surface (see the package __init__
delta ledger).

Upstream keeps five public names in ``madina/una/tools.py``;
``alternative_paths`` (verbatim below) is the paths-facing one and lands
with the paths task.  ``validate_zonal_ready`` / ``accessibility`` /
``service_area`` join from ``.accessibility`` and ``betweenness`` from
``.betweenness`` when those facade modules land, reproducing the upstream
tools namespace exactly.  The ``from .betweenness import ...`` line of the
upstream import block is ledgered until then; the remaining imports are
upstream's own (their names become module attributes in both trees).
"""

import math
import numpy as np
import geopandas as gpd

from shapely import GeometryCollection
from .paths import turn_o_scope, path_generator  # noqa: F401
from ..zonal import Zonal


def alternative_paths(
    zonal: Zonal,
    origin_id: int,
    search_radius: float | int,
    detour_ratio: float | int = 1,
    turn_penalty: bool = False,
):
    """Generates all alternative patha between an origin and all reachable destinations within a detour from the shortest path

    :param zonal: A zonal object populated with a network, origins, destinations and a graph
    :type zonal: Zonal
    :param origin_id: an ID for the origin used as a start. Must be an ID from the origin layer. 
    :type origin_id: int
    :param search_radius: The maximum distance to search for reachable destinatations. In the same unit as the network CRS.
    :type search_radius: float | int
    :param detour_ratio: A percentage of detour over the shortest path between an origin and a destination when generating alternative paths. , defaults to 1 and only generates the shortest path. Must be greater than or equal to one. if set to a large number, could result in severe performance issues and memory overflow
    :type detour_ratio: float | int, optional
    :param turn_penalty: If True, turn penalty is enabled, defaults to False
    :type turn_penalty: bool, optional
    :return: This function returns a GeoDataFrame of all paths generated to all reachable destinations. The GeoDataFrame has three columns: 
        - `destination`: the destination ID where a path ends
        - `distance`: the weight of this path: reflecting the network settings for network weight, and turn penalty.
        ` `geometry` a column of Shapely GeometryCollection containing all network segments along the path. origin and destination segments are not trimmed but returned whole.
    :rtype: GeoDataFrame
    """
    origin_gdf = zonal.network.nodes[zonal.network.nodes['type'] == 'origin']


    if not isinstance(origin_id, (int, list)):
        raise TypeError(f"Parameter 'origin_id' must be {int} representing an origin id. {type(origin_id)} was given.")
    if origin_id not in origin_gdf['source_id'].values:
        raise ValueError(f"Parameter 'origin_id': is not for an origin included in the network.")

    if not isinstance(search_radius, (int, float)):
        raise TypeError(f"Parameter 'search_radius' must be either {int, float}. {type(search_radius)} was given.")
    elif search_radius < 0:
        raise ValueError(f"Parameter 'search_radius': Cannot be negative. search_radius={search_radius} was given.")

    if not isinstance(detour_ratio, (int, float)):
        raise TypeError(f"Parameter 'detour_ratio' must be either {int, float}. {type(detour_ratio)} was given.")
    elif detour_ratio < 1:
        raise ValueError(f"Parameter 'detour_ratio': Cannot be less than 1. detour_ratio={detour_ratio} was given.")

    if not isinstance(turn_penalty, bool):
        raise TypeError(f"Parameter 'turn_penalty' must either be a boolean True or False, {type(turn_penalty)} was given.")



    o_idx = origin_gdf[origin_gdf['source_id'] == origin_id].iloc[0].name

    path_edges, distances, d_idxs = path_generator(
        network=zonal.network,
        o_idx=o_idx,
        search_radius=search_radius,
        detour_ratio=detour_ratio,
        turn_penalty=turn_penalty
    )

    destination_list = []
    distance_list = []
    path_geometries = []

    for d_idx, destination_distances in distances.items():
        destination_id = zonal.network.nodes.at[d_idx, 'source_id']
        destination_list = destination_list + [destination_id] * len(destination_distances)
        distance_list = distance_list + list(destination_distances)
        for segment_list in path_edges[d_idx]:
            origin_segment_id = int(zonal.network.nodes.at[o_idx, 'nearest_edge_id'])
            destination_segment_id = int(zonal.network.nodes.at[d_idx, 'nearest_edge_id'])
            path_segments = [zonal.network.edges.at[origin_segment_id, 'geometry']] + list(zonal.network.edges.loc[segment_list]['geometry']) + [zonal.network.edges.at[destination_segment_id, 'geometry']]

            path_geometries.append(
                GeometryCollection(
                    path_segments
                )
            )


    destination_gdf = gpd.GeoDataFrame({'destination': destination_list, 'distance': distance_list, 'geometry': path_geometries}, crs = zonal.network.nodes.crs)
    destination_gdf = destination_gdf.sort_values("distance").reset_index(drop=True)
    return destination_gdf

