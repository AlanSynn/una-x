"""Madina una.tools parity: interim surface (see the package __init__
delta ledger).

Upstream keeps five public names in ``madina/una/tools.py``.
``validate_zonal_ready`` / ``accessibility`` / ``service_area`` /
``alternative_paths`` are verbatim below (the first three delivered by
the accessibility task, the last by the paths task); ``betweenness``
joins from ``.betweenness`` with MADINA_FLOW, reproducing the upstream
tools namespace exactly.  INTERIM import-line delta (ledgered): upstream
line 7 reads ``from .betweenness import paralell_betweenness_exposure,
parallel_access``; the facade imports ``parallel_access`` only until
MADINA_FLOW delivers ``paralell_betweenness_exposure``, then the line
becomes the verbatim two-name form.  ``accessibility`` consumes the
imported ``parallel_access`` exactly as upstream does.
"""

import math
import numpy as np
import geopandas as gpd

from shapely import GeometryCollection
from .paths import turn_o_scope, path_generator  # noqa: F401
# INTERIM: upstream line 7 also imports paralell_betweenness_exposure;
# that name lands with MADINA_FLOW, then this line becomes verbatim.
from .betweenness import parallel_access  # noqa: F401
from ..zonal import Zonal


def validate_zonal_ready(zonal: Zonal):
    if not isinstance(zonal, Zonal):
        raise TypeError(f"Parameter 'zonal' must be {Zonal}. {type(zonal)} was given.")
    
    if len(zonal.layers.layers) == 0:
        raise ValueError("Zonal object does not have layers, call zonal.load_layer(name='layer_name', source='layer_source' to add layers")

    
    node_gdf = zonal.network.nodes

    if (node_gdf is None) or (node_gdf[node_gdf["type"] == "street_node"].shape[0] == 0):
        raise ValueError("Zonal object does not have network nodes and edges, call zonal.create_street_network() first.")

    if node_gdf[node_gdf["type"] == "origin"].shape[0] == 0:
        raise ValueError("Zonal object does not have origin nodes, call zonal.insert_node(layer_name, label='origin') to add origins from layers.")
    
    if node_gdf[node_gdf["type"] == "destination"].shape[0] == 0:
        raise ValueError("Zonal object does not have destination nodes, call zonal.insert_node(layer_name, label='destination') to add destinations from layers.")


    if zonal.network.d_graph is None:
        raise ValueError("Zonal object does not have a d_graph, call zonal..create_graph() to create graphs first.")
    return

def accessibility(
    zonal: Zonal,
    search_radius: float | int,
    destination_weight: str = None,
    alpha:float=1,
    beta:float=None, 
    save_reach_as: str = None, 
    save_gravity_as: str = None,
    knn_weights: list | str = None, 
    knn_plateau: int | float = None,
    save_knn_access_as: str = None,
    closest_facility: bool = False,
    save_closest_facility_as: str = None, 
    save_closest_facility_distance_as: str = None, 
    turn_penalty: bool = False,
    num_cores: int = 1,
) -> None:
    """Measures accessibility metrics like reach and gravity to reachable destinations within a search radius to all origins in the network. 

    :param zonal: A zonal object populated with a network, origins, destinations and a graph
    :type zonal: Zonal
    :param search_radius: the maximum search distance for accessible destinations from an origin, measured as a network distance in the same units as the network's CRS, defaults to None
    :type search_radius: float | int, optional
    :param destination_weight: destination weight, must be an attribute in the destination layer, defaults to None: equal weight of 1 for all destinations by default.
    :type weight: str, optional
    :param alpha: in gravity calculations, the alpha term increases the importance of destination weight by applying a power. default is 1, destination weight is not adjusted, defaults to 1
    :type alpha: float, optional
    :param beta: In gravity calculations, the beta parameter represent the sensitivity to walk. a smaller beta value means that people are less sensitive to walking. When units are in meters, a typical beta value ranges between 0.001 (Low sensitivity) and 0.004 (High sensitivity), defaults to None
    :type beta: float, optional
    :param save_reach_as: Save the reach metric back to the origin layer as a column with this name, defaults to None
    :type save_reach_as: str, optional
    :param save_gravity_as: Save the gravity metric back to the origin layer as a column with this name, defaults to None
    :type save_gravity_as: str, optional
    :param knn_weights: A list of the form [0.5, 0.25, 0.25], where the length of the list represent the number of suffecient destinations. the values in the list represent each weight the nth destination is given in the KNN access score. , defaults to None
    :type knn_weights: list | str, optional
    :param knn_plateau: apply a decay penalty for destinations that are further than the specified plateau. if set to 0, a distance penalty is applied to all destinations. Cannot be larger than the search radius. Defaults to be equal to the search radius which means no penalty is applied. 
    :type knn_plateau: float | int, optional
    :param save_knn_access_as: Save the KNN Access metric back to the origin layer as a column with this name, defaults to None
    :type save_gravity_as: str, optional
    :param closest_facility: restrict reach and access such that a destination is assigned its closest origin, defaults to False
    :type closest_facility: bool, optional
    :param save_closest_facility_as: if closest_facility=True, save the closest origin ID as a column in the destination layer with this name, defaults to None
    :type save_closest_facility_as: str, optional
    :param save_closest_facility_distance_as: if closest_facility=True, save thw distance to the closest origin as a column in the destination layer with this name, defaults to None
    :type save_closest_facility_distance_as: str, optional
    :param turn_penalty: If True, turn penalty is enabled, defaults to False. Uses the turn penalty amount and turn degree threshold specified in the network
    :type turn_penalty: bool, optional
    :param num_cores: By default, only use a single core, set to as many cores as you want to use for running parallel calculations., defaults to 1
    :type num_cores: int, optional
    """    
    node_gdf = zonal.network.nodes
    origin_gdf = node_gdf[node_gdf["type"] == "origin"]
    origin_layer = origin_gdf.iloc[0]['source_layer']
    destination_gdf = node_gdf[node_gdf["type"] == "destination"]
    destination_layer = destination_gdf.iloc[0]['source_layer']

    validate_zonal_ready(zonal)

    if not isinstance(search_radius, (int, float)):
        raise TypeError(f"Parameter 'search_radius' must be either {int, float}. {type(search_radius)} was given.")
    elif search_radius < 0:
        raise ValueError(f"Parameter 'search_radius': Cannot be negative. search_radius={search_radius} was given.")


    if destination_weight is not None:
        if not isinstance(destination_weight, str):
            raise TypeError(f"Parameter 'destination_weight' must be {str}. {type(destination_weight)} was given.")
        elif destination_weight not in zonal[destination_layer].gdf.columns:
            raise ValueError(f"Parameter 'destination_weight': {destination_weight} not in layer {destination_layer}. Available attributes are: {list(zonal[destination_layer].gdf.columns)}")

    if not isinstance(alpha, (int, float)):
        raise TypeError(f"Parameter 'alpha' must be either {int, float}. {type(alpha)} was given.")
    
    if (beta is not None) and (not isinstance(beta, (int, float))):
        raise TypeError(f"Parameter 'beta' must be either {int, float}. {type(beta)} was given.")

    if (save_reach_as is not None) and (not isinstance(save_reach_as, str)):
        raise TypeError(f"Parameter 'save_reach_as' must be a string. {type(save_reach_as)} was given.")

    if (save_gravity_as is not None): 
        if not isinstance(save_gravity_as, str):
            raise TypeError(f"Parameter 'save_gravity_as' must be a string. {type(save_gravity_as)} was given.")
        if (beta is None):
            raise ValueError("Please specify parameter 'beta' when 'save_gravity_as' is provided")
        

    ##KNN input validation: 
    if save_knn_access_as is None:
        knn_weights=None
    else:
        if knn_weights is None:
            raise ValueError("Please specify parameter 'knn_weights' when 'save_knn_access_as' is provided")
        elif isinstance(knn_weights, str):
            knn_weights = knn_weights[1:-1].split(',')
            knn_weights = [float(x) for x in knn_weights]
        elif isinstance(knn_weights, list):
            knn_weights = knn_weights
        else:
            raise ValueError("knn_weight should be a list of numerical values like [0.5, 0.25, 0.25]")
        
        if knn_plateau is None:
            knn_plateau = search_radius
        elif not isinstance(knn_plateau, (int, float)):
            raise TypeError(f"Parameter 'knn_plateau' must be either {int, float}. {type(knn_plateau)} was given.")
        elif knn_plateau<0:
            raise ValueError(f"Parameter 'knn_plateau': Cannot be negative. knn_plateau={knn_plateau} was given.")
        elif knn_plateau>search_radius:
            raise ValueError(f"Parameter 'knn_plateau': Cannot be larger than search radius. knn_plateau={knn_plateau}, search_radius={search_radius} was given.")
        elif beta is None:
            raise ValueError("Please specify parameter 'beta' when 'knn_plateau' is provided")


    #closest_facility input validation
    if not isinstance(closest_facility, bool):
        raise TypeError(f"Parameter 'closest_facility' must either be a boolean True or False, {type(closest_facility)} was given.")
    elif not closest_facility:
        if save_closest_facility_as is not None:
            raise ValueError("Please set parameter 'closest_facility=True' when 'save_closest_facility_as' is provided")
        if save_closest_facility_distance_as is not None:
            raise ValueError("Please set parameter 'closest_facility=True' when 'save_closest_facility_distance_as' is provided")
    else:
        if not isinstance(save_closest_facility_as, str):
            raise TypeError(f"Parameter 'save_closest_facility_as' must be a string. {type(save_closest_facility_as)} was given.")
        
        if not isinstance(save_closest_facility_distance_as, str):
            raise TypeError(f"Parameter 'save_closest_facility_distance_as' must be a string. {type(save_closest_facility_distance_as)} was given.")

    if not isinstance(turn_penalty, bool):
        raise TypeError(f"Parameter 'turn_penalty' must either be a boolean True or False, {type(turn_penalty)} was given.")

    
    parallel_access(
        zonal,
        search_radius=search_radius,
        destination_weight=destination_weight,
        alpha=alpha,
        beta=beta, 
        knn_weights=knn_weights, 
        knn_plateau=knn_plateau,
        closest_facility=closest_facility,
        turn_penalty=turn_penalty,
        num_cores=num_cores,
    )



    # todo: isolate this into network.save_from_nodes_to_layer(zonal, layer_name, name_map={result:param_name..})
    if (save_reach_as is not None) or (save_gravity_as is not None) or (save_knn_access_as is not None): 

        saved_attributes = {}
        if save_reach_as is not None:
            saved_attributes['reach'] = save_reach_as

        if save_gravity_as is not None:
            saved_attributes['gravity'] = save_gravity_as

        if save_knn_access_as is not None:
            saved_attributes['knn_access'] = save_knn_access_as

        for key, value in saved_attributes.items():
            node_gdf.loc[origin_gdf.index, key] = node_gdf.loc[origin_gdf.index, key].fillna(0)
            if value in zonal[origin_layer].gdf.columns:
                zonal[origin_layer].gdf.drop(columns=[value], inplace=True)

        zonal[origin_layer].gdf = zonal[origin_layer].gdf.join(
            node_gdf.loc[origin_gdf.index,['source_id'] + list(saved_attributes.keys()) ].set_index("source_id").rename(columns=saved_attributes)
        )
        zonal[origin_layer].gdf.index = zonal[origin_layer].gdf.index.astype(int)



    if (save_closest_facility_as is not None) or (save_closest_facility_distance_as is not None): 


        saved_attributes = {}
        if save_closest_facility_as is not None:
            saved_attributes['closest_facility'] = save_closest_facility_as

        if save_closest_facility_distance_as is not None:
            saved_attributes['closest_facility_distance'] = save_closest_facility_distance_as

        for key, value in saved_attributes.items():
            #node_gdf.loc[destination_gdf.index, key] = node_gdf.loc[destination_gdf.index, key].fillna(0)
            # NA should not be replaced in closest facility/distance. 
            if value in zonal[destination_layer].gdf.columns:
                zonal[destination_layer].gdf.drop(columns=[value], inplace=True)

        zonal[destination_layer].gdf = zonal[destination_layer].gdf.join(
            node_gdf.loc[destination_gdf.index,['source_id'] + list(saved_attributes.keys()) ].set_index("source_id").rename(columns=saved_attributes)
        )
        zonal[destination_layer].gdf.index = zonal[destination_layer].gdf.index.astype(int)

    return

def service_area(
    zonal: Zonal,
    search_radius: float, 
    origin_ids: int | list = None,
    turn_penalty: bool = False,
):
    """For each origin, generate a polygon of its service area, defined by the boundary around the destinations it could reach within a specified searchj radius. A list of destinations is returned, as well as the network geometry inside the service area

    :param zonal: A zonal object populated with a network, origins, destinations and a graph
    :type zonal: Zonal
    :param origin_ids: If not provided, the service area for all origins is generated. This parameter can either be the id of an origin as an integer, or a list of origin IDs
    :type origin_ids: int | list
    :param search_radius: the maximum search distance for accessible destinations from an origin, measured as a network distance in the same units as the network's CRS, defaults to None
    :type search_radius: float | int
    :param turn_penalty: If True, turn penalty is enabled, defaults to False
    :type turn_penalty: bool, optional
    :raises ValueError: if an origin id is not for an origin in the origin layer.
    :return: 
        - `destinations`: A dataframe howing a list of destinations for each origin that falls within the search radius
        - `network_edges`: The network geometry that falls withn the service area of all origins
        - `scope_gdf`: for each origin, a polygon of its service ares
    :rtype: GeoDataFrame
    """

    validate_zonal_ready(zonal)

    if not isinstance(search_radius, (int, float)):
        raise TypeError(f"Parameter 'search_radius' must be either {int, float}. {type(search_radius)} was given.")
    elif search_radius < 0:
        raise ValueError(f"Parameter 'search_radius': Cannot be negative. search_radius={search_radius} was given.")

    if (origin_ids is not None) and not isinstance(origin_ids, (int, list)):
        raise TypeError(f"Parameter 'origin_ids' must be either {int, list} representing an origin id or a list of origin ids. {type(origin_ids)} was given.")


    if not isinstance(turn_penalty, bool):
        raise TypeError(f"Parameter 'turn_penalty' must either be a boolean True or False, {type(turn_penalty)} was given.")


    node_gdf = zonal.network.nodes
    edge_gdf = zonal.network.edges

    origin_gdf = node_gdf[node_gdf["type"] == "origin"]
    origin_layer = origin_gdf.iloc[0]['source_layer']

    destination_gdf = node_gdf[node_gdf["type"] == "destination"]
    destination_layer = destination_gdf.iloc[0]['source_layer']
    

    if origin_ids is None:
        o_idxs = list(origin_gdf.index)
    elif isinstance(origin_ids, int):
        # put user input in a list
        o_idxs = list(origin_gdf[origin_gdf['source_id'].isin([origin_ids])].index)
    elif len(set(origin_ids) - set(zonal[origin_layer].gdf.index)) != 0:
        raise ValueError(f"some of the indices given in `origin_ids` are not for an origin {set(origin_ids) - set(zonal[origin_layer].gdf.index)}")
    else: 
        ## convert user input to network IDs
        o_idxs = list(origin_gdf[origin_gdf['source_id'].isin(origin_ids)].index)

    scope_names = []
    scope_geometries = []
    scope_origin_ids = []

    all_network_edges = []
    all_destination_ids = set()
            

    # TODO: This loop assumes all destinations are the same layer, generalize to all layers.
    o_graph = zonal.network.d_graph
    for o_idx in o_idxs:

        
        zonal.network.add_node_to_graph(o_graph, o_idx)


        d_idxs, o_scope, o_scope_paths = turn_o_scope(
            network=zonal.network,
            o_idx=o_idx,
            search_radius=search_radius,
            detour_ratio=1.00, 
            turn_penalty=turn_penalty,
            o_graph=o_graph,
            return_paths=False
        )

        zonal.network.remove_node_to_graph(o_graph, o_idx)

        if (len(d_idxs)) == 0:
            continue

        destination_ids = node_gdf.loc[list(d_idxs.keys())]["source_id"]
        all_destination_ids = all_destination_ids.union(destination_ids)
        

        destination_scope = zonal[destination_layer].gdf.loc[destination_ids]["geometry"].unary_union.convex_hull
        network_scope = node_gdf.loc[list(o_scope.keys())]["geometry"].unary_union.convex_hull
        scope = destination_scope.union(network_scope)

        all_network_edges.append(edge_gdf.clip(scope)["geometry"].unary_union)

        origin_id = node_gdf.at[o_idx, "source_id"]
        origin_geom = zonal[origin_layer].gdf.at[origin_id, 'geometry']

        scope_names.append("service area border")
        scope_names.append("origin")
        scope_geometries.append(scope)
        scope_geometries.append(origin_geom)
        scope_origin_ids.append(origin_id)
        scope_origin_ids.append(origin_id)

    scope_gdf = gpd.GeoDataFrame(
        {
            "name": scope_names,
            "origin_id": scope_origin_ids,
        }
    ).set_geometry(scope_geometries).set_crs(zonal[destination_layer].gdf.crs)

    destinations = zonal[destination_layer].gdf.loc[list(all_destination_ids)]
    network_edges = gpd.GeoDataFrame({}).set_geometry(all_network_edges).set_crs(zonal.layers[destination_layer].gdf.crs).explode(index_parts=False)
    return destinations, network_edges, scope_gdf


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

