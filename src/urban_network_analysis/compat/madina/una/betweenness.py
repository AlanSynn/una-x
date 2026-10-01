"""Facade port of madina.una.betweenness — access-engine portion.

VERBATIM slices of the pinned upstream module
(City-Form-Lab/madina @ 8b5c3bd3b1c0048ae8b04054daed92bfb201b9d6, MIT —
see una/__init__.py for the full attribution and delta ledger).

INTERIM DELTA (ledgered in una/__init__.py): this module currently
carries ONLY the access-engine trio (get_origin_properties, one_access,
parallel_access — upstream lines 1016-1323) plus the verbatim module
import block.  Upstream also defines parallel_betweenness,
one_betweenness_2, clockwiseangle_and_distance, betweenness_exposure
and paralell_betweenness_exposure (upstream lines 30-1015); those land
with MADINA_FLOW extending this same file.  Until then the module
namespace is missing those five names vs upstream — recorded by the
access reflection test, not hidden.

Pinned upstream behaviors carried verbatim (do not "fix"):
  - parallel_access shuffles origins with UNSEEDED origin_gdf.sample(frac=1)
    (line ~1196): origin processing order (and therefore closest-facility
    tie resolution) is nondeterministic unless the caller seeds numpy's
    global RNG — the access suite seeds it per arm instead of dropping
    the shuffle.
  - one_access swallows exceptions (prints 'Issues in one access...' +
    traceback) and returns the partial processed_origins.
  - closest_facility assigns each DESTINATION its closest ORIGIN; the
    first origin processed wins distance ties, and distances compare by
    node index keys in dict iteration order.
  - parallel_access num_cores>1 takes the mp.Manager/ProcessPoolExecutor
    path; num_cores==1 calls one_access inline with reporting=True.
"""
# this lets geopandas exclusively use shapely (not pygeos) silences a warning about depreciating pygeos out of geopandas. This is not needed when geopandas 1.0 is released in the future
import os
os.environ['USE_PYGEOS'] = '0'

import time
import concurrent
from concurrent import futures

import multiprocessing as mp

import networkx as nx  ## when one_betweenness_2 is deleted, this import is no longer needed. 
import pandas as pd
import geopandas as gpd
import psutil

import math
import numpy as np
import random

import os
from sys import getsizeof




from ..zonal import Zonal
from ..zonal import Network
from .paths import path_generator, turn_o_scope, bfs_subgraph_generation, wandering_messenger

def get_origin_properties(
        self: Zonal,
        search_radius=800,
        beta=0.001,
        turn_penalty=False,
        o_idx=None,
        d_idxs=None,
        o_graph=None, 
        destination_weight=None,
        alpha=1.0
        ):
    node_gdf = self.network.nodes



    if d_idxs is None:
        d_idxs, _, _ = turn_o_scope(
            self.network,
            o_idx,
            search_radius,
            turn_penalty=turn_penalty,
            o_graph=None,
            return_paths=False
        )
    ## K-neareast Neighbor
    if self.network.knn_weight is not None:

        
        knn_weight = 0
        for neighbor_weight, neighbor_distance in zip(self.network.knn_weight, d_idxs.values()):
            if neighbor_distance < self.network.knn_plateau:
                knn_weight += neighbor_weight
            else:
                knn_weight += neighbor_weight / pow(math.e, (beta * (neighbor_distance-self.network.knn_plateau)))

        ## Vectorizing this could improve effeciency
        #eligible_neighbors = min (len(d_idxs), len(knn_weights))
        #neighbor_weights  = np.array(knn_weights[:eligible_neighbors])
        #neighbor_distances = np.array(list(d_idxs.values())[:eligible_neighbors])
        #knn_weight = neighbor_weights/ pow(math.e, beta * max(neighbor_distances-self.network.knn_plateau, 0))

        node_gdf.at[o_idx, "knn_weight"] = knn_weight *  node_gdf.at[o_idx, "weight"]
        node_gdf.at[o_idx, "knn_access"] = knn_weight

    
    #node_gdf.at[o_idx, "gravity"] = sum(1.0 / pow(math.e, (beta * np.array(list(d_idxs.values())))))
    #node_gdf.at[o_idx, "reach"] = int(len(d_idxs))
        
    

    o_closest_destinations = list(d_idxs.keys())
    o_destination_distances = np.array(list(d_idxs.values()))
    if destination_weight is None:
        #o_destination_weights = node_gdf.loc[o_closest_destinations, "weight"].values
        o_destination_weights = np.array([node_gdf.at[idx, 'weight'] for idx in o_closest_destinations], dtype=np.float64)
    else:
        ## This case could be optimized. Optimal to determine weights at node insertion..
        destination_gdf=node_gdf[node_gdf['type'] == 'destination']
        destination_layer = destination_gdf.iloc[0]['source_layer']
        source_ids = list(node_gdf.loc[o_closest_destinations, "source_id"])
        o_destination_weights = self[destination_layer].gdf.loc[source_ids, destination_weight].fillna(0).values
    
    node_gdf.at[o_idx, "reach"] = o_destination_weights.sum()
    if beta is not None:
        node_gdf.at[o_idx, "gravity"] = (np.power(o_destination_weights, alpha) / np.power(math.e, beta * o_destination_distances)).sum()


    return






def one_access(
    self: Zonal,
    origin_queue = None,
    search_radius: float = None, 
    destination_weight: str = None,
    alpha: float = None,
    beta: float = None, 
    knn_weights: list = None, 
    knn_plateau: int | float = None,
    closest_facility: bool = False,
    turn_penalty: bool = False, 
    reporting: bool = False, 
    ):

    node_gdf = self.network.nodes
    origin_gdf=node_gdf[node_gdf['type'] == 'origin']
    



    o_graph = self.network.d_graph
    
    start = time.time()
    i = 1
    processed_origins = []
    try: 
        while True:
            origin_idx = origin_queue.get()

            if origin_idx == "done":
                origin_queue.task_done()
                break
            processed_origins.append(origin_idx)



            self.network.add_node_to_graph(o_graph, origin_idx)
            d_idxs, _, _ = turn_o_scope(
                network=self.network,
                o_idx=origin_idx,
                search_radius=search_radius,
                detour_ratio=1,
                turn_penalty=turn_penalty,
                o_graph=o_graph,
                return_paths=False
            )
            self.network.remove_node_to_graph(o_graph, origin_idx)

            if len(d_idxs) == 0:
                continue

            d_idxs = dict(sorted(d_idxs.items(), key=lambda item: item[1]))


            if closest_facility:
                for d_idx in d_idxs:
                    if ("closest_facility" not in node_gdf.loc[d_idx]) or (np.isnan(node_gdf.loc[d_idx]["closest_facility"])):
                        node_gdf.at[d_idx, "closest_facility"] = origin_idx
                        node_gdf.at[d_idx, "closest_facility_distance"] = d_idxs[d_idx]
                    elif d_idxs[d_idx] < node_gdf.at[d_idx, "closest_facility_distance"]:
                        node_gdf.at[d_idx, "closest_facility"] = origin_idx
                        node_gdf.at[d_idx, "closest_facility_distance"] = d_idxs[d_idx]

            else:
                self.network.knn_weight = knn_weights
                self.network.knn_plateau = knn_plateau
                get_origin_properties(
                    self,
                    search_radius=search_radius,
                    beta=beta,
                    turn_penalty=turn_penalty,
                    o_idx=origin_idx,
                    d_idxs=d_idxs,
                    o_graph=o_graph,
                    destination_weight=destination_weight,
                    alpha=alpha
                )

            if reporting:
                print (f"Time spent: {round(time.time()-start):,}s [Done {i:,} of {origin_gdf.shape[0]:,} origins ({i  /origin_gdf.shape[0] * 100:4.2f}%)]",  end='\r')
                i = i + 1

    except Exception as ex:
        print ('Issues in one access...')
        import traceback
        traceback.print_exc()

    return node_gdf.loc[processed_origins], node_gdf[node_gdf['type'] == 'destination']




def parallel_access(
    self: Zonal,
    search_radius: float = None, 
    destination_weight: str = None,
    alpha: float = None,
    beta: float = None, 
    knn_weights: list = None, 
    knn_plateau: int | float = None,
    closest_facility: bool = False,
    turn_penalty: bool = False, 
    num_cores: int = None,
):
    node_gdf = self.network.nodes
    origin_gdf = node_gdf[node_gdf["type"] == "origin"]
    origin_gdf = origin_gdf.sample(frac=1)
    origin_gdf.index = origin_gdf.index.astype("int")

    destination_gdf = node_gdf[node_gdf["type"] == "destination"]
    destination_layer = destination_gdf.iloc[0]['source_layer']


    
    with mp.Manager() as manager:

        origin_queue = manager.Queue()

        for o_idx in origin_gdf.index:
            origin_queue.put(o_idx)

        for core_index in range(num_cores):
            origin_queue.put("done")

        if num_cores == 1:
            origin_gdf, destination_gdf = one_access(
                self=self,
                origin_queue=origin_queue,
                search_radius=search_radius,
                destination_weight=destination_weight,
                alpha=alpha,
                beta=beta,
                knn_weights=knn_weights,
                knn_plateau=knn_plateau,
                closest_facility=closest_facility,
                turn_penalty=turn_penalty,
                reporting=True
            )
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:        
                execution_results = [
                    executor.submit(
                        one_access,
                        self=self,
                        origin_queue=origin_queue,
                        search_radius=search_radius,
                        destination_weight=destination_weight,
                        alpha=alpha,
                        beta=beta,
                        knn_weights=knn_weights,
                        knn_plateau=knn_plateau,
                        closest_facility=closest_facility,
                        turn_penalty=turn_penalty,
                        reporting=False
                    ) for core_index in range(num_cores)
                ]

                # Progress update and raising exceptions..
                start = time.time()
                while not all([future.done() for future in execution_results]):
                    time.sleep(0.5)
                    done_so_far = origin_gdf.shape[0] - max(origin_queue.qsize() - num_cores, 0)
                    print (f"Time spent: {round(time.time()-start):,}s [Done {done_so_far:,} of {origin_gdf.shape[0]:,} origins ({done_so_far/origin_gdf.shape[0] * 100:4.2f}%)]",  end='\r')
                    for future in [f for f in execution_results if f.done() and (f.exception() is not None)]: # if a process is done and have an exception, raise it
                        raise (future.exception())
                
                

                ## COnsolidating output from cores
                core_origin_gdfs = []
                core_destination_gdfs = []

                for result in concurrent.futures.as_completed(execution_results):
                   o_gdf, d_gdf =  result.result()
                   core_origin_gdfs.append(o_gdf)
                   core_destination_gdfs.append(d_gdf)

                origin_gdf = pd.concat(core_origin_gdfs)

                node_gdf['reach'] = origin_gdf['reach']
                node_gdf['gravity'] = origin_gdf['gravity']
                if knn_weights is not None:
                    node_gdf["knn_weight"] = origin_gdf["knn_weight"]
                    node_gdf["knn_access"] = origin_gdf["knn_access"]

                if closest_facility:
                    # extracting closest facility over multiple destination_gdf coming from different cores, think about a panda solution instead of nested loops.
                    node_gdf["closest_facility"] = np.nan
                    node_gdf["closest_facility_distance"] = np.nan
                    for core_destination_gdf in core_destination_gdfs:
                        for d_idx in core_destination_gdf.index:
                            if np.isnan(node_gdf.loc[d_idx]["closest_facility"]):
                                node_gdf.at[d_idx, "closest_facility"] = core_destination_gdf.at[d_idx, "closest_facility"]
                                node_gdf.at[d_idx, "closest_facility_distance"] = core_destination_gdf.at[d_idx, "closest_facility_distance"]
                            elif core_destination_gdf.at[d_idx, "closest_facility_distance"] < node_gdf.at[d_idx, "closest_facility_distance"]:
                                node_gdf.at[d_idx, "closest_facility"] = core_destination_gdf.at[d_idx, "closest_facility"]
                                node_gdf.at[d_idx, "closest_facility_distance"] = core_destination_gdf.at[d_idx, "closest_facility_distance"]


                            

                    
                    
    if closest_facility:
        ## adjust origin reach ad gravity based on joint destinations
        for o_idx in origin_gdf.index:
            o_closest_destinations = list(node_gdf[node_gdf["closest_facility"] == o_idx].index)
            o_destination_distances = node_gdf.loc[o_closest_destinations]["closest_facility_distance"].values
            if destination_weight is None:
                o_destination_weights = node_gdf.loc[o_closest_destinations]["weight"].values
            else:
                source_ids = list(node_gdf.loc[o_closest_destinations]["source_id"])
                o_destination_weights = self[destination_layer].gdf.loc[source_ids][destination_weight].fillna(0).values
            
            node_gdf.at[o_idx, "reach"] = sum(o_destination_weights)
            if beta is not None:
                node_gdf.at[o_idx, "gravity"] = sum(pow(o_destination_weights, alpha) / pow(math.e, beta * o_destination_distances))

        ## reverting to layer indexing...
        ## This could be an apply function??
        for d_idx in destination_gdf.index: 
            if not np.isnan(node_gdf.at[d_idx, "closest_facility"]):
                o_idx = int(node_gdf.at[d_idx, "closest_facility"])
                origin_id = node_gdf.at[o_idx, "source_id"]
                node_gdf.at[d_idx, "closest_facility"] = origin_id


    self.network.nodes = node_gdf

    return




