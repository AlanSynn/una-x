
from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import geopandas as gpd

from ..Logger import Logger
from ..Topology import Topology
from ..Settings import Settings


# ──────────────────────────────────────────────────────────────────────
# Shared export utilities.
#
# Every export writer resolves an output folder the same way and writes a
# GeoDataFrame to a subset of geojson/feather/csv the same way. Module-level
# functions (not Base methods) so non-Base callers — e.g. UNA.RunBatch's
# composite-output export — can reuse them without an inheritance
# relationship they shouldn't have.
# ──────────────────────────────────────────────────────────────────────

def _writer_concurrency_of(carrier) -> int:
    """Runtime-only writer_concurrency transport (batch worker route).

    The batch worker pins a private ``_writer_concurrency`` attribute on
    the row's Topology (Settings is a validated dataclass — extra
    attributes raise in ``Settings.Validation()``) and pops it again
    before any state bundle is transported (batch/worker.py) — it is
    never a serialized field and never observable state.  Every other
    caller (serial route, parent-side composite export) sees no attribute
    and gets 1: the exact historical inline write path (default-path
    bitwise law).
    """
    wc = getattr(carrier, "_writer_concurrency", 1)
    try:
        wc = int(wc)
    except (TypeError, ValueError):
        return 1
    return wc if wc >= 1 else 1


def _run_write_jobs_parallel(jobs, writer_concurrency: int) -> bool:
    """Overlap independent format writes on a bounded thread pool.

    ``jobs`` is an ordered list of ``(run, log)`` callables in the exact
    serial publication order (feather -> csv -> geojson).  All writes are
    submitted, ALL are joined, and only then are the log lines emitted by
    walking the jobs in serial order — so the log order is the serial
    order regardless of write completion order, and no log line ever
    races a still-running write.  The first failure in serial order is
    re-raised after the pool has fully drained.

    DEF-24 discipline (this lineage predates the DEF-24 queue-hygiene
    fix, verified: ab357c6 is NOT an ancestor of the candidate base): the
    pool is explicitly ``shutdown(wait=True)`` in a ``finally`` BEFORE any
    exception propagates, and every submitted future's result is consumed,
    so no future is ever abandoned and ``concurrent.futures``' atexit
    thread join cannot wedge.  Bounded by construction: at most
    ``min(writer_concurrency, len(jobs))`` threads and ``len(jobs) <= 3``
    tasks.  Threads are joined before return — which is also what makes
    the caller's subsequent staging scan safe.

    Known, contained failure-path divergence: the serial path stops at
    the first failed write, so later formats are never attempted; here
    every submitted write runs to completion, so a failing row may have
    written a later format the serial path would not have.  Batch workers
    stage every write into a private staging directory that is discarded
    when a row fails, so the extra artifacts never publish.
    """
    n = max(1, min(int(writer_concurrency), len(jobs)))
    pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="una-writer")
    try:
        futures = [pool.submit(run) for run, _log in jobs]
        outcomes = []
        for fut in futures:
            try:
                fut.result()
                outcomes.append(None)
            except BaseException as exc:    # noqa: BLE001 — re-raised below
                outcomes.append(exc)
    finally:
        pool.shutdown(wait=True)
    wrote_any = False
    for (_run, log), exc in zip(jobs, outcomes):
        if exc is not None:
            raise exc
        log()
        wrote_any = True
    return wrote_any


def resolve_output_folder(settings: Settings, folder_prefix: str = "") -> str:
    """Resolve settings.output_folder (+ optional timestamp subfolder), create it, return it.

    Paths are normalised via os.path.normpath, which collapses accidental
    duplicate separators (e.g. a leading "//" from a user-supplied prefix
    that ended in "/" being joined onto an absolute-looking path).  POSIX
    treats "//" as an implementation-defined prefix and GDAL/pyogrio
    interprets it as a VSI network path (//host/share/…), silently
    mangling the geojson write while pandas.to_feather tolerates it —
    without this normalisation you get the very confusing "feather saved,
    geojson failed" outcome on the same call.
    """
    output_folder = settings.output_folder
    if settings.output_wStamp:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        output_folder = os.path.join(output_folder, f"{folder_prefix}{timestamp}")
    output_folder = os.path.normpath(output_folder)
    # os.path.normpath preserves POSIX's implementation-defined "//"
    # prefix, but GDAL/pyogrio interprets that as a VSI network path
    # (//host/share/…) and silently mangles geojson writes.  Explicitly
    # collapse any leading duplicate separators.
    while output_folder.startswith('//') and not output_folder.startswith('///'):
        output_folder = output_folder[1:]
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def write_gdf_outputs(
    gdf, settings: Settings, output_folder: str, file_name: str, logger, label: str,
    desc: str = "",
    write_feather: bool = True,
    write_csv: bool = True,
    write_geojson: bool = True,
    csv_drop_geometry: bool = True,
    writer_concurrency: int = 1,
) -> bool:
    """Write `gdf` to feather/csv/geojson per the settings.output_* flags.

    `desc` (e.g. "network-nodes", "routes") is folded into the log message
    ("Saved {desc} feather → ...") to preserve each caller's original
    wording; leave "" for a plain "Saved feather → ...".  Returns whether
    anything was written, so callers needing a fallback-format guarantee
    (e.g. UNA.RunBatch's composite export) can act on it.

    ``writer_concurrency`` > 1 overlaps the per-format writes on a bounded
    thread pool (same bytes, logs emitted after the join in serial order);
    callers derive it from the runtime-only Topology transport via
    :func:`_writer_concurrency_of`.  The default of 1 is the exact
    historical inline path.
    """
    wrote_any = False
    tag = f"{desc} " if desc else ""

    if writer_concurrency > 1:
        n_formats = (
            int(bool(write_feather and settings.output_feather))
            + int(bool(write_csv and settings.output_csv))
            + int(bool(write_geojson and settings.output_geojson
                       and "geometry" in gdf.columns)))
        if n_formats > 1:
            jobs = []
            if write_feather and settings.output_feather:
                path = os.path.join(output_folder, file_name + ".feather")
                jobs.append((
                    lambda gdf=gdf, path=path: gdf.to_feather(path),
                    lambda path=path: logger.log(
                        label, f"Saved {tag}feather → {path}", v=1),
                ))
            if write_csv and settings.output_csv:
                path = os.path.join(output_folder, file_name + ".csv")
                out = (gdf.drop(columns="geometry")
                       if (csv_drop_geometry and "geometry" in gdf.columns)
                       else gdf)
                jobs.append((
                    lambda out=out, path=path: out.to_csv(
                        path, sep=settings.csv_delimiter, index=False),
                    lambda path=path: logger.log(
                        label, f"Saved {tag}CSV → {path}", v=1),
                ))
            if write_geojson and settings.output_geojson \
                    and "geometry" in gdf.columns:
                path = os.path.join(output_folder, file_name + ".geojson")

                def _write_geojson(gdf=gdf, path=path):
                    try:
                        import pyogrio as _pyogrio
                        _pyogrio.write_dataframe(
                            gdf, path, driver="GeoJSON",
                            layer_options={"COORDINATE_PRECISION": "6"},
                        )
                    except ImportError:
                        gdf.to_file(path, driver="GeoJSON")

                jobs.append((
                    _write_geojson,
                    lambda path=path: logger.log(
                        label, f"Saved {tag}GeoJSON → {path}", v=1),
                ))
            # GDAL's GeoJSON driver embeds the output file's BASENAME in
            # the "name" member — writes always target the final staging
            # relpath (never a temp/thread-tagged name); the coordinator's
            # relpath-preserving publish keeps the bytes equal.
            return _run_write_jobs_parallel(jobs, writer_concurrency)
        # a single enabled format has nothing to overlap — historical
        # inline path below.

    if write_feather and settings.output_feather:
        path = os.path.join(output_folder, file_name + ".feather")
        gdf.to_feather(path)
        logger.log(label, f"Saved {tag}feather → {path}", v=1)
        wrote_any = True

    if write_csv and settings.output_csv:
        path = os.path.join(output_folder, file_name + ".csv")
        out = gdf.drop(columns="geometry") if (csv_drop_geometry and "geometry" in gdf.columns) else gdf
        out.to_csv(path, sep=settings.csv_delimiter, index=False)
        logger.log(label, f"Saved {tag}CSV → {path}", v=1)
        wrote_any = True

    if write_geojson and settings.output_geojson and "geometry" in gdf.columns:
        path = os.path.join(output_folder, file_name + ".geojson")
        try:
            import pyogrio as _pyogrio
            _pyogrio.write_dataframe(
                gdf, path, driver="GeoJSON",
                layer_options={"COORDINATE_PRECISION": "6"},
            )
        except ImportError:
            gdf.to_file(path, driver="GeoJSON")
        logger.log(label, f"Saved {tag}GeoJSON → {path}", v=1)
        wrote_any = True

    return wrote_any


def _export_results_parallel(
    engine, settings: Settings, output_folder: str, file_name: str,
    results_data: dict,
) -> bool:
    """Overlap ExportAccessibilityResults' format writes when the row's
    runtime-only ``_writer_concurrency`` transport is > 1 (batch worker
    route).  Builds the same frames and performs the same writes as the
    inline serial body, submits them in the serial publication order
    (feather -> csv -> geojson), joins, then emits the same log lines in
    that order.  Returns False (caller falls through to the untouched
    inline body) unless more than one format is enabled — a single
    enabled format has nothing to overlap.
    """
    wc = _writer_concurrency_of(engine.topology)
    if wc <= 1:
        return False
    has_geometry = 'geometry' in results_data
    n_formats = (
        int(bool(settings.output_feather))
        + int(bool(settings.output_csv))
        + int(bool(settings.output_geojson and has_geometry)))
    if n_formats <= 1:
        return False

    if has_geometry:
        results_gdf = gpd.GeoDataFrame(
            results_data,
            crs=engine.topology.origins.crs
            if hasattr(engine.topology.origins, 'crs') else None)
    else:
        results_df = pd.DataFrame(results_data)

    jobs = []
    if settings.output_feather:
        results_path = os.path.join(output_folder, f"{file_name}.feather")
        if has_geometry:
            jobs.append((
                lambda g=results_gdf, p=results_path: g.to_feather(p),
                lambda p=results_path: engine.logger.log(
                    'Export Results',
                    f"Combined results (with geometry) exported to {p}",
                    v=2),
            ))
        else:
            jobs.append((
                lambda d=results_df, p=results_path: d.to_feather(p),
                lambda p=results_path: engine.logger.log(
                    'Export Results',
                    f"Combined results exported to {p}", v=2),
            ))
    if settings.output_csv:
        csv_path = os.path.join(output_folder, f"{file_name}.csv")
        frame = results_gdf if has_geometry else results_df
        jobs.append((
            lambda f=frame, p=csv_path: f.to_csv(
                p, index=False, sep=settings.csv_delimiter),
            lambda p=csv_path: engine.logger.log(
                'Export Results', f"Combined results exported to {p}", v=2),
        ))
    if settings.output_geojson and has_geometry:
        geojson_path = os.path.join(output_folder, f"{file_name}.geojson")
        jobs.append((
            lambda g=results_gdf, p=geojson_path: g.to_file(
                p, driver='GeoJSON'),
            lambda p=geojson_path: engine.logger.log(
                'Export Results', f"Combined results exported to {p}", v=2),
        ))
    _run_write_jobs_parallel(jobs, wc)
    return True


class Base:

    # Import constants from madina for graph building
    # (imported only when needed in _build_compact_graph_from_topology)

    topology: Topology
    logger : Logger
    
    num_threads : int = None
    num_clusters : int = None

    reach : np.ndarray = None
    gravity_exponential : np.ndarray = None
    gravity_logistic : np.ndarray = None
    knn_access : np.ndarray = None
    od_matrix = None
    # result_prefix : str = ""

    # export_feather = True
    # export_geojson = False 
    # export_csv = False
    # csv_delimiter = ","

    def __init__(self, topology):
        self.topology = topology
        self.logger = self.topology.logger
        self.num_threads = topology.num_threads # i do not think i need that
        self.num_clusters = topology.num_clusters

        self.csv_delimiter = ","

    def UpdateToppology(self, topology: Topology):
        self.topology = topology
        self.logger.log('Base', "Topology updated in Base engine", v=2)

    def Centrality(self, settings: Settings):
        raise NotImplementedError("Do not call base Centrality() directly. Use specific engine implementations (e.g. Accessibility, Flow).")

    def ExportAccessibilityResults(self, settings: Settings, folder_prefix: str = "", file_name: str = ""):
        """
        Export calculated results to disk.
        
        Combines reach, gravity, and knn_access with origin location information
        into a single results table with attributes.
        
        Args:
            settings: Settings object containing export configuration
            folder_prefix: Prefix for output folder names (e.g. "R1500m_")
            file_name: Output file names (e.g. "CentralityResults")
        """
        output_folder = resolve_output_folder(settings, folder_prefix)

        # Build combined results dataframe with origin locations
        results_data = {}
        
        # Add origin location geometry
        if hasattr(self.topology.origins, 'geometry') and self.topology.origins.geometry is not None:
            results_data['geometry'] = self.topology.origins.geometry
        
        # Add accessibility metrics as attributes
        if self.reach is not None:
            results_data[settings.result_prefix + 'reach'] = self.reach
        
        if self.gravity_logistic is not None:
            results_data[settings.result_prefix + 'gravity_logistic'] = self.gravity_logistic

        if self.gravity_exponential is not None:
            results_data[settings.result_prefix + 'gravity_exponential'] = self.gravity_exponential

        knn_sufisx = ""
        if settings.knn_decay and settings.knn_decay.lower() != "none":
            knn_sufisx = "_" + settings.knn_decay

        if self.knn_access is not None:
            results_data[settings.result_prefix + 'knn' + knn_sufisx] = self.knn_access
        
        # Create combined results dataframe
        if results_data:
            # Runtime-only writer_concurrency transport (batch worker
            # route): overlap the format writes when the row requested
            # it; anything else falls through to the inline serial body
            # below, untouched.
            if _export_results_parallel(
                    self, settings, output_folder, file_name, results_data):
                self.logger.log('ExportAccessibilityResults',
                                f"All results exported to {output_folder}",
                                v=1)
                return

            if 'geometry' in results_data:
                results_gdf = gpd.GeoDataFrame(results_data, crs=self.topology.origins.crs if hasattr(self.topology.origins, 'crs') else None)
                if settings.output_feather:
                    results_path = os.path.join(output_folder, f"{file_name}.feather")
                    results_gdf.to_feather(results_path)
                    self.logger.log('Export Results', f"Combined results (with geometry) exported to {results_path}", v=2)
            else:
                results_df = pd.DataFrame(results_data)
                if settings.output_feather:
                    results_path = os.path.join(output_folder, f"{file_name}.feather")
                    results_df.to_feather(results_path)
                    self.logger.log('Export Results', f"Combined results exported to {results_path}", v=2)

            if settings.output_csv:
                csv_path = os.path.join(output_folder, f"{file_name}.csv")
                if 'geometry' in results_data:
                    results_gdf.to_csv(csv_path, index=False, sep=settings.csv_delimiter)
                else:
                    results_df.to_csv(csv_path, index=False, sep=settings.csv_delimiter)
                self.logger.log('Export Results', f"Combined results exported to {csv_path}", v=2)

            if settings.output_geojson and 'geometry' in results_data:
                geojson_path = os.path.join(output_folder, f"{file_name}.geojson")
                results_gdf.to_file(geojson_path, driver='GeoJSON')
                self.logger.log('Export Results', f"Combined results exported to {geojson_path}", v=2)
        
        self.logger.log('ExportAccessibilityResults', f"All results exported to {output_folder}", v=1)

    def ExportODM(
        self,
        settings: Settings,
        folder_prefix: str,
        file_name: str = "ODM",
        format: str = "Sqlite",
        speed: float = 5.0,
    ):
        """Export OD pairwise distance matrix.

        Must be called after OD_Matrix(). Requires origin_uid_column and
        destination_id_column to be set in settings.

        Columns: origin, destination, distance, duration (minutes).

        Args:
            format:        "Sqlite", "feather", "csv", or "tsv".
            speed: km/h used to compute duration column.
        """
        import sqlite3

        if self.od_matrix is None:
            raise ValueError("No OD matrix — call OD_Matrix() first.")

        orig_uid_col = (settings.origin_uid_column or '').strip()
        dest_id_col  = (settings.destination_id_column or '').strip()
        if not orig_uid_col:
            raise ValueError("ExportODM requires settings.origin_uid_column to be set.")
        if not dest_id_col:
            raise ValueError("ExportODM requires settings.destination_id_column to be set.")

        origin_uids = np.asarray(self.topology.origins.uid)
        dest_ids    = np.asarray(self.topology.destinations.uid)

        if isinstance(self.od_matrix, pd.DataFrame):
            df = self.od_matrix[['origin_id', 'destination_id', 'distance']].copy()
            df['origin']      = origin_uids[df['origin_id'].values]
            df['destination'] = dest_ids[df['destination_id'].values]
            df = df[['origin', 'destination', 'distance']]
        else:
            o_idx, d_idx = np.where(np.isfinite(self.od_matrix) & (self.od_matrix >= 0))
            df = pd.DataFrame({
                'origin':      origin_uids[o_idx],
                'destination': dest_ids[d_idx],
                'distance':    self.od_matrix[o_idx, d_idx],
            })

        meters_per_minute = float(speed) * 1000.0 / 60.0
        df['duration'] = df['distance'] / meters_per_minute

        output_folder = resolve_output_folder(settings, folder_prefix)

        fmt = (format or 'Sqlite').lower()

        if fmt == 'sqlite':
            db_path = os.path.join(output_folder, f"{file_name}.sqlite")
            conn = sqlite3.connect(db_path)
            df.to_sql('OD', conn, if_exists='replace', index=False)
            conn.close()
            self.logger.log('ExportODM', f"→ {db_path} ({len(df)} pairs)", v=1)

        elif fmt == 'feather':
            feather_path = os.path.join(output_folder, f"{file_name}.feather")
            df.to_feather(feather_path)
            self.logger.log('ExportODM', f"→ {feather_path} ({len(df)} pairs)", v=1)

        elif fmt == 'csv':
            csv_path = os.path.join(output_folder, f"{file_name}.csv")
            df.to_csv(csv_path, sep=',', index=False)
            self.logger.log('ExportODM', f"→ {csv_path} ({len(df)} pairs)", v=1)

        elif fmt == 'tsv':
            tsv_path = os.path.join(output_folder, f"{file_name}.tsv")
            df.to_csv(tsv_path, sep='\t', index=False)
            self.logger.log('ExportODM', f"→ {tsv_path} ({len(df)} pairs)", v=1)

        else:
            raise ValueError(f"Unknown format '{fmt}'. Use 'Sqlite', 'feather', 'csv', or 'tsv'.")

    # ──────────────────────────────────────────────────────────────────
    # Flow export
    # ──────────────────────────────────────────────────────────────────

    def ExportFlowResult(self, settings, folder_prefix="", file_name="flow"):
        if self.edge_flow is None:
            raise ValueError("No results — call Centrality() first.")
        output_folder = resolve_output_folder(settings, folder_prefix)

        prefix   = settings.result_prefix
        col_name = prefix + "flow"
        net = self.topology.network
        crs = net.geometry.crs if hasattr(net.geometry, "crs") else None
        return_directional = bool(settings.flow_return_directional)
        cols = {
            "edge_id": np.arange(len(net.weights), dtype=np.int64),
            col_name: self.edge_flow,
        }
        if return_directional and self.edge_flow_AB is not None:
            cols[col_name + "_AB"] = self.edge_flow_AB
            cols[col_name + "_BA"] = self.edge_flow_BA
        gdf = gpd.GeoDataFrame(cols, geometry=net.geometry.values, crs=crs)

        write_gdf_outputs(gdf, settings, output_folder, file_name, self.logger,
                          "Flow",
                          writer_concurrency=_writer_concurrency_of(
                              self.topology))

        if settings.flow_compute_node_flow and self.node_flow is not None:
            self._write_network_nodes_flow(
                settings, output_folder, file_name, self.node_flow,
            )

        if getattr(self, "observer_flow_total", None) is not None:
            self._write_observer_points(
                settings, output_folder, file_name,
            )

        if getattr(self, "obstacle_hits_total", None) is not None:
            self._write_obstacle_points_usage(
                settings, output_folder, file_name,
            )

        records = getattr(self, "_origin_tracking_records", None)
        if records is not None and len(records) > 0:
            self._write_destinations_used_origins(
                settings, output_folder, file_name, records,
            )

        if settings.flow_output_routes:
            if getattr(self, "_route_records", None):
                self._write_route_alternatives(settings, folder_prefix, file_name)
            else:
                self.logger.log(
                    "Flow",
                    "flow_output_routes=True but no routes were recorded "
                    "— check that origins reach their destinations within "
                    "search_radius (and that route ids match when "
                    "flow_route_id_column is set).", v=1,
                )

    def _write_network_nodes_flow(
        self, settings, output_folder, file_name, node_bw,
    ):
        net   = self.topology.network
        crs   = net.geometry.crs if hasattr(net.geometry, "crs") else None
        n_net = self._n_network_nodes
        xy    = net.node_points[:n_net, :2]
        from shapely.geometry import Point
        geom = [Point(float(x), float(y)) for x, y in xy]
        cols = {
            "node_id":   np.arange(n_net, dtype=np.int64),
            "node_flow": node_bw,
        }
        gdf      = gpd.GeoDataFrame(cols, geometry=geom, crs=crs)
        out_name = file_name + "_network_nodes"
        write_gdf_outputs(
            gdf, settings, output_folder, out_name, self.logger, "Flow",
            desc="network-nodes", write_csv=False,
            writer_concurrency=_writer_concurrency_of(
                self.topology),
        )
        nz = int((node_bw > 0).sum())
        self.logger.log("Flow",
            f"  node flow: {nz}/{n_net} nodes nonzero, "
            f"min={node_bw.min():.4f}, max={node_bw.max():.4f}, "
            f"mean={node_bw.mean():.4f}.", v=1)

    def _write_observer_points(
        self, settings, output_folder, file_name,
    ):
        obs = self.topology.observer_points
        crs = obs.geometry.crs if hasattr(obs.geometry, "crs") else None
        cols = {
            "observer_idx": np.arange(len(obs.geometry), dtype=np.int64),
            "flow_AB":      self.observer_flow_AB,
            "flow_BA":      self.observer_flow_BA,
            "flow_total":   self.observer_flow_total,
        }
        if getattr(obs, "uid", None) is not None:
            cols["observer_uid"] = np.asarray(obs.uid)
        gdf      = gpd.GeoDataFrame(cols, geometry=obs.geometry.values, crs=crs)
        out_name = file_name + "_observer_points"
        write_gdf_outputs(
            gdf, settings, output_folder, out_name, self.logger, "Flow",
            desc="observer-points", write_csv=False,
            writer_concurrency=_writer_concurrency_of(
                self.topology),
        )

    def _write_obstacle_points_usage(
        self, settings, output_folder, file_name,
    ):
        obst = self.topology.obstacles
        crs  = obst.geometry.crs if hasattr(obst.geometry, "crs") else None
        cols = {
            "obstacle_idx": np.arange(len(obst.geometry), dtype=np.int64),
            "penalty":      obst.penalty,
            "direction":    obst.direction,
            "hits_AB":      self.obstacle_hits_AB,
            "hits_BA":      self.obstacle_hits_BA,
            "hits_total":   self.obstacle_hits_total,
        }
        if getattr(obst, "uid", None) is not None:
            cols["obstacle_uid"] = np.asarray(obst.uid)
        gdf      = gpd.GeoDataFrame(cols, geometry=obst.geometry.values, crs=crs)
        out_name = file_name + "_obstacle_points_usage"
        write_gdf_outputs(
            gdf, settings, output_folder, out_name, self.logger, "Flow",
            desc="obstacle-usage", write_csv=False,
            writer_concurrency=_writer_concurrency_of(
                self.topology),
        )

    def _write_destinations_used_origins(self,
        settings, output_folder, file_name, records,
    ):
        from collections import defaultdict
        dest_to_origins: dict = defaultdict(list)
        dest_to_weights: dict = defaultdict(float)
        for origin_uid, dest_idx, o_weight in records:
            dest_to_origins[dest_idx].append(origin_uid)
            dest_to_weights[dest_idx] += float(o_weight)

        dests  = self.topology.destinations
        n_dest = len(dests.geometry)
        n_used_origins  = np.zeros(n_dest, dtype=np.int32)
        used_origin_ids = [""] * n_dest
        total_origin_w  = np.zeros(n_dest, dtype=np.float64)
        for d_idx, uids in dest_to_origins.items():
            if 0 <= d_idx < n_dest:
                seen, seen_set = [], set()
                for u in uids:
                    s = str(u)
                    if s not in seen_set:
                        seen.append(s); seen_set.add(s)
                n_used_origins[d_idx]  = len(seen)
                used_origin_ids[d_idx] = "|".join(seen)
                total_origin_w[d_idx]  = dest_to_weights[d_idx]

        crs  = dests.geometry.crs if hasattr(dests.geometry, "crs") else None
        cols = {
            "dest_idx":            np.arange(n_dest, dtype=np.int64),
            "n_used_origins":      n_used_origins,
            "used_trip_origins":   used_origin_ids,
            "total_origin_weight": total_origin_w,
        }
        dst_col = (settings.destination_id_column or "").strip()
        if dst_col and getattr(dests, "dest_id", None) is not None:
            cols[dst_col] = dests.dest_id
        gdf      = gpd.GeoDataFrame(cols, geometry=dests.geometry.values, crs=crs)
        out_name = file_name + "_destinations_used_origins"
        write_gdf_outputs(
            gdf, settings, output_folder, out_name, self.logger, "Flow",
            desc="destinations-used-origins", write_csv=False,
            writer_concurrency=_writer_concurrency_of(
                self.topology),
        )
        n_used = int((n_used_origins > 0).sum())
        self.logger.log("Flow",
            f"  destinations tracking: {n_used}/{n_dest} destinations "
            f"received trips from {len(records)} (origin, dest) records.", v=1)

    def _write_route_alternatives(self, settings, folder_prefix="", file_name="flow"):
        """Write the complete generated routes as <file_name>_routes.*.

        One row per alternative path:
          route_id    — the value shared by the origin and destination
                        (flow_route_id_column), or the origin uid when no
                        pairing column is set.
          origin_uid / dest_uid — endpoints of the OD pair.
          alt_rank    — 1 = shortest, 2..K = successive alternatives.
          route_cost  — actual (un-penalized) path cost, connector
                        partials included.
          n_edges / edge_ids — the network edge ids composing the route.
          geometry    — merged LineString of the route's edges.  Note:
                        the first/last SNAP EDGES are included in full
                        (geometries are not trimmed at the exact origin/
                        destination snap points).
        """
        from shapely.ops import linemerge

        output_folder = resolve_output_folder(settings, folder_prefix)

        net       = self.topology.network
        geoms_src = net.geometry.values
        crs       = net.geometry.crs if hasattr(net.geometry, "crs") else None
        dest_uids = np.asarray(self.topology.destinations.uid)

        route_ids, o_uids, d_uids = [], [], []
        ranks, costs, n_edges_col, edge_id_strs, geoms = [], [], [], [], []

        for (route_id, o_uid, d_arr_idx, rank, cost, packed) in self._route_records:
            eids = [int(p) >> 1 for p in packed]
            route_ids.append(route_id)
            o_uids.append(o_uid)
            d_uids.append(dest_uids[d_arr_idx])
            ranks.append(int(rank))
            costs.append(float(cost))
            n_edges_col.append(len(eids))
            edge_id_strs.append(" ".join(str(e) for e in eids))
            merged = linemerge([geoms_src[e] for e in eids])
            geoms.append(merged)

        gdf = gpd.GeoDataFrame(
            {
                "route_id":   route_ids,
                "origin_uid": o_uids,
                "dest_uid":   d_uids,
                "alt_rank":   ranks,
                "route_cost": costs,
                "n_edges":    n_edges_col,
                "edge_ids":   edge_id_strs,
            },
            geometry=geoms, crs=crs,
        )
        gdf = gdf.sort_values(
            ["route_id", "origin_uid", "dest_uid", "alt_rank"]
        ).reset_index(drop=True)

        base = file_name + "_routes"
        write_gdf_outputs(
            gdf, settings, output_folder, base, self.logger, "Flow", desc="routes",
        )

        self.logger.log(
            "Flow",
            f"Route alternatives: {len(gdf)} routes across "
            f"{gdf['route_id'].nunique()} route id(s).", v=1,
        )
