# Source and documentation use

Repository authority is pinned in sources.json. Read exact source at the recorded commit; newer
upstream/current dependencies do not silently redefine the legacy oracle. P00 raw/diagnostic
records take precedence over transcribed summaries. New candidate designs are proposals.

External primary documentation consulted for bounded design constraints (2026-10-04):
- https://docs.python.org/3.11/library/multiprocessing.html : Queue feeder vs SimpleQueue,
  spawn lifecycle and serialized result ownership. Use actual deployment Python docs at execution.
- https://gdal.org/en/stable/user/multithreading.html : independent GDAL objects; no sharing
  one dataset across workers. Raster-specific thread-safe facilities do not prove vector safety.
- https://shapely.readthedocs.io/en/stable/migration.html : Shapely 2 geometry immutability;
  this does not make surrounding mutable GeoSeries/array containers immutable.
- https://arrow.apache.org/docs/python/generated/pyarrow.set_cpu_count.html : Arrow compute
  thread pool is a resource variable; I/O pool is separately controlled/recorded when used.

These docs support ownership/resource constraints only. They provide no UNA performance estimate
and no evidence that a particular GDAL lock caused P00 idle. No production dataset was downloaded
or executed during authoring. No citations or package versions are invented by the utility tests.
