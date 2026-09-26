# Developer and user experience contract

Unchanged unless separately authorized:
- package name and import name,
- public API and Settings fields/defaults,
- current installation workflow,
- supported file formats and output formats,
- normal source-edit workflow,
- no mandatory new compiler,
- no hidden runtime download,
- no required backend selector/environment variable,
- CPU-only execution remains functional.

Optimization-specific developer tooling may live under the packet/tests/benchmarks, but ordinary users must not need to know about it.

Warnings/errors introduced solely by a fast-path admission guard must not appear on ordinary supported baseline inputs. Unsupported fast-path inputs transparently use the original verified implementation. Once an admitted optimized execution begins, genuine internal failure should be surfaced rather than silently recomputed under a different path unless baseline contract explicitly permits recovery.