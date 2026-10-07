/* N00 pybind11 binding for the native trace producer (campaign una_10x
 * N00, dossier 08).
 *
 * Checked int64/f64 interfaces: every array's dtype, rank and
 * C-contiguity are verified here, in Python-visible exceptions, BEFORE
 * the GIL is released; the core call runs GIL-free and its status
 * codes are mapped to Python exceptions only after re-acquiring the
 * GIL.  No NumPy buffer is ever copied: arrays that fail a check are
 * refused, not coerced.
 *
 * Output buffers are malloc'd by the core and owned by per-array
 * py::capsule destructors (explicit owner, no global state, no leaks
 * on partial failure).  The Python wrapper marks them read-only; the
 * core never writes through any input pointer.
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>

#include "trace_producer.h"

namespace py = pybind11;

namespace {

void expect(const bool ok, const std::string &msg) {
    if (!ok) {
        throw std::runtime_error(msg);
    }
}

/* C-contiguity from the public strides API (pybind11 3.x hides the
 * numpy C flags): strides must equal row-major itemsize strides. */
bool strides_c_contiguous(const py::array &a) {
    /* numpy normalizes size-0 strides to 0; empty buffers are
     * trivially contiguous and read by no pointer dereference. */
    if (a.size() == 0) {
        return true;
    }
    py::ssize_t expect =
        (py::ssize_t)py::dtype(a.dtype()).itemsize();
    for (int i = a.ndim() - 1; i >= 0; --i) {
        if (a.strides(i) != expect) {
            return false;
        }
        expect *= a.shape(i);
    }
    return true;
}

void check_array(const py::array &a, const std::string &name,
                 const std::string &dtype, int ndim) {
    expect(py::isinstance<py::array>(a),
           name + ": must be a numpy ndarray");
    expect(a.ndim() == ndim,
           name + ": ndim must be " + std::to_string(ndim) + ", got " +
               std::to_string(a.ndim()));
    const std::string got = (std::string)py::str(py::dtype(a.dtype()));
    expect(got == dtype,
           name + ": dtype must be " + dtype + ", got " + got);
    expect(strides_c_contiguous(a),
           name + ": must be C-contiguous (the producer reads buffers "
                  "zero-copy; coercing would hide a copy)");
}

/* Wrap a malloc'd block in a capsule-owned numpy array.  The capsule
 * is the sole owner; freeing happens in its destructor. */
template <typename T>
py::array own_array(T *data, int64_t n) {
    py::capsule cap(data, [](void *p) { std::free(p); });
    py::array_t<T> arr({(py::ssize_t)n},                       /* shape */
                       {(py::ssize_t)sizeof(T)},              /* strides */
                       (const T *)data,                       /* data  */
                       cap);                                  /* owner */
    return arr;
}

} /* namespace */

PYBIND11_MODULE(una_n00_trace_producer, m) {
    m.doc() = "N00 native trace producer (una_10x): exact AOT port of "
              "the _tr_epoch_scope_search_tile producer region.";

    m.def(
        "run_stream",
        [](py::array o_idx, py::array o_w, py::array adj_ptr,
           py::array adj_vec, py::array adj_w, py::array net,
           py::array d_idx, py::array d_w, py::array rev_ptr,
           py::array rev_vec, double cutoff, int64_t epoch_rollover,
           int64_t tile_origins, int64_t lanes,
           int64_t budget_cand_bytes, int64_t max_degree,
           int cancel) {
            (void)cancel; /* production streams are not canceled; the
                           * flag is exercised at the core ABI level */
            /* ---- checked interface, GIL held ---- */
            check_array(o_idx, "o_terminal_idxs", "int64", 2);
            check_array(o_w, "o_terminal_weights", "float64", 2);
            check_array(adj_ptr, "adjacency_pointer", "int64", 1);
            check_array(adj_vec, "adjacency_vector", "int64", 1);
            check_array(adj_w, "adjacency_vector_weights", "float64",
                        1);
            check_array(net, "adjacency_flag_column", "bool", 1);
            check_array(d_idx, "d_terminal_idxs", "int64", 2);
            check_array(d_w, "d_terminal_weights", "float64", 2);
            check_array(rev_ptr, "reverse_pointer", "int64", 1);
            check_array(rev_vec, "reverse_vector", "int64", 1);

            const int64_t O = o_idx.shape(0);
            const int64_t D = d_idx.shape(0);
            const int64_t N = adj_ptr.shape(0) - 1;
            const int64_t E = adj_vec.shape(0);
            expect(o_idx.shape(1) == 2,
                   "o_terminal_idxs: shape must be (O, 2)");
            expect(o_w.shape(0) == O && o_w.shape(1) == 2,
                   "o_terminal_weights: shape must be (O, 2)");
            expect(N >= 0, "adjacency_pointer: shape must be (N + 1,)");
            expect(adj_w.shape(0) == E,
                   "adjacency_vector_weights: must match "
                   "adjacency_vector length");
            expect(net.shape(0) == E,
                   "adjacency_flag_column: must match adjacency_vector "
                   "length");
            expect(d_idx.shape(1) == 2,
                   "d_terminal_idxs: shape must be (D, 2)");
            expect(d_w.shape(0) == D && d_w.shape(1) == 2,
                   "d_terminal_weights: shape must be (D, 2)");
            expect(rev_ptr.shape(0) == N + 1,
                   "reverse_pointer: shape must be (N + 1,)");
            expect(rev_vec.shape(0) == 2 * D,
                   "reverse_vector: shape must be (2 * D,) "
                   "(both terminal columns, duplicates kept)");
            expect(tile_origins >= 1,
                   "tile_origins: must be >= 1 (caller clamps)");
            expect(lanes >= 1, "lanes: must be >= 1");

            una_n00_input in;
            std::memset(&in, 0, sizeof(in));
            in.o_idx = (const int64_t *)o_idx.data();
            in.o_w = (const double *)o_w.data();
            in.adj_ptr = (const int64_t *)adj_ptr.data();
            in.adj_vec = (const int64_t *)adj_vec.data();
            in.adj_w = (const double *)adj_w.data();
            in.net = (const uint8_t *)net.data();
            in.d_idx = (const int64_t *)d_idx.data();
            in.d_w = (const double *)d_w.data();
            in.rev_ptr = (const int64_t *)rev_ptr.data();
            in.rev_vec = (const int64_t *)rev_vec.data();
            in.O = O;
            in.N = N;
            in.D = D;
            in.E = E;
            in.cutoff = cutoff;
            in.epoch_rollover = epoch_rollover;
            in.tile_origins = tile_origins;
            in.lanes = (int32_t)lanes;
            in.budget_cand_bytes = budget_cand_bytes;
            in.max_degree = max_degree;

            una_n00_output out;
            std::memset(&out, 0, sizeof(out));

            /* ---- GIL-free core call ---- */
            {
                py::gil_scoped_release release;
                una_n00_run_stream(&in, &out, nullptr);
            }

            /* ---- exception mapping, GIL held ---- */
            if (out.status == UNA_N00_ERR_ALLOC) {
                una_n00_output_free(&out);
                throw std::bad_alloc();
            }
            if (out.status == UNA_N00_ERR_ARG) {
                int32_t st = out.status;
                una_n00_output_free(&out);
                throw std::runtime_error(
                    "native producer refused arguments (status " +
                    std::to_string(st) + ")");
            }
            if (out.status == UNA_N00_CANCELED) {
                /* Unreachable through this binding (no cancel pointer
                 * is wired), but the mapping must stay total: never
                 * capsule-wrap a canceled stream's null buffers. */
                una_n00_output_free(&out);
                throw std::runtime_error(
                    "native producer canceled at a tile barrier");
            }

            /* ---- capsule-owned results (explicit owner) ---- */
            py::dict res;
            res["ids"] = own_array(out.out_ids, out.nnz);
            res["dist"] = own_array(out.out_dist, out.nnz);
            res["counts"] = own_array(out.counts, O);
            res["lane_counters"] = own_array(out.lane_counters,
                                             (int64_t)out.lanes_written *
                                                 2);
            res["nnz"] = py::int_(out.nnz);
            res["cand_total"] = py::int_(out.cand_total);
            res["touched_total"] = py::int_(out.touched_total);
            res["tiles_done"] = py::int_(out.tiles_done);
            res["budget_hit"] = py::int_(out.budget_hit);
            res["scratch_peak_bytes"] =
                py::int_(out.scratch_peak_bytes);
            res["lanes"] = py::int_(out.lanes_written);
            res["status"] = py::int_(out.status);
            /* Ownership moved into the capsules; nothing left to
             * free. */
            std::memset(&out, 0, sizeof(out));
            return res;
        },
        py::arg("o_terminal_idxs"), py::arg("o_terminal_weights"),
        py::arg("adjacency_pointer"), py::arg("adjacency_vector"),
        py::arg("adjacency_vector_weights"),
        py::arg("adjacency_flag_column"), py::arg("d_terminal_idxs"),
        py::arg("d_terminal_weights"), py::arg("reverse_pointer"),
        py::arg("reverse_vector"), py::arg("cutoff"),
        py::arg("epoch_rollover"), py::arg("tile_origins"),
        py::arg("lanes"), py::arg("budget_cand_bytes"),
        py::arg("max_degree"), py::arg("cancel") = 0);

    m.def("build_info", []() {
        py::dict info;
        info["module"] = "una_n00_trace_producer";
        info["region"] =
            "_trace_access_scratch._tr_epoch_scope_search_tile + "
            "build_trace tile loop";
#ifdef UNA_N00_BUILD_FLAGS
        info["compile_flags"] = UNA_N00_BUILD_FLAGS;
#endif
#ifdef UNA_N00_BUILD_CC
        info["compiler"] = UNA_N00_BUILD_CC;
#endif
        return info;
    });
}
