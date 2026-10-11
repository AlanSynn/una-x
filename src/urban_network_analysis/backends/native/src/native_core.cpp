// NATIVE_CORE pybind11 binding (dossier 08 ABI law).
//
// Narrow binding surface; every array crosses as a dtype-agnostic
// py::array view (never forcecast — unsupported inputs are REFUSED, not
// cast).  All validation happens BEFORE the GIL is released; the
// released region touches only raw pointers, per-worker scratch and the
// cancellation flag — no Python object access.  Owner references stay
// alive because the arrays are held by this function's arguments for the
// whole call.  Structured status codes are translated to Python after
// the GIL is reacquired (the Python route in backends/native owns the
// routing policy: admission refusals fall back to the portable route,
// enforcement failures here are loud internal mismatches).
#include "scope_core.hpp"
#include "isa.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace una_native;

namespace {

// Layout enforcement failure: raised while the GIL is held (binding
// argument checks only); pybind translates it to RuntimeError.  The
// Python route pre-checks layout as POLICY (delegate on refusal), so
// tripping this in the binding is a loud internal mismatch.
struct AbiFailure : std::runtime_error {
    explicit AbiFailure(const std::string& why) : std::runtime_error(why) {}
};

py::dtype k_i64() { return py::dtype::of<std::int64_t>(); }
py::dtype k_f64() { return py::dtype::of<double>(); }
py::dtype k_bool() { return py::dtype::of<bool>(); }

// One borrowed 1-D view after layout enforcement.  Fixed element widths
// (int64/double/bool-as-uint8) — never platform C long.
struct View1D {
    const void* data;
    std::int64_t len;
};

// One borrowed (n, 2) C-contiguous view.
struct View2D {
    const void* data;
    std::int64_t rows;
};

void require_common(py::array& a, const py::dtype& dt, const char* name) {
    if (!a.dtype().equal(dt))
        throw AbiFailure(std::string(name) + " has unsupported dtype (refused, "
                         "not cast)");
    if ((a.flags() & py::array::c_style) == 0)
        throw AbiFailure(std::string(name) + " must be C-contiguous");
}

View1D require_1d(py::array& a, const py::dtype& dt, const char* name) {
    if (a.ndim() != 1) throw AbiFailure(std::string(name) + " must be 1-D");
    require_common(a, dt, name);
    const std::int64_t len = static_cast<std::int64_t>(a.shape(0));
    if (len > 0 &&
        reinterpret_cast<std::uintptr_t>(a.data()) % alignof(std::int64_t) != 0)
        throw AbiFailure(std::string(name) + " is not naturally aligned");
    return View1D{a.data(), len};
}

View1D require_1d_f64(py::array& a, const char* name) {
    if (a.ndim() != 1) throw AbiFailure(std::string(name) + " must be 1-D");
    require_common(a, k_f64(), name);
    const std::int64_t len = static_cast<std::int64_t>(a.shape(0));
    if (len > 0 &&
        reinterpret_cast<std::uintptr_t>(a.data()) % alignof(double) != 0)
        throw AbiFailure(std::string(name) + " is not naturally aligned");
    return View1D{a.data(), len};
}

View1D require_1d_bool(py::array& a, const char* name) {
    if (a.ndim() != 1) throw AbiFailure(std::string(name) + " must be 1-D");
    require_common(a, k_bool(), name);
    return View1D{a.data(), static_cast<std::int64_t>(a.shape(0))};
}

View2D require_2d(py::array& a, const py::dtype& dt, const char* name) {
    if (a.ndim() != 2) throw AbiFailure(std::string(name) + " must be 2-D");
    require_common(a, dt, name);
    if (a.shape(1) != 2)
        throw AbiFailure(std::string(name) + " must have shape (n, 2)");
    const std::int64_t rows = static_cast<std::int64_t>(a.shape(0));
    if (rows > 0 &&
        reinterpret_cast<std::uintptr_t>(a.data()) % alignof(double) != 0)
        throw AbiFailure(std::string(name) + " is not naturally aligned");
    return View2D{a.data(), rows};
}

const CancelFlag* require_cancel(const py::object& cancel) {
    if (cancel.is_none()) return nullptr;
    if (!py::isinstance<CancelFlag>(cancel))
        throw AbiFailure("cancel must be a _native_core.CancelFlag or None");
    return cancel.cast<std::shared_ptr<CancelFlag>>().get();
}

py::dict engagement_dict(std::int32_t status, const Engagement& eng,
                         const std::string& reason) {
    py::dict d;
    d["status"] = py::int_(status);
    d["ok"] = py::bool_(status == STATUS_OK);
    d["reason"] = py::str(reason);
    d["heap_pushes"] = py::int_(static_cast<std::uint64_t>(eng.heap_pushes));
    d["heap_pops"] = py::int_(static_cast<std::uint64_t>(eng.heap_pops));
    d["rows_scanned"] = py::int_(static_cast<std::uint64_t>(eng.rows_scanned));
    d["eligible_collected"] =
        py::int_(static_cast<std::uint64_t>(eng.eligible_collected));
    d["first_incomplete_origin"] = py::int_(eng.first_incomplete_origin);
    return d;
}

// Shared argument assembly for both integrated drivers.
struct Assembled {
    SearchInputs in{};
    double* out = nullptr;
    const CancelFlag* cancel = nullptr;
};

Assembled assemble(py::array& ptr, py::array& vec, py::array& wts,
                   py::array& o_idx, py::array& o_wts, py::array& d_idx,
                   py::array& d_wts, double cutoff, py::object& cancel,
                   py::array& out, py::array* flag /* legacy only */) {
    Assembled a{};
    SearchInputs& in = a.in;

    const View1D v_ptr = require_1d(ptr, k_i64(), "adjacency_pointer");
    const View1D v_vec = require_1d(vec, k_i64(), "adjacency_vector");
    const View1D v_wts = require_1d_f64(wts, "adjacency_vector_weights");
    const View2D v_o_idx = require_2d(o_idx, k_i64(), "o_terminal_idxs");
    const View2D v_o_wts = require_2d(o_wts, k_f64(), "o_terminal_weights");
    const View2D v_d_idx = require_2d(d_idx, k_i64(), "d_terminal_idxs");
    const View2D v_d_wts = require_2d(d_wts, k_f64(), "d_terminal_weights");
    if (v_o_idx.rows != v_o_wts.rows || v_d_idx.rows != v_d_wts.rows)
        throw AbiFailure("terminal index/weight row counts disagree");

    in.ptr = static_cast<const std::int64_t*>(v_ptr.data);
    in.node_count = v_ptr.len - 1;
    if (in.node_count < 0)
        throw AbiFailure("adjacency_pointer needs at least one slot");
    in.vec = static_cast<const std::int64_t*>(v_vec.data);
    in.edge_count = v_vec.len;
    in.wts = static_cast<const double*>(v_wts.data);
    if (v_wts.len != v_vec.len)
        throw AbiFailure("adjacency_vector_weights length != adjacency_vector");
    in.flag = nullptr;
    if (flag != nullptr) {
        const View1D v_flag =
            require_1d_bool(*flag, "adjacynct_vector_network_node");
        if (v_flag.len != v_vec.len)
            throw AbiFailure("adjacynct_vector_network_node length != E");
        in.flag = static_cast<const std::uint8_t*>(v_flag.data);
    }
    in.o_idx = static_cast<const std::int64_t*>(v_o_idx.data);
    in.o_count = v_o_idx.rows;
    in.o_wts = static_cast<const double*>(v_o_wts.data);
    in.d_idx = static_cast<const std::int64_t*>(v_d_idx.data);
    in.d_wts = static_cast<const double*>(v_d_wts.data);
    in.d_count = v_d_idx.rows;
    in.cutoff = cutoff;
    in.max_degree = 0;  // recomputed (and used consistently) inside the core

    // Output: (o_count, d_count) float64 C-contiguous writeable.
    if (out.ndim() != 2) throw AbiFailure("out must be 2-D");
    if (!out.dtype().equal(k_f64())) throw AbiFailure("out must be float64");
    if ((out.flags() & py::array::c_style) == 0)
        throw AbiFailure("out must be C-contiguous");
    if (!out.writeable()) throw AbiFailure("out must be writeable");
    if (static_cast<std::int64_t>(out.shape(0)) != in.o_count ||
        static_cast<std::int64_t>(out.shape(1)) != in.d_count)
        throw AbiFailure("out shape must be (o_count, d_count)");
    a.out = static_cast<double*>(out.mutable_data());
    a.cancel = require_cancel(cancel);
    return a;
}

// Single-origin label vector for the bitwise label battery (test path;
// the GIL stays held — this is not the performance entry).
py::array search_labels_common(py::array& o_row, py::array& o_wts_row,
                               py::array& ptr, py::array& vec, py::array& wts,
                               py::array* flag, double cutoff, bool legacy,
                               std::int64_t domain) {
    if (o_row.ndim() != 1 || o_row.shape(0) != 2 ||
        !o_row.dtype().equal(k_i64()))
        throw AbiFailure("o_row must be (2,) int64");
    if (o_wts_row.ndim() != 1 || o_wts_row.shape(0) != 2 ||
        !o_wts_row.dtype().equal(k_f64()))
        throw AbiFailure("o_wts_row must be (2,) float64");

    SearchInputs in{};
    in.ptr = static_cast<const std::int64_t*>(
        require_1d(ptr, k_i64(), "adjacency_pointer").data);
    in.node_count = static_cast<std::int64_t>(ptr.shape(0)) - 1;
    in.vec = static_cast<const std::int64_t*>(
        require_1d(vec, k_i64(), "adjacency_vector").data);
    in.edge_count = static_cast<std::int64_t>(vec.shape(0));
    in.wts = static_cast<const double*>(
        require_1d_f64(wts, "adjacency_vector_weights").data);
    in.o_idx = static_cast<const std::int64_t*>(o_row.data());
    in.o_count = 1;
    in.o_wts = static_cast<const double*>(o_wts_row.data());
    in.d_idx = nullptr;
    in.d_wts = nullptr;
    in.d_count = 0;
    in.cutoff = cutoff;
    in.flag = nullptr;
    if (legacy) {
        if (flag == nullptr) throw AbiFailure("legacy labels need the flag column");
        const View1D v_flag = require_1d_bool(*flag, "adjacynct_vector_network_node");
        if (v_flag.len != in.edge_count)
            throw AbiFailure("adjacynct_vector_network_node length != E");
        in.flag = static_cast<const std::uint8_t*>(v_flag.data);
    }

    i64 max_degree = 0;
    const std::int32_t vstatus =
        validate_search_inputs(in, legacy, &max_degree);
    if (vstatus != STATUS_OK) throw AbiFailure("input refused by core validation");
    in.max_degree = max_degree;

    py::array_t<double> out(static_cast<py::ssize_t>(domain));
    std::vector<double> labels(static_cast<std::size_t>(domain));
    Engagement eng;
    if (legacy) {
        std::vector<i64> elig_off(
            static_cast<std::size_t>(std::max<i64>(max_degree, 1)));
        std::vector<double> elig_w(
            static_cast<std::size_t>(std::max<i64>(max_degree, 1)));
        ItemHeap heap;
        scope_search_legacy_one(in, /*origin=*/0, labels, elig_off, elig_w,
                                heap, eng);
    } else {
        ItemHeap heap;
        scope_search_corrected_one(in, /*origin=*/0, labels, heap, eng);
    }
    std::memcpy(out.mutable_data(), labels.data(),
                static_cast<std::size_t>(domain) * sizeof(double));
    return out;
}

}  // namespace

PYBIND11_MODULE(_native_core, m) {
    m.doc() = "NATIVE_CORE baseline binary: characterized scope-access search "
              "(kernels/SPEC.md). The fold region stays on the reference.";

    py::class_<CancelFlag, std::shared_ptr<CancelFlag>>(m, "CancelFlag")
        .def(py::init<>())
        .def("cancel", &CancelFlag::cancel)
        .def("is_cancelled", &CancelFlag::cancelled);

    m.def("run_legacy",
          [](std::string route, py::array ptr, py::array vec, py::array wts,
             py::array flag, py::array o_idx, py::array o_wts, py::array d_idx,
             py::array d_wts, double cutoff, std::int32_t threads,
             py::object cancel, py::array out) {
              if (route != "a3" && route != "a1")
                  throw AbiFailure("route must be 'a3' or 'a1'");
              Assembled a = assemble(ptr, vec, wts, o_idx, o_wts, d_idx, d_wts,
                                     cutoff, cancel, out, &flag);
              Engagement eng;
              std::int32_t status;
              {
                  py::gil_scoped_release release;  // pure native work below
                  status = run_integrated_legacy(a.in, route == "a3",
                                                 threads < 1 ? 1 : threads,
                                                 a.cancel, a.out, eng);
              }
              return engagement_dict(status, eng, "");
          });

    m.def("run_corrected",
          [](py::array ptr, py::array vec, py::array wts, py::array o_idx,
             py::array o_wts, py::array d_idx, py::array d_wts, double cutoff,
             std::int32_t threads, py::object cancel, py::array out) {
              Assembled a = assemble(ptr, vec, wts, o_idx, o_wts, d_idx, d_wts,
                                     cutoff, cancel, out, nullptr);
              Engagement eng;
              std::int32_t status;
              {
                  py::gil_scoped_release release;  // pure native work below
                  status = run_integrated_corrected(
                      a.in, threads < 1 ? 1 : threads, a.cancel, a.out, eng);
              }
              std::string reason;
              switch (status) {
                  case STATUS_MALFORMED_GRAPH:
                      reason = "adjacency structure is not a well-formed CSR";
                      break;
                  case STATUS_NONFINITE_COST:
                      reason = "a cost/weight/radius is NaN or infinite";
                      break;
                  case STATUS_NEGATIVE_COST:
                      reason = "a cost/weight/radius is negative";
                      break;
                  case STATUS_OOB_TERMINAL:
                      reason = "a terminal endpoint is outside the node domain";
                      break;
                  default:
                      break;
              }
              return engagement_dict(status, eng, reason);
          });

    m.def("search_labels_legacy",
          [](py::array o_row, py::array o_wts_row, py::array ptr, py::array vec,
             py::array wts, py::array flag, double cutoff, std::string domain,
             std::int64_t tail) {
              if (domain != "a3" && domain != "a1")
                  throw AbiFailure("domain must be 'a3' or 'a1'");
              if (tail < 0) throw AbiFailure("tail must be >= 0");
              if (domain == "a3") tail = 0;
              const i64 node_count =
                  static_cast<std::int64_t>(ptr.shape(0)) - 1;
              if (node_count < 0)
                  throw AbiFailure("adjacency_pointer needs at least one slot");
              const i64 dom = check_domain_size(node_count, tail);
              return search_labels_common(o_row, o_wts_row, ptr, vec, wts,
                                          &flag, cutoff, /*legacy=*/true, dom);
          });

    m.def("search_labels_corrected",
          [](py::array o_row, py::array o_wts_row, py::array ptr, py::array vec,
             py::array wts, double cutoff) {
              const i64 node_count =
                  static_cast<std::int64_t>(ptr.shape(0)) - 1;
              if (node_count < 0)
                  throw AbiFailure("adjacency_pointer needs at least one slot");
              return search_labels_common(o_row, o_wts_row, ptr, vec, wts,
                                          nullptr, cutoff, /*legacy=*/false,
                                          node_count);
          });

    m.def("check_domain_size",
          [](std::int64_t node_count, std::int64_t tail) {
              return check_domain_size(node_count, tail);
          });

    m.def("isa_report",
          []() {
              py::dict d;
              for (const auto& kv : isa_features())
                  d[py::str(kv.first)] = py::bool_(kv.second);
              // "baseline" is the HOST's baseline level (what this CPU
              // supports); "compiled_baseline" is what THIS BINARY
              // requires (build.sh compiles the generic baseline — no
              // -march).  Capability claims consume the latter.
              d[py::str("baseline")] = py::str(isa_baseline_level());
              d[py::str("compiled_baseline")] = py::str(ISA_COMPILED_BASELINE);
              return d;
          });

    m.def("build_fingerprint", []() { return py::str(build_fingerprint()); });
    m.def("abi_version", []() { return py::int_(ABI_VERSION); });

    // Name->value mapping of the STATUS_* enum so the Python route never
    // hard-codes the numeric codes (single source: scope_core.hpp).
    m.def("status_codes", []() {
        const auto code = [](una_native::Status s) {
            return py::int_(static_cast<std::int64_t>(s));
        };
        py::dict d;
        d[py::str("OK")] = code(STATUS_OK);
        d[py::str("MALFORMED_GRAPH")] = code(STATUS_MALFORMED_GRAPH);
        d[py::str("NONFINITE_COST")] = code(STATUS_NONFINITE_COST);
        d[py::str("NEGATIVE_COST")] = code(STATUS_NEGATIVE_COST);
        d[py::str("OOB_TERMINAL")] = code(STATUS_OOB_TERMINAL);
        d[py::str("ERR_ABI")] = code(STATUS_ERR_ABI);
        d[py::str("ERR_MEMORY")] = code(STATUS_ERR_MEMORY);
        d[py::str("ERR_CANCELLED")] = code(STATUS_ERR_CANCELLED);
        d[py::str("ERR_OVERFLOW")] = code(STATUS_ERR_OVERFLOW);
        d[py::str("ERR_INTERNAL")] = code(STATUS_ERR_INTERNAL);
        return d;
    });
}
