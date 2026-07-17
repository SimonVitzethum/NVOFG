// Loader for a trained learned-FG model (.nvfgw) used by the NVOFG_INTERP_CNN backend (Plan A B4).
// Parses the versioned fp16 format the training exporter writes (training/export.py). The backend
// (CUDA/coopmat) uploads these tensors and runs the fusion-net residual over the classical warp.
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace nvofg {

struct CnnTensor {
    std::vector<int>      dims;   // e.g. {Cout, Cin, 3, 3}
    std::vector<uint16_t> half;   // fp16 weights, row-major
    int count() const { int p = 1; for (int d : dims) p *= d; return p; }
};

struct CnnModel {
    uint32_t version = 0, base = 0, cin = 0, mode = 0;   // mode: 0=interp, 1=extrap
    std::map<std::string, CnnTensor> tensors;
    bool valid = false;
    size_t paramCount() const { size_t n = 0; for (auto& kv : tensors) n += kv.second.count(); return n; }
};

// Parse a .nvfgw file. Returns false on a missing/corrupt file or bad magic.
bool loadCnnModel(const char* path, CnnModel& out);

}  // namespace nvofg
