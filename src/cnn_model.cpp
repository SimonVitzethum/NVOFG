#include "cnn_model.hpp"
#include <cstdio>
#include <cstring>

namespace nvofg {

static uint32_t rd32(FILE* f) { uint32_t v = 0; if (fread(&v, 4, 1, f) != 1) v = 0; return v; }

bool loadCnnModel(const char* path, CnnModel& m) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    char magic[8];
    if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "NVFGW\x01\x00\x00", 8) != 0) {
        std::fclose(f); return false;
    }
    m.version = rd32(f); m.base = rd32(f); m.cin = rd32(f); m.mode = rd32(f);
    uint32_t nt = rd32(f);
    for (uint32_t i = 0; i < nt; ++i) {
        uint32_t nl = rd32(f);
        if (nl == 0 || nl > 256) { std::fclose(f); return false; }
        std::string name(nl, '\0');
        if (std::fread(&name[0], 1, nl, f) != nl) { std::fclose(f); return false; }
        uint32_t nd = rd32(f);
        if (nd == 0 || nd > 8) { std::fclose(f); return false; }
        CnnTensor t;
        long cnt = 1;
        for (uint32_t j = 0; j < nd; ++j) { int d = (int)rd32(f); t.dims.push_back(d); cnt *= d; }
        if (cnt <= 0 || cnt > (1 << 28)) { std::fclose(f); return false; }
        t.half.resize((size_t)cnt);
        if (std::fread(t.half.data(), 2, (size_t)cnt, f) != (size_t)cnt) { std::fclose(f); return false; }
        m.tensors[name] = std::move(t);
    }
    std::fclose(f);
    m.valid = (m.tensors.size() > 0);
    return m.valid;
}

}  // namespace nvofg
