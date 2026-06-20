// src/profile.hpp
#pragma once
// Tiny, dependency-free wall-clock profiler for the matmul (linear) kernel.
//
// Off by default. When ie::prof::registry().enabled is true (the bench flips it
// on when IE_PROFILE is set), each linear() call records its elapsed time into a
// bucket keyed by matmul shape [in -> out]. The per-shape split answers "which
// matmul is the wall" with no external profiler, handy where perf isn't
// available (e.g. WSL2 kernels), and, unlike instruction-counting tools, it
// measures real elapsed time, so memory-bandwidth stalls show up.
//
// Single-threaded by design: linear() calls run sequentially (one layer after
// another, one matmul after another), and the Scope records once per CALL on the
// calling thread, not per row, so this stays correct after linear()'s inner
// loop is parallelized with OpenMP in a later step.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace ie {
namespace prof {

struct Bucket {
    int in_dim = 0;
    int out_dim = 0;
    std::uint64_t calls = 0;
    double ns = 0.0;
};

struct Registry {
    bool enabled = false;
    std::vector<Bucket> buckets;  // few distinct shapes; a linear scan is fine

    void add(int in_dim, int out_dim, double ns) {
        for (Bucket& b : buckets) {
            if (b.in_dim == in_dim && b.out_dim == out_dim) {
                ++b.calls;
                b.ns += ns;
                return;
            }
        }
        buckets.push_back(Bucket{in_dim, out_dim, 1, ns});
    }

    void reset() { buckets.clear(); }

    void report(const char* title) const {
        std::vector<Bucket> sorted = buckets;
        std::sort(sorted.begin(), sorted.end(),
                  [](const Bucket& a, const Bucket& b) { return a.ns > b.ns; });
        double tot = 0.0;
        for (const Bucket& b : sorted) tot += b.ns;
        std::printf("[prof] %s -- %.3f ms in linear() across %zu shape(s):\n", title, tot / 1e6,
                    sorted.size());
        for (const Bucket& b : sorted) {
            std::printf("[prof]   linear[%5d -> %5d]  %10.3f ms  %6.1f%%  (%llu calls)\n", b.in_dim,
                        b.out_dim, b.ns / 1e6, tot > 0.0 ? 100.0 * b.ns / tot : 0.0,
                        static_cast<unsigned long long>(b.calls));
        }
    }
};

inline Registry& registry() {
    static Registry r;
    return r;
}

// RAII timer: records [in->out] elapsed into the registry on destruction, but
// only when profiling is enabled (one bool read per call when off).
struct Scope {
    int in_dim;
    int out_dim;
    bool on;
    std::chrono::steady_clock::time_point t0;

    Scope(int in_dim_, int out_dim_) : in_dim(in_dim_), out_dim(out_dim_), on(registry().enabled) {
        if (on) t0 = std::chrono::steady_clock::now();
    }
    ~Scope() {
        if (on) {
            const auto t1 = std::chrono::steady_clock::now();
            registry().add(in_dim, out_dim,
                           std::chrono::duration<double, std::nano>(t1 - t0).count());
        }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

}  // namespace prof
}  // namespace ie