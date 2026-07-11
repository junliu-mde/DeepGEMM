#pragma once

#include <filesystem>
#include <memory>
#include <unordered_map>

#include "kernel_runtime.hpp"

namespace deep_gemm {

class KernelRuntimeCache {
    std::unordered_map<std::string, std::shared_ptr<KernelRuntime>> cache;

public:
    // TODO: consider cache capacity
    KernelRuntimeCache() = default;

    std::shared_ptr<KernelRuntime> get(const std::filesystem::path& dir_path) {
        // Hit the runtime cache
        if (const auto iterator = cache.find(dir_path); iterator != cache.end())
            return iterator->second;

        // NOTES: constructing the runtime reads `kernel.cubin` from the cache directory;
        // on distributed filesystems this can fail transiently (e.g. stale NFS handles),
        // so treat any failure as a cache miss and let the caller fall back to compiling
        try {
            if (KernelRuntime::check_validity(dir_path))
                return cache[dir_path] = std::make_shared<KernelRuntime>(dir_path);
        } catch (const std::exception& e) {
            printf("Failed to load cached kernel from %s (%s), treating as a cache miss\n",
                   dir_path.c_str(), e.what());
        }
        return nullptr;
    }
};

static auto kernel_runtime_cache = std::make_shared<KernelRuntimeCache>();

} // namespace deep_gemm
