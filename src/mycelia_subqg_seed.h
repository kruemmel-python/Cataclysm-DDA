#pragma once
#ifndef CATA_SRC_MYCELIA_SUBQG_SEED_H
#define CATA_SRC_MYCELIA_SUBQG_SEED_H

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

namespace mycelia {

struct subqg_seed_result {
    std::int64_t seed = 0;
    bool unsigned_output = false;
    bool used_fallback = false;
    std::string error;
};

subqg_seed_result generate_subqg_seed( std::optional<std::uint64_t> base_seed,
                                       int gpu_index,
                                       std::chrono::milliseconds timeout,
                                       bool unsigned_output );

void log_subqg_startup_status();

} // namespace mycelia

#endif
