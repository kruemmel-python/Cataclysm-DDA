#pragma once
#ifndef CATA_SRC_MYCELIA_SWARM_H
#define CATA_SRC_MYCELIA_SWARM_H

#include <vector>

#include "coordinates.h"

namespace mycelia {

struct SpeciesProfile {
    float exploration_mul = 1.0f;
    float food_attraction_mul = 1.0f;
    float danger_aversion_mul = 1.0f;
    float deposit_food_mul = 1.0f;
    float deposit_danger_mul = 1.0f;
    float resource_weight_mul = 1.0f;
    float molecule_weight_mul = 1.0f;
    float mycel_attraction_mul = 0.0f;
    float novelty_weight = 0.0f;
    float mutation_sigma_mul = 1.0f;
    float exploration_delta_mul = 1.0f;
    float dna_binding = 1.0f;
    float over_density_threshold = 0.0f;
    float counter_deposit_mul = 0.0f;
};

struct SwarmParams {
    float cohesion_weight = 0.0f;
    float avoidance_weight = 0.0f;
    float exploration_weight = 0.0f;
    float target_weight = 2.0f;
    float density_threshold = 3.0f;
    int neighbor_radius = 6;
    int separation_radius = 2;
    int max_deviation = 1;
};

struct SwarmDecision {
    tripoint_bub_ms step;
    float score = 0.0f;
    bool used = false;
};

SwarmParams swarm_params_from_profile( const SpeciesProfile &profile, float bionic_factor );

SwarmDecision choose_swarm_step( const tripoint_bub_ms &self_pos,
                                 const tripoint_bub_ms &desired,
                                 const std::vector<tripoint_bub_ms> &neighbors,
                                 const std::vector<tripoint_bub_ms> &candidates,
                                 const SwarmParams &params );

} // namespace mycelia

#endif
