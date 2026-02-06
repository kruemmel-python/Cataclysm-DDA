#include "mycelia_swarm.h"

#include <algorithm>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <random>

#include "mycelia_subqg_seed.h"

namespace mycelia {
namespace {

struct vec2f {
    float x = 0.0f;
    float y = 0.0f;
};

vec2f to_vec( const tripoint_bub_ms &p ) {
    const tripoint raw = p.raw();
    return { static_cast<float>( raw.x ), static_cast<float>( raw.y ) };
}

vec2f dir_vec( const tripoint_bub_ms &from, const tripoint_bub_ms &to ) {
    const tripoint a = from.raw();
    const tripoint b = to.raw();
    return { static_cast<float>( b.x - a.x ), static_cast<float>( b.y - a.y ) };
}

float normalize( vec2f &v ) {
    const float len = std::sqrt( v.x * v.x + v.y * v.y );
    if( len > 0.0f ) {
        v.x /= len;
        v.y /= len;
    }
    return len;
}

float dot( const vec2f &a, const vec2f &b ) {
    return a.x * b.x + a.y * b.y;
}

int chebyshev_dist( const tripoint_bub_ms &a, const tripoint_bub_ms &b ) {
    const tripoint ra = a.raw();
    const tripoint rb = b.raw();
    return std::max( std::abs( ra.x - rb.x ), std::abs( ra.y - rb.y ) );
}

std::mt19937_64 &swarm_rng() {
    static std::mt19937_64 rng = []() {
        const subqg_seed_result seed = generate_subqg_seed( std::nullopt, 0,
                                      std::chrono::milliseconds( 2000 ), true );
        const std::uint64_t v = static_cast<std::uint64_t>( seed.seed );
        return std::mt19937_64( v );
    }();
    return rng;
}

float swarm_random_unit() {
    static std::mutex rng_mutex;
    static std::uniform_real_distribution<float> dist( -1.0f, 1.0f );
    std::lock_guard<std::mutex> guard( rng_mutex );
    return dist( swarm_rng() );
}

} // namespace

SwarmParams swarm_params_from_profile( const SpeciesProfile &profile, float bionic_factor ) {
    SwarmParams params;
    const float factor = std::clamp( bionic_factor, 0.0f, 1.0f );

    const float cohesion_base = 0.25f * profile.food_attraction_mul +
                                0.25f * profile.resource_weight_mul +
                                0.2f * profile.molecule_weight_mul +
                                0.2f * ( 1.0f + profile.mycel_attraction_mul ) +
                                0.1f * profile.deposit_food_mul;

    const float avoidance_base = 0.5f * profile.danger_aversion_mul +
                                 0.2f * profile.deposit_danger_mul +
                                 0.2f * ( 1.0f + profile.over_density_threshold ) +
                                 0.1f * profile.counter_deposit_mul;

    const float exploration_base = 0.4f * profile.exploration_mul +
                                   0.2f * profile.novelty_weight +
                                   0.2f * profile.exploration_delta_mul +
                                   0.2f * profile.mutation_sigma_mul;

    params.cohesion_weight = cohesion_base * factor;
    params.avoidance_weight = avoidance_base * factor;
    params.exploration_weight = exploration_base * factor;
    params.target_weight = 1.8f + 0.7f * profile.dna_binding;
    params.density_threshold = profile.over_density_threshold > 0.0f ?
                               profile.over_density_threshold :
                               params.density_threshold;
    return params;
}

SwarmDecision choose_swarm_step( const tripoint_bub_ms &self_pos,
                                 const tripoint_bub_ms &desired,
                                 const std::vector<tripoint_bub_ms> &neighbors,
                                 const std::vector<tripoint_bub_ms> &candidates,
                                 const SwarmParams &params )
{
    SwarmDecision best{ desired, std::numeric_limits<float>::lowest(), false };
    if( neighbors.empty() || candidates.empty() ) {
        return { desired, 0.0f, false };
    }

    vec2f desired_dir = dir_vec( self_pos, desired );
    const float desired_len = normalize( desired_dir );

    vec2f center{ 0.0f, 0.0f };
    int neighbor_count = 0;
    for( const tripoint_bub_ms &pt : neighbors ) {
        if( chebyshev_dist( self_pos, pt ) > params.neighbor_radius ) {
            continue;
        }
        const vec2f v = to_vec( pt );
        center.x += v.x;
        center.y += v.y;
        neighbor_count++;
    }

    vec2f cohesion_dir{ 0.0f, 0.0f };
    float cohesion_len = 0.0f;
    if( neighbor_count > 0 ) {
        center.x /= neighbor_count;
        center.y /= neighbor_count;
        vec2f self = to_vec( self_pos );
        cohesion_dir = { center.x - self.x, center.y - self.y };
        cohesion_len = normalize( cohesion_dir );
    }

    float density_mul = 1.0f;
    if( neighbor_count > params.density_threshold ) {
        density_mul += ( neighbor_count - params.density_threshold ) * 0.1f;
    }

    for( const tripoint_bub_ms &candidate : candidates ) {
        if( params.max_deviation >= 0 &&
            chebyshev_dist( candidate, desired ) > params.max_deviation ) {
            continue;
        }

        vec2f cand_dir = dir_vec( self_pos, candidate );
        if( normalize( cand_dir ) <= 0.0f ) {
            continue;
        }

        float score = 0.0f;
        if( desired_len > 0.0f ) {
            score += params.target_weight * dot( cand_dir, desired_dir );
        }
        if( cohesion_len > 0.0f ) {
            score += params.cohesion_weight * dot( cand_dir, cohesion_dir );
        }

        float avoid_score = 0.0f;
        for( const tripoint_bub_ms &pt : neighbors ) {
            const int dist = chebyshev_dist( self_pos, pt );
            if( dist <= 0 || dist > params.separation_radius ) {
                continue;
            }
            vec2f to_neighbor = dir_vec( self_pos, pt );
            const float nlen = normalize( to_neighbor );
            if( nlen <= 0.0f ) {
                continue;
            }
            const float toward = dot( cand_dir, to_neighbor );
            avoid_score += ( -toward ) / nlen;
        }
        score += params.avoidance_weight * density_mul * avoid_score;
        if( params.exploration_weight > 0.0f ) {
            score += params.exploration_weight * swarm_random_unit();
        }

        if( score > best.score ) {
            best = { candidate, score, candidate != desired };
        }
    }

    if( best.score == std::numeric_limits<float>::lowest() ) {
        return { desired, 0.0f, false };
    }

    return best;
}

} // namespace mycelia
