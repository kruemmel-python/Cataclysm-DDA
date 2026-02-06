#include "mycelia_subqg_seed.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "debug.h"
#include "filesystem.h"
#include "path_info.h"
#include "rng.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace mycelia {
namespace {

constexpr int RESULT_SUCCESS = 0;

struct dynamic_library {
#ifdef _WIN32
    HMODULE handle = nullptr;
#else
    void *handle = nullptr;
#endif

    ~dynamic_library() {
        close();
    }

    void close() {
#ifdef _WIN32
        if( handle ) {
            FreeLibrary( handle );
            handle = nullptr;
        }
#else
        if( handle ) {
            dlclose( handle );
            handle = nullptr;
        }
#endif
    }

    bool open( const std::filesystem::path &path, std::string &err ) {
        close();
#ifdef _WIN32
        handle = LoadLibraryW( path.wstring().c_str() );
        if( !handle ) {
            err = "LoadLibraryW failed (" + std::to_string( GetLastError() ) + ")";
            return false;
        }
#else
        handle = dlopen( path.c_str(), RTLD_LAZY );
        if( !handle ) {
            const char *msg = dlerror();
            err = msg ? msg : "dlopen failed";
            return false;
        }
#endif
        return true;
    }

    void *symbol( const char *name ) const {
        if( !handle ) {
            return nullptr;
        }
#ifdef _WIN32
        return reinterpret_cast<void *>( GetProcAddress( handle, name ) );
#else
        return dlsym( handle, name );
#endif
    }
};

using myc_init_fn = int (*)();
using myc_get_device_count_fn = int (*)();
using myc_create_context_fn = int (*)( int, void ** );
using myc_set_seed_fn = int (*)( void *, std::uint64_t );
using myc_process_buffer_fn = int (*)( void *, std::uint8_t *, std::size_t, std::size_t );
using myc_destroy_context_fn = void (*)( void * );
using myc_get_last_error_fn = const char *(*)();

struct subqg_api {
    dynamic_library lib;
    myc_init_fn myc_init = nullptr;
    myc_get_device_count_fn myc_get_device_count = nullptr;
    myc_create_context_fn myc_create_context = nullptr;
    myc_set_seed_fn myc_set_seed = nullptr;
    myc_process_buffer_fn myc_process_buffer = nullptr;
    myc_destroy_context_fn myc_destroy_context = nullptr;
    myc_get_last_error_fn myc_get_last_error = nullptr;
};

struct subqg_attempt {
    std::int64_t seed = 0;
    std::string error;
};

std::string last_error( const subqg_api &api ) {
    if( api.myc_get_last_error ) {
        const char *msg = api.myc_get_last_error();
        if( msg ) {
            return std::string( msg );
        }
    }
    return std::string();
}

bool bind_api( subqg_api &api, std::string &err ) {
    api.myc_init = reinterpret_cast<myc_init_fn>( api.lib.symbol( "myc_init" ) );
    api.myc_get_device_count = reinterpret_cast<myc_get_device_count_fn>(
        api.lib.symbol( "myc_get_device_count" ) );
    api.myc_create_context = reinterpret_cast<myc_create_context_fn>(
        api.lib.symbol( "myc_create_context" ) );
    api.myc_set_seed = reinterpret_cast<myc_set_seed_fn>( api.lib.symbol( "myc_set_seed" ) );
    api.myc_process_buffer = reinterpret_cast<myc_process_buffer_fn>(
        api.lib.symbol( "myc_process_buffer" ) );
    api.myc_destroy_context = reinterpret_cast<myc_destroy_context_fn>(
        api.lib.symbol( "myc_destroy_context" ) );
    api.myc_get_last_error = reinterpret_cast<myc_get_last_error_fn>(
        api.lib.symbol( "myc_get_last_error" ) );

    if( !api.myc_init || !api.myc_get_device_count || !api.myc_create_context ||
        !api.myc_set_seed || !api.myc_process_buffer || !api.myc_destroy_context ) {
        err = "Missing required SubQG symbols";
        return false;
    }
    return true;
}

void append_candidate_path( std::vector<std::filesystem::path> &out,
                            const std::filesystem::path &candidate )
{
    if( candidate.empty() ) {
        return;
    }
    for( const std::filesystem::path &existing : out ) {
        if( existing == candidate ) {
            return;
        }
    }
    out.emplace_back( candidate );
}

#ifdef _WIN32
std::filesystem::path executable_directory()
{
    std::array<wchar_t, 4096> module_path {};
    const DWORD copied = GetModuleFileNameW( nullptr, module_path.data(),
                         static_cast<DWORD>( module_path.size() ) );
    if( copied == 0 || copied >= module_path.size() ) {
        return std::filesystem::path();
    }
    return std::filesystem::path( module_path.data(),
                                  module_path.data() + copied ).parent_path();
}
#endif

std::vector<std::filesystem::path> candidate_paths() {
    std::vector<std::filesystem::path> out;
    const std::filesystem::path base_dir = PATH_INFO::base_path().get_unrelative_path();
#ifdef _WIN32
    append_candidate_path( out, base_dir / "CC_OpenCl.dll" );
    append_candidate_path( out, base_dir / "bin" / "CC_OpenCl.dll" );

    std::error_code cwd_err;
    const std::filesystem::path cwd_dir = std::filesystem::current_path( cwd_err );
    if( !cwd_err ) {
        append_candidate_path( out, cwd_dir / "CC_OpenCl.dll" );
        append_candidate_path( out, cwd_dir / "bin" / "CC_OpenCl.dll" );
    }

    const std::filesystem::path exe_dir = executable_directory();
    if( !exe_dir.empty() ) {
        append_candidate_path( out, exe_dir / "CC_OpenCl.dll" );
        append_candidate_path( out, exe_dir / "bin" / "CC_OpenCl.dll" );
    }
#else
#if defined(__APPLE__)
    append_candidate_path( out, base_dir / "bin" / "libCC_OpenCl.dylib" );
#endif
    append_candidate_path( out, base_dir / "bin" / "libCC_OpenCl.so" );
    append_candidate_path( out, base_dir / "bin" / "CC_OpenCl.dll" );
#endif
    return out;
}

bool load_api( subqg_api &api, std::string &err, std::filesystem::path *loaded_path = nullptr ) {
    std::string open_err;
    for( const std::filesystem::path &cand : candidate_paths() ) {
        if( !file_exist( cand ) ) {
            continue;
        }
        if( api.lib.open( cand, open_err ) ) {
            if( bind_api( api, err ) ) {
                if( loaded_path ) {
                    *loaded_path = cand;
                }
                return true;
            } else {
                api.lib.close();
            }
        }
    }
    if( open_err.empty() ) {
        err = "SubQG library not found";
    } else {
        err = open_err;
    }
    return false;
}

std::uint64_t random_u64() {
    std::random_device rd;
    std::uint64_t hi = static_cast<std::uint64_t>( rd() );
    std::uint64_t lo = static_cast<std::uint64_t>( rd() );
    std::uint64_t val = ( hi << 32 ) ^ lo;
    if( val == 0 ) {
        val = ( static_cast<std::uint64_t>( rng_bits() ) << 32 ) ^ rng_bits();
    }
    return val;
}

std::int64_t i64_from_u64( std::uint64_t n ) {
    std::int64_t out = 0;
    std::memcpy( &out, &n, sizeof( out ) );
    return out;
}

subqg_attempt attempt_subqg_seed( std::uint64_t base_seed, int gpu_index, bool unsigned_output ) {
    subqg_attempt attempt;
    (void)unsigned_output;
    subqg_api api;
    std::string err;
    if( !load_api( api, err ) ) {
        attempt.error = err;
        return attempt;
    }

    if( api.myc_init() != RESULT_SUCCESS ) {
        attempt.error = "myc_init failed: " + last_error( api );
        return attempt;
    }

    const int device_count = api.myc_get_device_count();
    if( device_count <= 0 ) {
        attempt.error = "No GPUs available";
        return attempt;
    }

    void *ctx = nullptr;
    const int rc_ctx = api.myc_create_context( gpu_index, &ctx );
    if( rc_ctx != RESULT_SUCCESS || !ctx ) {
        attempt.error = "myc_create_context failed: " + last_error( api );
        return attempt;
    }

    const int rc_seed = api.myc_set_seed( ctx, static_cast<std::uint64_t>( base_seed ) );
    if( rc_seed != RESULT_SUCCESS ) {
        api.myc_destroy_context( ctx );
        attempt.error = "myc_set_seed failed: " + last_error( api );
        return attempt;
    }

    std::array<std::uint8_t, 8> buf {};
    const int rc_proc = api.myc_process_buffer( ctx, buf.data(), buf.size(), 0 );
    api.myc_destroy_context( ctx );
    if( rc_proc != RESULT_SUCCESS ) {
        attempt.error = "myc_process_buffer failed: " + last_error( api );
        return attempt;
    }

    std::uint64_t seed_u64 = 0;
    std::memcpy( &seed_u64, buf.data(), sizeof( seed_u64 ) );
    attempt.seed = i64_from_u64( seed_u64 );
    return attempt;
}

struct seed_job {
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;
    subqg_attempt attempt;
};

} // namespace

subqg_seed_result generate_subqg_seed( std::optional<std::uint64_t> base_seed,
                                       int gpu_index,
                                       std::chrono::milliseconds timeout,
                                       bool unsigned_output )
{
    subqg_seed_result result;
    result.unsigned_output = unsigned_output;

    const std::uint64_t seed = base_seed.value_or( random_u64() );
    auto job = std::make_shared<seed_job>();

    std::thread worker( [job, seed, gpu_index, unsigned_output]() {
        subqg_attempt attempt = attempt_subqg_seed( seed, gpu_index, unsigned_output );
        {
            std::lock_guard<std::mutex> guard( job->mutex );
            job->attempt = std::move( attempt );
            job->done = true;
        }
        job->cv.notify_one();
    } );

    {
        std::unique_lock<std::mutex> lock( job->mutex );
        if( job->cv.wait_for( lock, timeout, [&job]() { return job->done; } ) ) {
            worker.join();
            if( job->attempt.error.empty() ) {
                result.seed = job->attempt.seed;
                DebugLog( D_INFO, D_MAIN ) << "SubQG seed OK (gpu " << gpu_index << ").";
                return result;
            }
            result.error = job->attempt.error;
        } else {
            worker.detach();
            result.error = "timeout";
        }
    }

    result.used_fallback = true;
    const std::uint64_t fallback = random_u64();
    result.seed = i64_from_u64( fallback );
    DebugLog( D_WARNING, D_MAIN ) << "SubQG seed fallback (" << result.error << ").";
    return result;
}

void log_subqg_startup_status()
{
    static std::atomic<bool> logged { false };
    if( logged.exchange( true ) ) {
        return;
    }
    subqg_api api;
    std::string err;
    std::filesystem::path loaded_path;
    if( load_api( api, err, &loaded_path ) ) {
        DebugLog( D_INFO, D_MAIN ) << "SubQG DLL load OK (" << loaded_path.generic_u8string() << ").";
    } else {
        DebugLog( D_WARNING, D_MAIN ) << "SubQG DLL load failed (" << err << ").";
    }
}

} // namespace mycelia
