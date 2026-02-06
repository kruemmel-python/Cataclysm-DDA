#include "i18n_api.h"
#include "i18n_engine.h"

#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <cstdint>

namespace {
constexpr uint32_t ABI_VERSION = 1;
constexpr uint32_t BINARY_VERSION_SUPPORTED_MAX = 2;
constexpr size_t RESULT_TOO_LARGE_LIMIT = 16ull * 1024ull * 1024ull;

I18nEngine *as_engine( void *ptr ) {
    return static_cast<I18nEngine *>( ptr );
}

bool begin_engine_call( I18nEngine *eng ) {
    if( !eng ) {
        return false;
    }
    clear_engine_error( eng );
    return true;
}
}

extern "C" {

I18N_API void *i18n_new() {
    return new I18nEngine();
}

I18N_API void i18n_free( void *ptr ) {
    delete static_cast<I18nEngine *>( ptr );
}

I18N_API const char *i18n_last_error( void *ptr ) {
    if( !ptr ) {
        return "ptr == nullptr";
    }
    return static_cast<I18nEngine *>( ptr )->get_last_error();
}

I18N_API int i18n_last_error_copy( void *ptr, char *out_buf, int buf_size ) {
    if( !ptr ) {
        return -1;
    }
    auto *e = as_engine( ptr );
    const char *s = e->get_last_error();
    const int len = static_cast<int>( std::strlen( s ) );

    if( out_buf && buf_size > 0 ) {
        const int n = ( len < ( buf_size - 1 ) ) ? len : ( buf_size - 1 );
        if( n > 0 ) {
            std::memcpy( out_buf, s, static_cast<size_t>( n ) );
        }
        out_buf[n] = '\0';
    }
    return len;
}

static int copy_to_buffer( I18nEngine *engine, const std::string &src, char *out_buf, int buf_size ) {
    const size_t full_len = src.size();
    if( full_len >= RESULT_TOO_LARGE_LIMIT ) {
        set_engine_error( engine, "RESULT_TOO_LARGE" );
        return -1;
    }
    if( full_len > static_cast<size_t>( std::numeric_limits<int>::max() ) ) {
        set_engine_error( engine, "RESULT_TOO_LARGE" );
        return -1;
    }

    const int result_len = static_cast<int>( full_len );
    if( out_buf && buf_size > 0 ) {
        const int n = ( result_len < ( buf_size - 1 ) ) ? result_len : ( buf_size - 1 );
        if( n > 0 ) {
            std::memcpy( out_buf, src.data(), static_cast<size_t>( n ) );
        }
        out_buf[n] = '\0';
    }
    return result_len;
}

I18N_API int i18n_get_meta_locale_copy( void *ptr, char *out_buf, int buf_size ) {
    if( !ptr ) {
        return -1;
    }
    auto *e = as_engine( ptr );
    if( !begin_engine_call( e ) ) {
        return -1;
    }
    return copy_to_buffer( e, e->get_meta_locale(), out_buf, buf_size );
}

I18N_API int i18n_get_meta_fallback_copy( void *ptr, char *out_buf, int buf_size ) {
    if( !ptr ) {
        return -1;
    }
    auto *e = as_engine( ptr );
    if( !begin_engine_call( e ) ) {
        return -1;
    }
    return copy_to_buffer( e, e->get_meta_fallback(), out_buf, buf_size );
}

I18N_API int i18n_get_meta_note_copy( void *ptr, char *out_buf, int buf_size ) {
    if( !ptr ) {
        return -1;
    }
    auto *e = as_engine( ptr );
    if( !begin_engine_call( e ) ) {
        return -1;
    }
    return copy_to_buffer( e, e->get_meta_note(), out_buf, buf_size );
}

I18N_API int i18n_get_meta_plural_rule( void *ptr ) {
    if( !ptr ) {
        return -1;
    }
    auto *e = as_engine( ptr );
    if( !begin_engine_call( e ) ) {
        return -1;
    }
    return static_cast<int>( e->get_meta_plural_rule() );
}

I18N_API int i18n_load_txt( void *ptr, const char *txt_str, int strict ) {
    if( !ptr || !txt_str ) {
        return -1;
    }
    auto *e = as_engine( ptr );
    if( !begin_engine_call( e ) ) {
        return -1;
    }
    return e->load_txt_catalog( txt_str, strict != 0 ) ? 0 : -1;
}

I18N_API int i18n_load_txt_file( void *ptr, const char *path, int strict ) {
    if( !ptr || !path ) {
        return -1;
    }
    auto *e = as_engine( ptr );
    if( !begin_engine_call( e ) ) {
        return -1;
    }
    return e->load_txt_file( path, strict != 0 ) ? 0 : -1;
}

I18N_API int i18n_reload( void *ptr ) {
    if( !ptr ) {
        return -1;
    }
    auto *e = as_engine( ptr );
    if( !begin_engine_call( e ) ) {
        return -1;
    }
    return e->reload() ? 0 : -1;
}

I18N_API unsigned int i18n_abi_version( void ) {
    return ABI_VERSION;
}

I18N_API unsigned int i18n_binary_version_supported_max( void ) {
    return BINARY_VERSION_SUPPORTED_MAX;
}

static std::vector<std::string> build_vec_args( const char **args, int args_len ) {
    std::vector<std::string> vec_args;
    if( !args || args_len <= 0 ) {
        return vec_args;
    }
    vec_args.reserve( static_cast<size_t>( args_len ) );
    for( int i = 0; i < args_len; ++i ) {
        vec_args.emplace_back( args[i] ? args[i] : "" );
    }
    return vec_args;
}

I18N_API int i18n_translate( void *ptr,
                             const char *token,
                             const char **args,
                             int args_len,
                             char *out_buf,
                             int buf_size ) {
    if( !ptr || !token ) {
        return -1;
    }
    auto vec_args = build_vec_args( args, args_len );
    auto *e = as_engine( ptr );
    if( !begin_engine_call( e ) ) {
        return -1;
    }
    const std::string res = e->translate( token, vec_args );
    return copy_to_buffer( e, res, out_buf, buf_size );
}

I18N_API int i18n_translate_plural( void *ptr,
                                    const char *token,
                                    int count,
                                    const char **args,
                                    int args_len,
                                    char *out_buf,
                                    int buf_size ) {
    if( !ptr || !token ) {
        return -1;
    }
    auto vec_args = build_vec_args( args, args_len );
    auto *e = as_engine( ptr );
    if( !begin_engine_call( e ) ) {
        return -1;
    }
    const std::string res = e->translate_plural( token, count, vec_args );
    return copy_to_buffer( e, res, out_buf, buf_size );
}

I18N_API int i18n_print( void *ptr, char *out_buf, int buf_size ) {
    if( !ptr ) {
        return -1;
    }
    auto *e = as_engine( ptr );
    if( !begin_engine_call( e ) ) {
        return -1;
    }
    const std::string res = e->dump_table();
    return copy_to_buffer( e, res, out_buf, buf_size );
}

I18N_API int i18n_find( void *ptr, const char *query, char *out_buf, int buf_size ) {
    if( !ptr || !query ) {
        return -1;
    }
    auto *e = as_engine( ptr );
    if( !begin_engine_call( e ) ) {
        return -1;
    }
    const std::string res = e->find_any( query );
    return copy_to_buffer( e, res, out_buf, buf_size );
}

I18N_API int i18n_check( void *ptr, char *report_buf, int report_size ) {
    if( !ptr ) {
        return -1;
    }
    auto *e = as_engine( ptr );
    if( !begin_engine_call( e ) ) {
        return -1;
    }
    int code = 0;
    const std::string rep = e->check_catalog_report( code );
    copy_to_buffer( e, rep, report_buf, report_size );
    return code;
}

I18N_API int i18n_export_binary( void *ptr, const char *path ) {
    if( !ptr || !path ) {
        return -1;
    }
    auto *e = as_engine( ptr );
    if( !begin_engine_call( e ) ) {
        return -1;
    }
    return e->export_binary_catalog( path ) ? 0 : -1;
}

}
