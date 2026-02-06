#if defined(LOCALIZE)

#include <cstring>
#include <limits>
#include <utility>

#include "cached_options.h"
#include "debug.h"
#include "filesystem.h"
#include "path_info.h"
#include "translations.h"
#include "translation_manager_impl.h"

namespace
{
constexpr const char *i18n_token_not_found = "TOKEN_NOT_FOUND";
}

std::uint32_t TranslationManager::Impl::Hash( const char *str )
{
    std::uint32_t hash = 5381U;
    while( *str != '\0' ) {
        hash = hash * 33 + ( *str++ );
    }
    return hash;
}

std::optional<std::pair<std::size_t, std::size_t>> TranslationManager::Impl::LookupString(
            const char *query ) const
{
    if( strings.empty() ) {
        return std::nullopt;
    }
    std::uint32_t hash = Hash( query );
    auto it = strings.find( hash );
    if( it == strings.end() ) {
        return std::nullopt;
    }
    for( const std::pair<size_t, size_t> &entry : it->second ) {
        const std::size_t document = entry.first;
        const std::size_t index = entry.second;
        if( strcmp( documents[document].GetOriginalString( index ), query ) == 0 ) {
            return std::optional<std::pair<std::size_t, std::size_t>> { entry };
        }
    }
    return std::nullopt;
}

std::string TranslationManager::Impl::LanguageCodeOfPath( std::string_view path )
{
    const std::size_t end = path.rfind( "/LC_MESSAGES" );
    if( end == std::string::npos ) {
        return std::string();
    }
    const std::size_t begin = path.rfind( '/', end - 1 ) + 1;
    if( begin == std::string::npos ) {
        return std::string();
    }
    return std::string( path.substr( begin, end - begin ) );
}

void TranslationManager::Impl::InvalidateI18nOverrides()
{
    i18n_override_load_attempted = false;
    i18n_override_loaded = false;
}

void TranslationManager::Impl::TryLoadI18nOverrides() const
{
    if( i18n_override_load_attempted ) {
        return;
    }
    i18n_override_load_attempted = true;
    i18n_override_loaded = false;

    std::vector<std::string> candidates;
    const std::string config_dir = PATH_INFO::config_dir();
    const std::string fallback_catalog = config_dir + "i18n_overrides.txt";

    if( !current_language_code.empty() ) {
        const std::string language_catalog = config_dir + "i18n_overrides." + current_language_code + ".txt";
        if( language_catalog != fallback_catalog ) {
            candidates.emplace_back( language_catalog );
        }

        const std::size_t split = current_language_code.find_first_of( "_-" );
        if( split != std::string::npos ) {
            const std::string language_short = current_language_code.substr( 0, split );
            if( !language_short.empty() ) {
                const std::string short_catalog = config_dir + "i18n_overrides." + language_short + ".txt";
                if( short_catalog != language_catalog && short_catalog != fallback_catalog ) {
                    candidates.emplace_back( short_catalog );
                }
            }
        }
    }

    candidates.emplace_back( fallback_catalog );

    for( const std::string &candidate : candidates ) {
        if( !file_exist( candidate ) ) {
            continue;
        }
        if( i18n_override_engine.load_txt_file( candidate, false ) ) {
            i18n_override_loaded = true;
            DebugLog( D_INFO, DC_ALL ) << "[i18n] Loaded override catalog from " << candidate;
        } else {
            DebugLog( D_WARNING, DC_ALL ) << "[i18n] Failed to load override catalog '" << candidate
                                          << "': " << i18n_override_engine.get_last_error();
        }
        return;
    }
}

const char *TranslationManager::Impl::StoreI18nOverrideResult( std::string text ) const
{
    std::string &slot = i18n_override_buffer[i18n_override_buffer_index];
    slot = std::move( text );
    const char *result = slot.c_str();
    i18n_override_buffer_index = ( i18n_override_buffer_index + 1 ) % i18n_override_buffer.size();
    return result;
}

const char *TranslationManager::Impl::TryTranslateI18nToken( const std::string &token ) const
{
    TryLoadI18nOverrides();
    if( !i18n_override_loaded ) {
        return nullptr;
    }
    const std::string translated = i18n_override_engine.translate( token, {} );
    if( std::strcmp( i18n_override_engine.get_last_error(), i18n_token_not_found ) == 0 ) {
        return nullptr;
    }
    return StoreI18nOverrideResult( translated );
}

const char *TranslationManager::Impl::TryTranslateI18nPluralToken( const std::string &token,
        std::size_t n ) const
{
    TryLoadI18nOverrides();
    if( !i18n_override_loaded ) {
        return nullptr;
    }
    const int count = n > static_cast<std::size_t>( std::numeric_limits<int>::max() ) ?
                      std::numeric_limits<int>::max() : static_cast<int>( n );
    const std::string translated = i18n_override_engine.translate_plural( token, count, {} );
    if( std::strcmp( i18n_override_engine.get_last_error(), i18n_token_not_found ) == 0 ) {
        return nullptr;
    }
    return StoreI18nOverrideResult( translated );
}

void TranslationManager::Impl::ScanTranslationDocuments()
{
    std::vector<std::pair<std::string, std::string>> mo_dirs;
    if( dir_exist( PATH_INFO::user_moddir() ) ) {
        DebugLog( D_INFO, DC_ALL ) << "[i18n] Scanning mod translations from " << PATH_INFO::user_moddir();
        for( const std::string & dir
             : get_files_from_path( "LC_MESSAGES", PATH_INFO::user_moddir(), true ) ) {
            mo_dirs.emplace_back( dir, ".mo" );
        }
    }
    if( dir_exist( locale_dir() ) ) {
        DebugLog( D_INFO, DC_ALL ) << "[i18n] Scanning core translations from " << locale_dir();
        for( const std::string &dir : get_files_from_path( "LC_MESSAGES", locale_dir(), true ) ) {
            mo_dirs.emplace_back( dir, "cataclysm-dda.mo" );
        }
    }
    for( const std::pair<std::string, std::string> &entry : mo_dirs ) {
        const std::string &dir = entry.first;
        const std::string &pattern = entry.second;
        std::vector<std::string> mo_dir_files = get_files_from_path( pattern, dir, false, true );
        for( const std::string &file : mo_dir_files ) {
            const std::string lang = LanguageCodeOfPath( file );
            if( mo_files.count( lang ) == 0 ) {
                mo_files[lang] = std::vector<std::string>();
            }
            mo_files[lang].emplace_back( file );
        }
    }
}

void TranslationManager::Impl::Reset()
{
    documents.clear();
    strings.clear();
    strings.max_load_factor( 1.0f );
    InvalidateI18nOverrides();
}

TranslationManager::Impl::Impl()
{
    current_language_code = "en";
}

std::unordered_set<std::string> TranslationManager::Impl::GetAvailableLanguages()
{
    if( mo_files.empty() ) {
        ScanTranslationDocuments();
    }
    std::unordered_set<std::string> languages;
    for( const auto &kv : mo_files ) {
        languages.insert( kv.first );
    }
    return languages;
}

void TranslationManager::Impl::SetLanguage( const std::string &language_code )
{
    if( mo_files.empty() ) {
        ScanTranslationDocuments();
    }
    if( language_code != current_language_code ) {
        current_language_code = language_code;
        if( mo_files.count( current_language_code ) == 0 ) {
            Reset();
        } else {
            LoadDocuments( mo_files[current_language_code] );
        }
    } else {
        InvalidateI18nOverrides();
    }
    TryLoadI18nOverrides();
}

std::string TranslationManager::Impl::GetCurrentLanguage() const
{
    return current_language_code;
}

void TranslationManager::Impl::LoadDocuments( const std::vector<std::string> &files )
{
    Reset();
    for( const std::string &file : files ) {
        try {
            // Skip loading MO files from TEST_DATA mods if not in test mode
            if( !test_mode ) {
                if( file.find( "TEST_DATA" ) != std::string::npos ) {
                    continue;
                }
            }
            if( file_exist( file ) ) {
                documents.emplace_back( file );
            }
        } catch( const InvalidTranslationDocumentException &e ) {
            DebugLog( D_ERROR, DC_ALL ) << e.what();
        }
    }
    for( std::size_t document = 0; document < documents.size(); document++ ) {
        for( std::size_t i = 0; i < documents[document].Count(); i++ ) {
            const char *message = documents[document].GetOriginalString( i );
            if( message[0] != '\0' ) {
                const std::uint32_t hash = Hash( message );
                if( strings.count( hash ) == 0 ) {
                    strings[hash] = std::vector<std::pair<std::size_t, std::size_t>>( 1 );
                }
                strings[hash].emplace_back( document, i );
            }
        }
    }
}

const char *TranslationManager::Impl::Translate( const std::string &message ) const
{
    return Translate( message.c_str() );
}

const char *TranslationManager::Impl::Translate( const char *message ) const
{
    if( const char *override_message = TryTranslateI18nToken( message ) ) {
        return override_message;
    }

    std::optional<std::pair<std::size_t, std::size_t>> entry = LookupString( message );
    if( entry ) {
        const std::size_t document = entry->first;
        const std::size_t string_index = entry->second;
        return documents[document].GetTranslatedString( string_index );
    }
    return message;
}

const char *TranslationManager::Impl::TranslatePlural( const char *singular, const char *plural,
        std::size_t n ) const
{
    if( const char *override_message = TryTranslateI18nPluralToken( singular, n ) ) {
        return override_message;
    }

    std::optional<std::pair<std::size_t, std::size_t>> entry = LookupString( singular );
    if( entry ) {
        const std::size_t document = entry->first;
        const std::size_t string_index = entry->second;
        return documents[document].GetTranslatedStringPlural( string_index, n );
    }
    if( n == 1 ) {
        return singular;
    } else {
        return plural;
    }
}

std::string TranslationManager::Impl::ConstructContextualQuery( const char *context,
        const char *message ) const
{
    std::string query;
    query.reserve( strlen( context ) + 1 + strlen( message ) );
    query.append( context );
    query.append( "\004" );
    query.append( message );
    return query;
}

const char *TranslationManager::Impl::TranslateWithContext( const char *context,
        const char *message ) const
{
    // Context-specific overrides are not encoded in the TXT format yet.
    // A plain token override still applies here as a global fallback.
    if( const char *override_message = TryTranslateI18nToken( message ) ) {
        return override_message;
    }

    std::string query = ConstructContextualQuery( context, message );
    std::optional<std::pair<std::size_t, std::size_t>> entry = LookupString( query.c_str() );
    if( entry ) {
        const std::size_t document = entry->first;
        const std::size_t string_index = entry->second;
        return documents[document].GetTranslatedString( string_index );
    }
    return message;
}

const char *TranslationManager::Impl::TranslatePluralWithContext( const char *context,
        const char *singular,
        const char *plural,
        std::size_t n ) const
{
    // Context-specific overrides are not encoded in the TXT format yet.
    // A plain token override still applies here as a global fallback.
    if( const char *override_message = TryTranslateI18nPluralToken( singular, n ) ) {
        return override_message;
    }

    std::string query = ConstructContextualQuery( context, singular );
    std::optional<std::pair<std::size_t, std::size_t>> entry = LookupString( query.c_str() );
    if( entry ) {
        const std::size_t document = entry->first;
        const std::size_t string_index = entry->second;
        return documents[document].GetTranslatedStringPlural( string_index, n );
    }
    if( n == 1 ) {
        return singular;
    } else {
        return plural;
    }
}

#endif // defined(LOCALIZE)
