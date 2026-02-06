#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "cata_utility.h"
#include "debug.h"
#include "filesystem.h"
#include "get_version.h"
#include "i18n_engine.h"
#include "options.h"
#include "path_info.h"
#include "system_locale.h"
#include "text_snippets.h"
#include "translations.h"
#include "translation_gendered.h"

// int version/generation that is incremented each time language is changed
// used to invalidate translation cache
static int current_language_version = INVALID_LANGUAGE_VERSION + 1;

int detail::get_current_language_version()
{
    return current_language_version;
}

#if !defined(LOCALIZE)
namespace
{
constexpr const char *i18n_token_not_found = "TOKEN_NOT_FOUND";

I18nEngine &no_localize_i18n_engine()
{
    static I18nEngine engine;
    return engine;
}

bool &no_localize_i18n_loaded()
{
    static bool loaded = false;
    return loaded;
}

std::string &no_localize_i18n_selected_lang()
{
    static std::string selected_lang;
    return selected_lang;
}

std::string &no_localize_i18n_last_config_dir()
{
    static std::string config_dir;
    return config_dir;
}

std::string &no_localize_i18n_last_effective_lang()
{
    static std::string effective_lang;
    return effective_lang;
}

std::array<std::string, 32> &no_localize_i18n_buffer()
{
    static std::array<std::string, 32> buffer {};
    return buffer;
}

std::size_t &no_localize_i18n_buffer_index()
{
    static std::size_t index = 0;
    return index;
}

std::string no_localize_i18n_effective_lang()
{
    const std::string &selected = no_localize_i18n_selected_lang();
    if( !selected.empty() ) {
        return selected;
    }
    return SystemLocale::Language().value_or( "en" );
}

bool no_localize_is_english_language_code( const std::string &language_code )
{
    std::string lower = language_code;
    std::transform( lower.begin(), lower.end(), lower.begin(), []( const unsigned char c ) {
        return static_cast<char>( std::tolower( c ) );
    } );
    return lower == "en" || lower.rfind( "en_", 0 ) == 0 || lower.rfind( "en-", 0 ) == 0;
}

void try_load_no_localize_i18n_overrides()
{
    const std::string config_dir = PATH_INFO::config_dir();
    if( config_dir.empty() ) {
        return;
    }

    const std::string effective_lang = no_localize_i18n_effective_lang();
    std::string &last_config_dir = no_localize_i18n_last_config_dir();
    std::string &last_effective_lang = no_localize_i18n_last_effective_lang();
    bool &loaded = no_localize_i18n_loaded();
    if( config_dir == last_config_dir && effective_lang == last_effective_lang ) {
        return;
    }

    last_config_dir = config_dir;
    last_effective_lang = effective_lang;
    loaded = false;
    I18nEngine &engine = no_localize_i18n_engine();

    std::vector<std::string> candidates;

    if( !effective_lang.empty() ) {
        const std::string language_catalog = config_dir + "i18n_overrides." + effective_lang + ".txt";
        candidates.emplace_back( language_catalog );

        const std::size_t split = effective_lang.find_first_of( "_-" );
        if( split != std::string::npos ) {
            const std::string short_lang = effective_lang.substr( 0, split );
            const std::string short_catalog = config_dir + "i18n_overrides." + short_lang + ".txt";
            if( short_catalog != language_catalog ) {
                candidates.emplace_back( short_catalog );
            }
        }
    }

    // Selecting English should disable generic fallback overrides unless an
    // explicit English catalog exists.
    if( !no_localize_is_english_language_code( effective_lang ) ) {
        const std::string fallback_catalog = config_dir + "i18n_overrides.txt";
        candidates.emplace_back( fallback_catalog );
    }

    for( const std::string &candidate : candidates ) {
        if( !file_exist( candidate ) ) {
            continue;
        }
        if( engine.load_txt_file( candidate, false ) ) {
            loaded = true;
            DebugLog( D_INFO, D_MAIN ) << "[i18n] Loaded no-localize override catalog from " << candidate;
        } else {
            DebugLog( D_WARNING, D_MAIN ) << "[i18n] Failed to load no-localize override catalog '" << candidate
                                          << "': " << engine.get_last_error();
        }
        return;
    }
}

const char *store_no_localize_i18n_text( std::string text )
{
    std::array<std::string, 32> &buffer = no_localize_i18n_buffer();
    std::size_t &index = no_localize_i18n_buffer_index();
    std::string &slot = buffer[index];
    slot = std::move( text );
    const char *result = slot.c_str();
    index = ( index + 1 ) % buffer.size();
    return result;
}
} // namespace

const char *translate_no_localize_lookup( const char *msgid )
{
    if( msgid == nullptr ) {
        return msgid;
    }

    try_load_no_localize_i18n_overrides();
    if( !no_localize_i18n_loaded() ) {
        return msgid;
    }

    I18nEngine &engine = no_localize_i18n_engine();
    const std::string translated = engine.translate( msgid, {} );
    if( std::strcmp( engine.get_last_error(), i18n_token_not_found ) == 0 ) {
        return msgid;
    }

    return store_no_localize_i18n_text( translated );
}
#endif // !LOCALIZE

#if defined(LOCALIZE)
#include "uilist.h"

std::string select_language()
{
    auto languages = get_options().get_option( "USE_LANG" ).getItems();

    languages.erase( std::remove_if( languages.begin(),
    languages.end(), []( const options_manager::id_and_option & lang ) {
        return lang.first.empty() || lang.second.empty();
    } ), languages.end() );

    uilist sm;
    sm.allow_cancel = false;
    sm.text = _( "Select your language" );
    for( size_t i = 0; i < languages.size(); i++ ) {
        sm.addentry( i, true, MENU_AUTOASSIGN, languages[i].second.translated() );
    }
    sm.query();

    return languages[sm.ret].first;
}
#endif // LOCALIZE

std::string locale_dir()
{
    std::string loc_dir = PATH_INFO::langdir();
#if defined(LOCALIZE)

#if (defined(__DragonFly__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)) && !defined(CATA_IS_ON_BSD)
#define CATA_IS_ON_BSD
#endif

#if !defined(__ANDROID__) && ((defined(__linux__) || defined(CATA_IS_ON_BSD) || (defined(MACOSX) && !defined(TILES))))
    if( !PATH_INFO::base_path().get_logical_root_path().empty() ) {
        loc_dir = ( PATH_INFO::base_path() / "share" / "locale" ).generic_u8string();
    } else {
        loc_dir = PATH_INFO::langdir();
    }
#endif
#endif // LOCALIZE
    return loc_dir;
}

void set_language_from_options()
{
#if defined(LOCALIZE)
    const std::string system_lang = SystemLocale::Language().value_or( "en" );
    std::string lang_opt = get_option<std::string>( "USE_LANG" ).empty() ? system_lang :
                           get_option<std::string>( "USE_LANG" );
    set_language( lang_opt );
#else
    const std::string system_lang = SystemLocale::Language().value_or( "en" );
    std::string lang_opt = get_option<std::string>( "USE_LANG" ).empty() ? system_lang :
                           get_option<std::string>( "USE_LANG" );
    set_language( lang_opt );
#endif
}

void set_language( const std::string &lang )
{
#if defined(LOCALIZE)
    DebugLog( D_INFO, D_MAIN ) << "Setting language to: '" << lang << '\'';
    TranslationManager::GetInstance().SetLanguage( lang );
#if defined(_WIN32)
    // Use the ANSI code page 1252 to work around some language output bugs. (#8665)
    if( setlocale( LC_ALL, ".1252" ) == nullptr ) {
        DebugLog( D_WARNING, D_MAIN ) << "Error while setlocale(LC_ALL, '.1252').";
    }
#endif

    reset_sanity_check_genders();
#else
    no_localize_i18n_selected_lang() = lang;
    DebugLog( D_INFO, D_MAIN ) << "Setting no-localize language to: '" << lang << '\'';
#endif // LOCALIZE

    // increment version to invalidate translation cache
    do {
        current_language_version++;
    } while( current_language_version == INVALID_LANGUAGE_VERSION );

    // Names depend on the language settings. They are loaded from different files
    // based on the currently used language. If that changes, we have to reload the
    // names.
    SNIPPET.reload_names( PATH_INFO::names() );

    set_title( string_format( _( "Cataclysm: Dark Days Ahead - %s" ), getVersionString() ) );
}
