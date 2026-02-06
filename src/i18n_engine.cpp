#include "i18n_engine.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>

namespace {
constexpr size_t RESULT_TOO_LARGE_LIMIT = 16ull * 1024ull * 1024ull;
}

I18nEngine::I18nEngine() = default;

const char *I18nEngine::get_last_error() const
{
    return last_error.c_str();
}

void I18nEngine::set_last_error( const std::string &msg )
{
    last_error = msg;
}

void I18nEngine::clear_last_error()
{
    last_error.clear();
}

const std::string &I18nEngine::get_meta_locale() const
{
    return meta_locale;
}

const std::string &I18nEngine::get_meta_fallback() const
{
    return meta_fallback;
}

const std::string &I18nEngine::get_meta_note() const
{
    return meta_note;
}

int I18nEngine::get_meta_plural_rule() const
{
    return meta_plural_rule;
}

bool I18nEngine::is_ws( unsigned char c ) noexcept
{
    return std::isspace( c ) != 0;
}

bool I18nEngine::is_digit( unsigned char c ) noexcept
{
    return std::isdigit( c ) != 0;
}

bool I18nEngine::is_xdigit( unsigned char c ) noexcept
{
    return std::isxdigit( c ) != 0;
}

bool I18nEngine::is_digit_uc( char c ) noexcept
{
    return std::isdigit( static_cast<unsigned char>( c ) ) != 0;
}

bool I18nEngine::is_xdigit_uc( char c ) noexcept
{
    return std::isxdigit( static_cast<unsigned char>( c ) ) != 0;
}

void I18nEngine::trim_inplace( std::string &s )
{
    auto isspace_uc = []( unsigned char c ) { return std::isspace( c ) != 0; };

    auto it1 = std::find_if_not( s.begin(), s.end(),
                                 [&]( char ch ) { return isspace_uc( static_cast<unsigned char>( ch ) ); } );
    auto it2 = std::find_if_not( s.rbegin(), s.rend(),
                                 [&]( char ch ) { return isspace_uc( static_cast<unsigned char>( ch ) ); } ).base();

    if( it1 >= it2 ) {
        s.clear();
        return;
    }

    s.assign( it1, it2 );
}

bool I18nEngine::is_hex_token( const std::string &s )
{
    if( s.size() < 6 || s.size() > 32 ) {
        return false;
    }
    for( unsigned char c : s ) {
        if( !is_xdigit( c ) ) {
            return false;
        }
    }
    return true;
}

void I18nEngine::strip_utf8_bom( std::string &s )
{
    if( s.size() >= 3 &&
        static_cast<unsigned char>( s[0] ) == 0xEF &&
        static_cast<unsigned char>( s[1] ) == 0xBB &&
        static_cast<unsigned char>( s[2] ) == 0xBF ) {
        s.erase( 0, 3 );
    }
}

std::string I18nEngine::to_lower_ascii( std::string s )
{
    for( char &c : s ) {
        c = static_cast<char>( std::tolower( static_cast<unsigned char>( c ) ) );
    }
    return s;
}

std::string I18nEngine::unescape_txt_min( const std::string &s )
{
    std::string out;
    out.reserve( s.size() );
    for( size_t i = 0; i < s.size(); ++i ) {
        if( s[i] == '\\' && i + 1 < s.size() ) {
            char c = s[i + 1];
            switch( c ) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case '\\': out += '\\'; break;
                case ':': out += ':'; break;
                default: out += c; break;
            }
            ++i;
        } else {
            out += s[i];
        }
    }
    return out;
}

bool I18nEngine::parse_line( const std::string &line_in,
                             std::string &out_token,
                             std::string &out_label,
                             std::string &out_text,
                             std::string &out_err )
{
    out_err.clear();
    out_token.clear();
    out_label.clear();
    out_text.clear();

    std::string line = line_in;
    trim_inplace( line );
    if( line.empty() ) {
        return false;
    }
    if( line[0] == '#' ) {
        return false;
    }

    const auto colon = line.find( ':' );
    if( colon == std::string::npos ) {
        out_err = "Kein ':' gefunden.";
        return false;
    }

    std::string head = line.substr( 0, colon );
    std::string text = line.substr( colon + 1 );
    trim_inplace( head );
    while( !text.empty() && is_ws( static_cast<unsigned char>( text.front() ) ) ) {
        text.erase( text.begin() );
    }

    std::string token;
    std::string label;

    const auto paren_open = head.find( '(' );
    if( paren_open == std::string::npos ) {
        token = head;
    } else {
        token = head.substr( 0, paren_open );
        trim_inplace( token );

        const auto paren_close = head.find( ')', paren_open + 1 );
        if( paren_close == std::string::npos ) {
            out_err = "Label '(' ohne schlie?ende ')'.";
            return false;
        }

        label = head.substr( paren_open + 1, paren_close - paren_open - 1 );
        trim_inplace( label );

        std::string tail = head.substr( paren_close + 1 );
        trim_inplace( tail );
        if( !tail.empty() ) {
            out_err = "Unerwarteter Text nach Label.";
            return false;
        }
    }

    trim_inplace( token );
    if( token.empty() ) {
        out_err = "Leerer Token.";
        return false;
    }

    out_token = token;
    out_label = label;
    out_text = unescape_txt_min( text );
    return true;
}

const I18nEngine::Entry *I18nEngine::find_entry( const std::string &token,
                                                 const std::string &label ) const
{
    auto it = table.find( token );
    if( it == table.end() ) {
        return nullptr;
    }
    const std::string label_lc = to_lower_ascii( label );
    for( const Entry &entry : it->second ) {
        if( entry.label == label_lc ) {
            return &entry;
        }
    }
    return nullptr;
}

std::string I18nEngine::apply_args( const std::string &text, const std::vector<std::string> &args )
{
    std::string out;
    out.reserve( text.size() );
    for( size_t i = 0; i < text.size(); ++i ) {
        if( text[i] == '{' ) {
            const size_t close = text.find( '}', i + 1 );
            if( close == std::string::npos ) {
                out += text[i];
                continue;
            }
            const std::string key = text.substr( i + 1, close - i - 1 );
            bool replaced = false;
            if( !key.empty() && std::all_of( key.begin(), key.end(), is_digit_uc ) ) {
                const size_t idx = static_cast<size_t>( std::stoi( key ) );
                if( idx < args.size() ) {
                    out += args[idx];
                    replaced = true;
                }
            }
            if( !replaced ) {
                out.append( text, i, close - i + 1 );
            }
            i = close;
        } else {
            out += text[i];
        }
    }
    return out;
}

bool I18nEngine::load_txt_catalog( const std::string &txt, bool strict )
{
    clear_last_error();
    table.clear();
    meta_locale.clear();
    meta_fallback.clear();
    meta_note.clear();
    meta_plural_rule = 0;

    std::string buffer = txt;
    strip_utf8_bom( buffer );

    std::istringstream in( buffer );
    std::string line;
    size_t line_num = 0;
    while( std::getline( in, line ) ) {
        line_num++;
        std::string token;
        std::string label;
        std::string text;
        std::string err;
        if( !parse_line( line, token, label, text, err ) ) {
            if( strict && !err.empty() ) {
                set_last_error( "Parse error at line " + std::to_string( line_num ) + ": " + err );
                return false;
            }
            continue;
        }

        if( token.rfind( "@meta.", 0 ) == 0 ) {
            if( token == "@meta.locale" ) {
                meta_locale = text;
            } else if( token == "@meta.fallback" ) {
                meta_fallback = text;
            } else if( token == "@meta.note" ) {
                meta_note = text;
            } else if( token == "@meta.plural_rule" ) {
                try {
                    meta_plural_rule = std::stoi( text );
                } catch( const std::exception & ) {
                    meta_plural_rule = 0;
                }
            }
            continue;
        }

        const std::string label_lc = to_lower_ascii( label );
        auto &bucket = table[token];
        auto it = std::find_if( bucket.begin(), bucket.end(),
        [&]( const Entry &e ) {
            return e.label == label_lc;
        } );
        if( it != bucket.end() ) {
            if( strict ) {
                set_last_error( "Duplicate token at line " + std::to_string( line_num ) );
                return false;
            }
            it->text = text;
        } else {
            bucket.push_back( Entry{ label_lc, text } );
        }
    }

    last_loaded_txt = txt;
    last_loaded_path.reset();
    return true;
}

bool I18nEngine::load_txt_file( const std::string &path, bool strict )
{
    clear_last_error();
    std::ifstream file( path );
    if( !file.is_open() ) {
        set_last_error( "File not found" );
        return false;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    if( buffer.tellp() > static_cast<std::streampos>( RESULT_TOO_LARGE_LIMIT ) ) {
        set_last_error( "RESULT_TOO_LARGE" );
        return false;
    }
    const bool ok = load_txt_catalog( buffer.str(), strict );
    if( ok ) {
        last_loaded_path = path;
        last_loaded_txt.reset();
    }
    return ok;
}

bool I18nEngine::reload()
{
    if( last_loaded_txt ) {
        return load_txt_catalog( *last_loaded_txt, false );
    }
    if( last_loaded_path ) {
        return load_txt_file( *last_loaded_path, false );
    }
    set_last_error( "No catalog to reload" );
    return false;
}

std::string I18nEngine::translate( const std::string &token, const std::vector<std::string> &args )
{
    clear_last_error();
    const Entry *entry = find_entry( token, "" );
    if( !entry ) {
        set_last_error( "TOKEN_NOT_FOUND" );
        return token;
    }
    return apply_args( entry->text, args );
}

std::string I18nEngine::translate_plural( const std::string &token, int count,
                                          const std::vector<std::string> &args )
{
    clear_last_error();
    const bool singular = count == 1;
    const Entry *entry = nullptr;
    if( singular ) {
        entry = find_entry( token, "one" );
        if( !entry ) {
            entry = find_entry( token, "singular" );
        }
    } else {
        entry = find_entry( token, "other" );
        if( !entry ) {
            entry = find_entry( token, "plural" );
        }
    }
    if( !entry ) {
        entry = find_entry( token, "" );
    }
    if( !entry ) {
        set_last_error( "TOKEN_NOT_FOUND" );
        return token;
    }
    std::string res = apply_args( entry->text, args );
    const std::string count_str = std::to_string( count );
    size_t pos = 0;
    while( ( pos = res.find( "{count}", pos ) ) != std::string::npos ) {
        res.replace( pos, 7, count_str );
        pos += count_str.size();
    }
    return res;
}

std::string I18nEngine::dump_table() const
{
    std::ostringstream out;
    for( const auto &pair : table ) {
        for( const Entry &entry : pair.second ) {
            if( entry.label.empty() ) {
                out << pair.first << ": " << entry.text << "\n";
            } else {
                out << pair.first << "(" << entry.label << "): " << entry.text << "\n";
            }
        }
    }
    return out.str();
}

std::string I18nEngine::find_any( const std::string &query ) const
{
    std::ostringstream out;
    const std::string q = to_lower_ascii( query );
    for( const auto &pair : table ) {
        const std::string token_lc = to_lower_ascii( pair.first );
        for( const Entry &entry : pair.second ) {
            const std::string label_lc = entry.label;
            const std::string text_lc = to_lower_ascii( entry.text );
            if( token_lc.find( q ) != std::string::npos ||
                label_lc.find( q ) != std::string::npos ||
                text_lc.find( q ) != std::string::npos ) {
                if( entry.label.empty() ) {
                    out << pair.first << ": " << entry.text << "\n";
                } else {
                    out << pair.first << "(" << entry.label << "): " << entry.text << "\n";
                }
            }
        }
    }
    return out.str();
}

std::string I18nEngine::check_catalog_report( int &code ) const
{
    std::ostringstream out;
    if( table.empty() ) {
        code = 1;
        out << "EMPTY_CATALOG";
        return out.str();
    }
    code = 0;
    out << "OK";
    return out.str();
}

bool I18nEngine::export_binary_catalog( const std::string & ) const
{
    return false;
}

void set_engine_error( I18nEngine *eng, const std::string &msg )
{
    if( eng ) {
        eng->set_last_error( msg );
    }
}

void clear_engine_error( I18nEngine *eng )
{
    if( eng ) {
        eng->clear_last_error();
    }
}
