#pragma once
#ifndef CATA_SRC_I18N_ENGINE_H
#define CATA_SRC_I18N_ENGINE_H

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

class I18nEngine
{
    public:
        I18nEngine();

        const char *get_last_error() const;
        void set_last_error( const std::string &msg );
        void clear_last_error();

        const std::string &get_meta_locale() const;
        const std::string &get_meta_fallback() const;
        const std::string &get_meta_note() const;
        int get_meta_plural_rule() const;

        bool load_txt_catalog( const std::string &txt, bool strict );
        bool load_txt_file( const std::string &path, bool strict );
        bool reload();

        std::string translate( const std::string &token, const std::vector<std::string> &args );
        std::string translate_plural( const std::string &token, int count,
                                      const std::vector<std::string> &args );

        std::string dump_table() const;
        std::string find_any( const std::string &query ) const;
        std::string check_catalog_report( int &code ) const;
        bool export_binary_catalog( const std::string &path ) const;

    private:
        struct Entry {
            std::string label;
            std::string text;
        };

        std::unordered_map<std::string, std::vector<Entry>> table;
        std::string last_error;
        std::string meta_locale;
        std::string meta_fallback;
        std::string meta_note;
        int meta_plural_rule = 0;
        std::optional<std::string> last_loaded_path;
        std::optional<std::string> last_loaded_txt;

        static bool is_ws( unsigned char c ) noexcept;
        static bool is_digit( unsigned char c ) noexcept;
        static bool is_xdigit( unsigned char c ) noexcept;
        static bool is_digit_uc( char c ) noexcept;
        static bool is_xdigit_uc( char c ) noexcept;

        static void trim_inplace( std::string &s );
        static bool is_hex_token( const std::string &s );
        static void strip_utf8_bom( std::string &s );
        static std::string to_lower_ascii( std::string s );
        static std::string unescape_txt_min( const std::string &s );
        static bool parse_line( const std::string &line_in,
                                std::string &out_token,
                                std::string &out_label,
                                std::string &out_text,
                                std::string &out_err );

        const Entry *find_entry( const std::string &token, const std::string &label ) const;
        static std::string apply_args( const std::string &text, const std::vector<std::string> &args );
};

void set_engine_error( I18nEngine *eng, const std::string &msg );
void clear_engine_error( I18nEngine *eng );

#endif
