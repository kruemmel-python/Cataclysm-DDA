#pragma once
#ifndef CATA_SRC_TRANSLATION_MANAGER_IMPL_H
#define CATA_SRC_TRANSLATION_MANAGER_IMPL_H

#if defined(LOCALIZE)

#include <array>
#include <optional>
#include <unordered_map>

#include "i18n_engine.h"
#include "translation_document.h"
#include "translation_manager.h"

class TranslationManager::Impl
{
    private:
        std::vector<TranslationDocument> documents;

        std::unordered_map<std::uint32_t, std::vector<std::pair<std::size_t, std::size_t>>> strings;
        static std::uint32_t Hash( const char *str );
        std::optional<std::pair<std::size_t, std::size_t>> LookupString( const char *query ) const;

        std::unordered_map<std::string, std::vector<std::string>> mo_files;
        static std::string LanguageCodeOfPath( std::string_view path );
        void ScanTranslationDocuments();
        std::string ConstructContextualQuery( const char *context, const char *message ) const;
        void InvalidateI18nOverrides();
        void TryLoadI18nOverrides() const;
        const char *TryTranslateI18nToken( const std::string &token ) const;
        const char *TryTranslateI18nPluralToken( const std::string &token, std::size_t n ) const;
        const char *StoreI18nOverrideResult( std::string text ) const;
        void Reset();
        std::string current_language_code;
        mutable I18nEngine i18n_override_engine;
        mutable bool i18n_override_load_attempted = false;
        mutable bool i18n_override_loaded = false;
        mutable std::array<std::string, 32> i18n_override_buffer {};
        mutable std::size_t i18n_override_buffer_index = 0;
    public:
        Impl();
        std::unordered_set<std::string> GetAvailableLanguages();
        void SetLanguage( const std::string &language_code );
        std::string GetCurrentLanguage() const;
        void LoadDocuments( const std::vector<std::string> &files );

        const char *Translate( const std::string &message ) const;
        const char *Translate( const char *message ) const;
        const char *TranslatePlural( const char *singular, const char *plural, std::size_t n ) const;
        const char *TranslateWithContext( const char *context, const char *message ) const;
        const char *TranslatePluralWithContext( const char *context, const char *singular,
                                                const char *plural, std::size_t n ) const;
};

#endif // defined(LOCALIZE)

#endif // CATA_SRC_TRANSLATION_MANAGER_IMPL_H
