#pragma once
#ifndef CATA_SRC_I18N_API_H
#define CATA_SRC_I18N_API_H

#ifdef _WIN32
#define I18N_API
#else
#define I18N_API
#endif

extern "C" {

I18N_API void *i18n_new();
I18N_API void i18n_free( void *ptr );
I18N_API const char *i18n_last_error( void *ptr );
I18N_API int i18n_last_error_copy( void *ptr, char *out_buf, int buf_size );

I18N_API int i18n_get_meta_locale_copy( void *ptr, char *out_buf, int buf_size );
I18N_API int i18n_get_meta_fallback_copy( void *ptr, char *out_buf, int buf_size );
I18N_API int i18n_get_meta_note_copy( void *ptr, char *out_buf, int buf_size );
I18N_API int i18n_get_meta_plural_rule( void *ptr );

I18N_API int i18n_load_txt( void *ptr, const char *txt_str, int strict );
I18N_API int i18n_load_txt_file( void *ptr, const char *path, int strict );
I18N_API int i18n_reload( void *ptr );

I18N_API unsigned int i18n_abi_version( void );
I18N_API unsigned int i18n_binary_version_supported_max( void );

I18N_API int i18n_translate( void *ptr,
                             const char *token,
                             const char **args,
                             int args_len,
                             char *out_buf,
                             int buf_size );
I18N_API int i18n_translate_plural( void *ptr,
                                    const char *token,
                                    int count,
                                    const char **args,
                                    int args_len,
                                    char *out_buf,
                                    int buf_size );
I18N_API int i18n_print( void *ptr, char *out_buf, int buf_size );
I18N_API int i18n_find( void *ptr, const char *query, char *out_buf, int buf_size );
I18N_API int i18n_check( void *ptr, char *report_buf, int report_size );
I18N_API int i18n_export_binary( void *ptr, const char *path );

}

#endif
