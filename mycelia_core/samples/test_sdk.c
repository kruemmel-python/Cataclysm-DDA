#include <stdio.h>
#include <string.h>
// Da wir im "samples" Ordner sind oder von root kompilieren, 
// binden wir den Header so ein, wie es ein Kunde tun würde:
#include "mycelia.h" 

int main() {
    printf("[SDK Test] Starte Mycelia Core...\n");

    // 1. Init
    if (myc_init() != MYC_SUCCESS) {
        printf("FEHLER: Init fehlgeschlagen. DLL nicht gefunden oder inkompatibel.\n");
        return 1;
    }

    int count = myc_get_device_count();
    printf("[SDK Test] Gefundene GPUs: %d\n", count);

    // 2. Context
    myc_context_t ctx;
    if (myc_create_context(0, &ctx) != MYC_SUCCESS) {
        printf("FEHLER: Context konnte nicht erstellt werden.\n");
        return 1;
    }

    // 3. Seed
    myc_set_seed(ctx, 999999);

    // 4. Test Crypto
    char data[] = "Hello World from Enterprise SDK!";
    printf("[SDK Test] Original: %s\n", data);

    // Encrypt
    myc_process_buffer(ctx, (unsigned char*)data, strlen(data), 0);
    printf("[SDK Test] Verschluesselt (Hex): ");
    for(int i=0; i<strlen(data); i++) printf("%02X", (unsigned char)data[i]);
    printf("\n");

    // Decrypt (Reset Seed für Stream-Synchronität)
    myc_set_seed(ctx, 999999);
    myc_process_buffer(ctx, (unsigned char*)data, strlen(data), 0);
    printf("[SDK Test] Entschluesselt: %s\n", data);

    myc_destroy_context(ctx);
    printf("[SDK Test] Test erfolgreich beendet.\n");
    return 0;
}