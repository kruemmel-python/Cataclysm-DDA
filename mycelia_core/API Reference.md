# Mycelia Security SDK – API Reference

**Version:** 4.1 Enterprise  
**Module:** Core Cryptography (`CC_OpenCl.dll`)  
**Plattformen:** Windows x64, Linux x64  
**Sprachen:** C/C++, Python, C# (.NET)

---

## 1. Übersicht & Architektur

Das Mycelia Security SDK bietet Zugriff auf die **VRAM-basierte Chaos-Engine**.
Die API ist als **C-Interface** konzipiert, um maximale Kompatibilität mit Hochsprachen (C#, Python, Java) zu gewährleisten.

### Kernkonzepte

*   **Context (`myc_context_t`):** Repräsentiert eine Instanz der Engine auf einer spezifischen GPU. Ein Kontext hält den VRAM-Speicher für das Chaos-Feld (Grid) reserviert.
*   **Biological Seed:** Ein 64-Bit Integer, der den deterministischen Startzustand der Simulation definiert.
*   **Bio-CTR (Counter Mode):** Die Verschlüsselung unterstützt Random Access und Streaming. Durch Angabe eines `stream_offset` kann jeder beliebige Byte-Block einer Datei unabhängig berechnet werden.
*   **Thread Safety:** Die Core-DLL ist **nicht thread-safe** pro Kontext. Zugriff auf denselben Kontext aus mehreren Threads muss durch Mutex/Locks synchronisiert werden (siehe Python/C# Wrapper).

---

## 2. Native C API (`mycelia.h`)

Diese Funktionen werden direkt aus der `CC_OpenCl.dll` exportiert.

### Datentypen

#### `myc_result` (Enum)
Rückgabewerte für alle API-Operationen.

| Wert | Konstante | Beschreibung |
| :--- | :--- | :--- |
| `0` | `MYC_SUCCESS` | Operation erfolgreich. |
| `-1` | `MYC_ERR_UNKNOWN` | Unbekannter Fehler (Speicher, System). |
| `-2` | `MYC_ERR_NO_GPU` | Keine OpenCL-fähige GPU gefunden. |
| `-3` | `MYC_ERR_INIT_FAILED` | Treiber konnte nicht geladen werden. |
| `-4` | `MYC_ERR_INVALID_PARAM` | Null-Pointer oder ungültige Werte übergeben. |
| `-6` | `MYC_ERR_OPENCL` | Interner Fehler der Grafikkarte. |

#### `myc_context_t` (Handle)
Ein opaker Zeiger auf die interne Kontext-Struktur. Darf vom Client nicht dereferenziert werden.

---

### System Management

#### `myc_init`
Initialisiert das Subsystem, lädt OpenCL-Treiber und enumeriert verfügbare Geräte. Muss vor jeder anderen Funktion aufgerufen werden.

```c
MY_API myc_result myc_init();
```

#### `myc_get_device_count`
Gibt die Anzahl der verfügbaren, kompatiblen GPUs zurück.

```c
MY_API int myc_get_device_count();
```

#### `myc_get_last_error`
Gibt eine menschenlesbare Beschreibung des letzten Fehlers zurück.

```c
MY_API const char* myc_get_last_error();
```

---

### Context Management

#### `myc_create_context`
Erstellt eine neue Kryptografie-Instanz auf der gewählten GPU und reserviert VRAM.

```c
MY_API myc_result myc_create_context(int gpu_index, myc_context_t* out_ctx);
```
*   **gpu_index:** Index der GPU (0 bis `myc_get_device_count()` - 1).
*   **out_ctx:** Zeiger auf das zu erstellende Handle.

#### `myc_destroy_context`
Gibt alle Ressourcen (RAM und VRAM) frei.

```c
MY_API void myc_destroy_context(myc_context_t ctx);
```

---

### Cryptography Operations

#### `myc_set_seed`
Setzt den "biologischen" Seed (Master Key). Dies löst den Reset der Chaossimulation im VRAM aus.

```c
MY_API myc_result myc_set_seed(myc_context_t ctx, uint64_t seed);
```
*   **seed:** 64-Bit Integer. In der Praxis oft ein XOR-Produkt aus `Random` und `Hash(Passwort)`.

#### `myc_process_buffer`
Führt die Verschlüsselung oder Entschlüsselung durch (Symmetrisches XOR-Stream-Verfahren). Die Operation erfolgt **In-Place**.

```c
MY_API myc_result myc_process_buffer(
    myc_context_t ctx, 
    uint8_t* data, 
    size_t len, 
    size_t stream_offset
);
```
*   **data:** Zeiger auf die Rohdaten (werden überschrieben!).
*   **len:** Anzahl der zu verarbeitenden Bytes.
*   **stream_offset:** Die absolute Position dieser Daten im Gesamtstrom (z.B. Dateiposition). Wichtig für die korrekte Synchronisation des Chaos-Stroms bei Blockverarbeitung.

---

## 3. Python Wrapper API

Das Modul `mycelia_chat_engine.py` kapselt die C-API für Hochsprachen-Nutzung.

### Klasse `MyceliaChatEngine`

#### Konstruktor
```python
engine = MyceliaChatEngine(gpu_index=0)
```
Initialisiert die DLL und den GPU-Kontext. Wirft `OSError` bei Fehlschlag.

#### `set_password(password: str)`
Konvertiert ein String-Passwort in einen 64-Bit Hash (`self.password_hash`), der zur Maskierung des Seeds verwendet wird.

#### `encrypt_bytes(data: bytes) -> bytes`
Verschlüsselt einen Byte-Blob (z.B. Text oder Datei).
*   **Rückgabe:** Ein fertig gepacktes Protokoll-Paket:
    `[MaskedSeed (8B)] [Length (4B)] [EncryptedPayload]`
*   **Hinweis:** Generiert für jeden Aufruf einen neuen, zufälligen Master-Seed.

#### `decrypt_packet_to_bytes(packet: bytes) -> bytes`
Nimmt ein Paket entgegen, extrahiert den Header, demaskiert den Seed (mit dem gesetzten Passwort) und entschlüsselt den Payload.
*   **Rückgabe:** Entschlüsselte Rohdaten oder `None` bei Fehler.

---

## 4. C# / .NET API Reference

Für die Integration in Enterprise-Anwendungen via `MyceliaEngine.cs`.

### Klasse `Mycelia.Security.MyceliaEngine`
Implementiert `IDisposable`.

#### `public MyceliaEngine(int gpuIndex = 0)`
Konstruktor. Lädt die unmanaged DLL und reserviert Speicher.
*   *Throws:* `Exception` bei Initialisierungsfehlern.

#### `public void SetSeed(ulong seed)`
Setzt den Master-Seed für nachfolgende Operationen.

#### `public void Process(byte[] data, ulong streamOffset = 0)`
Verschlüsselt das Byte-Array **in-place**.
*   **data:** Der zu bearbeitende Puffer.
*   **streamOffset:** (Optional) Offset für Streaming großer Dateien. Standard ist 0.

#### `public void Dispose()`
Gibt die C-Kontexte und GPU-Handles sauber frei. Muss aufgerufen werden (oder `using`-Block verwenden).

---

## 5. Integration Best Practices

### A. Threading & Concurrency
Die `myc_process_buffer` Funktion verändert den internen Zustand des GPU-Kontexts (Simulationsschritte).
*   **Python:** Nutzen Sie `threading.Lock()` um Aufrufe an die DLL.
*   **C#:** Nutzen Sie `lock(engineInstance) { ... }`.
*   **High Performance:** Für maximale Parallelität erstellen Sie **mehrere Kontexte** (Instanzen), z.B. einen pro verfügbare GPU.

### B. Speichermanagement
Für Dateien > 100 MB wird **Chunking** empfohlen.
1.  Datei in 64KB - 1MB Blöcke lesen.
2.  `myc_process_buffer` mit dem entsprechenden `stream_offset` aufrufen.
3.  Block schreiben.
Dies verhindert RAM-Überlauf und maximiert die Pipeline-Effizienz.

### C. Integrität (Tamper Protection)
Die Core-Engine führt reine XOR-Operationen durch. Um Manipulationen zu erkennen, muss die Anwendungsebene (Python/C# Wrapper) eine Integritätsprüfung implementieren.
*   **Standard:** Mycelia V4 nutzt **Zlib** (CRC32/Adler32) für schnelle Integritätschecks.
*   **High-Security:** Nutzen Sie Encrypt-then-MAC (z.B. BLAKE2b Hash des Ciphertexts anhängen), wie in der Chat-App demonstriert.

---

## 6. Fehlerbehandlung

| Fehlerbild | Mögliche Ursache | Lösung |
| :--- | :--- | :--- |
| `DllNotFoundException` | `CC_OpenCl.dll` fehlt. | DLL in das Ausführungsverzeichnis (`bin/`) kopieren. |
| `BadImageFormatException` | Architektur-Konflikt. | Zielplattform auf **x64** stellen (nicht x86/Any CPU). |
| `Integrity Error` (Zlib) | Falsches Passwort oder Bit-Flip. | Passwort prüfen. Datei ist korrupt. |
| `OpenCL Error -4` | GPU-Speicher voll oder defekt. | Anwendung neu starten, GPU-Auslastung prüfen. |
