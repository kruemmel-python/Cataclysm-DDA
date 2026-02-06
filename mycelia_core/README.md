# Mycelia Security SDK

**Hardware-Bound VRAM Cryptography & Emergent Security**

> **"Security ends where a key exists. So we built a system that doesn't need one."**

Mycelia Security is an experimental cryptography engine that generates keystreams directly within GPU VRAM using deterministic chaos and emergent agent simulations. Unlike traditional crypto, it does not store keys in RAM or on disk—they are grown on demand and decay instantly after use.

---

## 🔥 Features

*   **Keyless Cryptography:** Keys are grown physically in VRAM based on a seed and specific GPU instructions.
*   **Hardware Agnostic Core:** Uses highly optimized Integer-Math kernels to ensure compatibility across NVIDIA, AMD, and Intel GPUs without data corruption.
*   **Streaming Encryption (Bio-CTR):** Handles files of any size (TB+) with minimal RAM usage using a block-based deterministic chaos generator.
*   **Secure Messenger:** Includes a fully encrypted, GPU-accelerated chat application with file transfer support.
*   **Integrity & Compression:** Encrypt-then-MAC architecture using BLAKE2b hashing and integrated zlib compression for maximum efficiency and tamper evidence.
*   **Enterprise GUI:** Professional Dark-mode GUI included for easy usage.

---

## 📂 Project Structure

*   `src/`: The High-Performance C++ Core (OpenCL Driver & Integer Chaos Engine).
*   `include/`: C-Header for SDK integration (`mycelia.h`).
*   `python/`: Python Wrappers, GUI, and Chat applications.
    *   `mycelia_vault_v4.py`: The core Python logic wrapper.
    *   `mycelia_gui_v4.py`: The File Vault GUI.
    *   `mycelia_chat.py`: The Secure Chat Client.
    *   `chat_server.py`: The TCP Relay Server.
*   `bin/`: Compiled Binaries (`CC_OpenCl.dll`, `test_sdk.exe`).
*   `lib/`: Import Libraries (`libCC_OpenCl.a`) for C/C++ linking.
*   `samples/`: Example C code for SDK usage.

---

## 🚀 Quick Start (Vault GUI)

**Prerequisites:** Windows (x64) with OpenCL-capable GPU drivers installed.

1.  **Run Pre-compiled EXE:**
    If you have built the executable, simply run `dist/MyceliaVault_Enterprise.exe`.

2.  **Run from Source:**
    Navigate to the `python` folder and ensure `CC_OpenCl.dll` is present in `../bin/` or the current folder.
    ```bash
    cd python
    python mycelia_gui_v4.py
    ```

---

## 💬 Secure Messenger (MyceliaChat)

Mycelia includes a GPU-accelerated secure messenger that supports text and file transfer.

### Architecture
*   **End-to-End Encryption:** Messages are encrypted in VRAM. The server only sees encrypted noise.
*   **Double Layer Security:** Traffic is secured by the Mycelia Engine **PLUS** a Shared Secret (Password) that masks the biological seeds.
*   **Auto-Compression:** Files are automatically compressed (zlib) before encryption to speed up transfer.

### Usage

**1. Start the Server (Relay)**
```bash
python python/chat_server.py
```

**2. Start Clients**
```bash
python python/mycelia_chat.py
```
*   Enter the Shared Secret (Password).
*   Type text or click 📎 to send files (Images, PDFs, etc.).
*   *Note:* Ensure `SERVER_IP` in `mycelia_chat.py` is set to the server's address.

---

## 🛠️ Building the Core & EXEs

**Requirements:** MinGW-w64 (GCC), OpenCL SDK, Python, PyInstaller.

### 1. Build the C++ DLL (The Engine)
Run `build_scripts/build_win.bat` or execute manually:
```bash
g++ -std=c++17 -O3 -march=native -ffast-math -funroll-loops -fstrict-aliasing -DNDEBUG -DCL_TARGET_OPENCL_VERSION=120 -DCL_FAST_OPTS -DMYCELIA_EXPORTS -shared ./src/mycelia_core.c -o ./bin/CC_OpenCl.dll -I"./include" -I"./src" -I"./CL" -L"./CL" -lOpenCL "-Wl,--out-implib,./lib/libCC_OpenCl.a" -static-libstdc++ -static-libgcc
```

### 2. Build Standalone EXEs
To distribute the Vault and Chat without requiring Python:

**Vault EXE:**
```bash
cd python
pyinstaller --noconsole --onefile --name "MyceliaVault_Enterprise" --add-data "../bin/CC_OpenCl.dll;." --hidden-import "mycelia_vault_v4" mycelia_gui_v4.py
```

**Chat Client EXE:**
```bash
cd python
pyinstaller --noconsole --onefile --name "MyceliaChat_Client" --add-data "../bin/CC_OpenCl.dll;." --hidden-import "mycelia_chat_engine" mycelia_chat.py
```

---

## 💻 Integration Guide (Python & C#)

### Python Integration
```python
from mycelia_vault_v4 import MyceliaVaultV4

# Initialize Engine
vault = MyceliaVaultV4()

# Encrypt File (Streamed)
vault.encrypt("secret.pdf", "secret.box")

# Decrypt File
vault.decrypt("secret.box", ".")
```

### C-SDK Usage
You can link against `mycelia.h` and `libCC_OpenCl.a` to integrate Mycelia into your C/C++ applications.

```c
#include "mycelia.h"

// Initialize
myc_init();

// Create Context on GPU 0
myc_context_t ctx;
myc_create_context(0, &ctx);

// Set Biological Seed
myc_set_seed(ctx, 123456789);

// Encrypt Buffer (In-Place)
myc_process_buffer(ctx, my_data, data_len, 0);
```

---

## ⚠️ Troubleshooting

*   **`DLL load failed`:** Ensure `CC_OpenCl.dll` is in the same folder as the script/EXE or in `../bin`.
*   **`Integrity Error`:** The password used for decryption does not match the encryption password, or the file was corrupted during transfer.
*   **`GPU Init Failed`:** Update your graphics drivers. The engine requires OpenCL 1.2 or higher.

## 📜 License

MIT License - Copyright (c) 2025 Mycelia Security
```
