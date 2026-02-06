

# **Mycelia Security SDK – Visual Documentation (V4 Enterprise)**

Die folgende Dokumentation beschreibt die zentralen Oberflächen, Funktionsabläufe und Systemkomponenten der Mycelia V4 Enterprise Suite anhand visueller Beispiele.
Sie umfasst:

* **Mycelia Vault** – GPU-basierte Datei-Verschlüsselung
* **Mycelia Encrypted Chat** – End-to-End VRAM-Kryptografie
* **MCP Relay Server** – verschlüsselter Transportlayer
* **Integritätsmechanismen und Seed-Handling**

Alle gezeigten Komponenten verwenden den proprietären Treiber **CC_OpenCl.dll**, der deterministische Chaos-Simulationen im VRAM ausführt.

---

## **1. Mycelia Vault – Benutzeroberfläche**

### **Startansicht**

Die Startansicht zeigt:

<img width="877" height="727" alt="MYCELIA_VAULT" src="https://github.com/user-attachments/assets/b7bef53c-c298-475d-87af-59085cdb7149" />

* GPU-Engine-Status (*ENGINE AKTIV*)
* Eingabefelder für Quell- und Zieldateien
* Buttons für **Verschlüsseln (Stream)** und **Entschlüsseln (Auto)**
* Systemkonsole mit Engine-Logs

```text
> Systemstart...
> [System] OpenCL Engine (1 GPU) bereit.
```

---

## **2. Verschlüsselung (Streaming, GPU-beschleunigt)**

Beispiel: Verschlüsselung einer PDF-Datei („Die Kunst des Prompting“).

<img width="877" height="727" alt="MYCELIA_VAULT2" src="https://github.com/user-attachments/assets/93b95d2d-ac0d-41f8-b79d-7ac9249fcf99" />

Der Vault generiert:

* einen neuen **64-Bit Master-Seed**
* ein **Integrity Tag** auf Basis von Zlib
* GPU-basierte Streaming-Verschlüsselung

Beispiel-Log:

```text
[Task] Starte Verschlüsselung -> promting_book
[Vault] Neuer Master-Seed: 670487782618104300
[Vault] Integrity Tag generiert: 44c4bccaa77e31ee...
[Success] Vorgang beendet in 0.66s
```

---

## **3. Automatische Entschlüsselung & Integritätsprüfung**

Nach der Verschlüsselung kann optional sofort entschlüsselt werden.

<img width="877" height="727" alt="MYCELIA_VAULT3" src="https://github.com/user-attachments/assets/e6935ee7-4527-4b15-a2e3-9c5257eb1e76" />

Die Vault-Engine erkennt:

* Originaldateiname
* Seed
* Integrität der Datei

Log-Beispiel:

```text
[Task] Starte Entschlüsselung...
[Vault] Erkannt: 'Die Kunst des Prompting.pdf'
[Vault] Prüfe Integrität...
[✔] INTEGRITÄT BESTÄTIGT. Datei ist authentisch.
[Success] Vorgang beendet in 0.09s
```

Dieses Resultat beweist:

* Seed wurde korrekt demaskiert
* Keystream wurde deterministisch reproduziert
* Datei wurde **nicht manipuliert**

---

## **4. Mycelia Encrypted Chat (MCP-Protokoll)**

### **Passwort-Eingabe (Shared Secret)**

<img width="817" height="731" alt="chat1" src="https://github.com/user-attachments/assets/4df133d4-1ce3-4aaa-9f0f-70c6a181dfd5" />

Vor Verbindungsaufbau wird das Shared Secret abgefragt.
Es bildet die Basis des **Masking Layers**:

* Passwort → 64-Bit Shared Hash
* Shared Hash ⊕ Master Seed → MaskedSeed
* MaskedSeed wird über Netzwerk übertragen

---

### **Gesicherter Chat-Kanal**

<img width="1632" height="732" alt="chat2" src="https://github.com/user-attachments/assets/4705d3d7-f817-4342-a0d0-8c5f73e6ad4c" />

Nach erfolgreichem Austausch zeigt der Chat:

```text
[System] Kanal gesichert (Hash aktiv).
[System] Verbunden mit 127.0.0.1:5555
```

Nachrichten erscheinen farblich strukturiert:

* **Me:** Blau / Grün
* **Peer:** Orange
* Systemmeldungen in Grau

Beispiel:

```text
Me: Hallo
Peer: Hi
```

---

### **GPU-verschlüsselte Dateiübertragung**

Der Chat unterstützt verschlüsselte Dateiübertragung inklusive Zlib-Kompression.

Beispiel:

```text
Me: Sende '1.pdf' (353.9KB -> 227.5KB | -35.7%)...
Peer: Lieben Dank, Datei erhalten
```

Der Empfänger speichert automatisch:

```text
-> Gespeichert: Mycelia_Downloads\1.pdf (353.9 KB)
```

---

## **5. Relay Server – Transportlayer**

Der MCP-Server ist ein reiner Weiterleiter („Blind Relay“).
Er sieht nur:

<img width="582" height="164" alt="server" src="https://github.com/user-attachments/assets/3597daf4-544d-416b-96ab-b80a4c28e519" />

* MaskedSeed
* Payload-Längen
* Zlib-komprimierte XOR-Pakete
* **niemals Klartext oder Schlüssel**

Beispiel:

```text
[Server] Listening on 0.0.0.0:5555
[Server] Connected with ('127.0.0.1', 49156)
[Server] Connected with ('127.0.0.1', 49163)
```

Damit erfüllt der Server das **Zero-Knowledge-Prinzip**.

---

## **6. Zusammenfassung**

Die Bilder und Logs zeigen eindeutig:

### **1. GPU-Engine ist stabil und deterministisch**

OpenCL-Kontext → VRAM-Chaos-Feld → Keystream-Generierung

### **2. Mycelia Vault ist produktionsreif**

Streaming Encryption, Auto-Decryption, CRC-basierte Integrität

### **3. Mycelia Chat ist vollständig einsatzfähig**

End-to-End, Masked Seed, Zlib-Kompression, Dateiübertragung

### **4. Server ist vollständig blind**

kein Klartext, keine Schlüssel, kein Zugriff auf Payloads

### **5. Das SDK ist technisch marktreif**

Alle Kernkomponenten der Enterprise-Kryptografie laufen real.

---

## **7. Dateistruktur-Hinweis**

```
Mycelia-Security-SDK/
│
├── tools/
│   ├── MyceliaChat_Client.exe           # Chat-Client (DLL embedded)
│   ├── MyceliaChat_Server.exe           # Relay Server (pure Python → exe)
│   ├── MyceliaVault_Enterprise.exe      # Vault GUI (DLL embedded)
│   └── build/
│       └── kernel_cache/                # Automatisch erstellter GPU-Kernel-Cache
│           ├── gfx90c_*_kernel1.bin
│           ├── gfx90c_*_kernel2.bin
│           ├── gfx90c_*_...             # ~100 JIT-optimierte Kernels
│           └── ...
│
└── Doku/
    └── MYCELIA VAULT/
        ├── MYCELIA_VAULT.png
        ├── MYCELIA_VAULT2.png
        ├── MYCELIA_VAULT3.png
        ├── chat1.png
        ├── chat2.png
        ├── server.png
        └── promting_book
```

---


