# Mycelia Security Architecture – Technical Whitepaper

**Version:** 4.1 Enterprise  
**Klassifizierung:** Technical Documentation  
**Technologie:** GPU-Based Emergent Cryptography

---

## 1. Einführung: Das Ende der statischen Schlüssel

Klassische Kryptografie (AES, RSA) basiert auf mathematischen Problemen, die für Computer schwer zu lösen sind. Sie hat jedoch einen gravierenden systemischen Schwachpunkt: **Der Schlüssel muss existieren.** Er muss generiert, im RAM gehalten, auf Festplatten gespeichert oder über Netzwerke ausgetauscht werden.

**Mycelia Security** eliminiert diesen Schwachpunkt durch **Emergenz**.
Wir speichern keine Schlüssel. Wir speichern nur die "DNA" (einen Seed), aus der ein Schlüssel im Millisekundenbereich im Grafikspeicher (VRAM) wächst und sofort wieder zerfällt.

---

## 2. Die Kern-Technologie: SubQG & Bio-CTR

Das Herzstück von Mycelia ist der **`CC_OpenCl.dll`** Treiber. Er ist keine Standard-Bibliothek, sondern eine hochspezialisierte Physik-Engine.

### 2.1 Wie der Schlüssel entsteht ("Grow-on-Demand")
Anstatt einen zufälligen String als Schlüssel zu verwenden, nutzt Mycelia **deterministisches Chaos**.

1.  **Initialisierung:** Ein 64-Bit Seed wird an die GPU gesendet.
2.  **Simulation:** Im VRAM startet eine Simulation ("SubQG"). Tausende virtuelle Agenten interagieren nach komplexen, nicht-linearen Regeln (Integer-based Chaos).
3.  **Emergenz:** Aus dem Seed entsteht ein hochkomplexes, einzigartiges Datenmuster im VRAM. Dieses Muster ist nicht mathematisch einfach vorhersehbar, aber **zu 100% reproduzierbar**, wenn man denselben Seed und **dieselbe Physik-Engine** (den Treiber) besitzt.
4.  **Keystream:** Dieses Muster wird als Schlüsselstrom (Keystream) ausgelesen.

### 2.2 Bio-CTR (Counter Mode)
Für große Dateien oder Streams wird der Inhalt in 64KB-Blöcke unterteilt. Jeder Block erhält einen eigenen, abgeleiteten Seed. Dadurch entsteht ein endloser, sich nicht wiederholender Schlüsselstrom, der parallel auf hunderten GPU-Kernen berechnet werden kann.

---

## 3. Das Mycelia Chat-Protokoll (MCP)

Die Kommunikation im Mycelia Chat erfolgt nach dem **Dual-Dependency-Prinzip**. Hier ist der exakte Ablauf einer Nachrichtenübertragung von Alice zu Bob:

### Schritt A: Sender (Alice)

1.  **Input:** Alice sendet ein Bild (z.B. `plan.jpg`).
2.  **Kompression:** Die Datei wird mittels **Zlib** komprimiert. Das reduziert die Größe und fügt CRC-Prüfsummen hinzu (Schutz vor Bit-Fehlern).
3.  **Seed-Generierung:** Der Client würfelt einen zufälligen **Master-Seed** für diese Übertragung.
4.  **Maskierung (Layer 1):** Der Master-Seed wird mit dem **Shared Secret (Passwort-Hash)** XOR-verknüpft.
    *   *Ergebnis:* Ein `MaskedSeed`, den nur jemand mit dem Passwort lesen kann.
5.  **Verschlüsselung (Layer 2):**
    *   Der **reine** Master-Seed wird an die lokale GPU gesendet.
    *   Die `CC_OpenCl.dll` lässt das Chaos-Feld wachsen.
    *   Die komprimierten Daten werden mit dem Chaos-Feld XOR-verknüpft.
6.  **Framing:** Ein Paket wird geschnürt: `[Länge][MaskedSeed][VerschlüsselteDaten]`.

### Schritt B: Netzwerk (Der Tunnel)

Über das Netzwerk (TCP) fließt nur weißes Rauschen. Der Server (Relay) leitet die Pakete weiter, kann sie aber nicht interpretieren, da ihm sowohl das Passwort als auch die GPU-Logik fehlt.

### Schritt C: Empfänger (Bob)

1.  **Empfang:** Bob empfängt das Paket.
2.  **Demaskierung:** Bob wendet sein Passwort auf den `MaskedSeed` an.
    *   *Passt das Passwort:* Er erhält den korrekten Master-Seed.
    *   *Falsches Passwort:* Er erhält einen falschen Seed.
3.  **Rekonstruktion:** Bob sendet den Seed an **seine** `CC_OpenCl.dll`.
    *   Besitzt Bob die Software, generiert seine GPU exakt dasselbe Chaos-Feld wie Alice.
4.  **Entschlüsselung:** Das Chaos-Feld wird vom Datenstrom abgezogen (XOR).
5.  **Verifikation:** Die resultierenden Daten werden an Zlib zur Dekompression übergeben.
    *   War das Passwort falsch? -> Zlib meldet "Data Error".
    *   War die Software falsch? -> Zlib meldet "Data Error".
    *   Wurde die Datei manipuliert? -> Zlib meldet "Checksum Error".
    *   Alles korrekt? -> Das Bild `plan.jpg` wird wiederhergestellt.

---

## 4. Sicherheitsanalyse: Warum Mycelia einzigartig ist

Die Sicherheit von Mycelia beruht auf der **Zwei-Faktor-Abhängigkeit (Dual Dependency)**. Ein Angreifer muss zwei Hürden gleichzeitig überwinden.

### Szenario 1: Der Angreifer hat die verschlüsselte Datei & das Passwort
Ein Mitarbeiter verliert einen USB-Stick mit verschlüsselten Daten und hat das Passwort auf einem Zettel notiert.
*   **Klassische Krypto (AES):** Der Angreifer nutzt ein Standard-Tool (OpenSSL), gibt das Passwort ein und hat die Daten.
*   **Mycelia:** Der Angreifer hat das Passwort, aber er besitzt nicht die **proprietäre Engine (`CC_OpenCl.dll`)**.
    *   Er kann den Seed zwar demaskieren.
    *   Aber er hat keine Software, die aus der Zahl `948273...` den korrekten 64KB-Keystream generiert. Die Algorithmen der "biologischen Simulation" sind nicht öffentlich (kein Standard wie AES).
    *   **Ergebnis:** Die Daten bleiben unlesbar.

### Szenario 2: Der Angreifer stiehlt die Software
Ein Hacker bricht in das Firmennetzwerk ein und kopiert die `MyceliaChat.exe` und die DLL.
*   Er hat nun die "Maschine", um Schlüssel zu züchten.
*   Er fängt den Datenverkehr ab. Er sieht den `MaskedSeed`.
*   Da er das **Chat-Passwort** nicht kennt, kann er den `MaskedSeed` nicht in den echten `MasterSeed` umwandeln.
*   Gibt er den maskierten Seed in die Engine ein, erzeugt die GPU ein falsches Chaos-Feld.
*   **Ergebnis:** Die Entschlüsselung liefert nur Müll.

### Szenario 3: Quantencomputer
Quantencomputer bedrohen Algorithmen, die auf Faktorisierung (RSA) oder diskreten Logarithmen (ECC) basieren.
*   Mycelia basiert auf **deterministischem Chaos** und einfachen **Integer-Bit-Operationen** (XOR, Shift, Rotate).
*   Es gibt keine mathematische "Abkürzung", um aus dem Keystream den Seed zu berechnen, ohne die Simulation Schritt für Schritt rückwärts zu rechnen (was bei Chaos extrem schwierig bis unmöglich ist).
*   Da kein Public-Key-Verfahren für den Payload genutzt wird, sind Shors Algorithmus & Co. wirkungslos.

---

## 5. Technische Zusammenfassung für Entwickler

| Komponente | Technologie | Funktion |
| :--- | :--- | :--- |
| **Engine** | C++ / OpenCL | Erzeugt Keystreams im VRAM. Hardware-agnostisch (Integer Math), aber Software-gebunden (kompilierte Logik). |
| **Wrapper** | Python `ctypes` | Bindet die DLL ein, managed Memory und Threading (`C_LOCK`). |
| **Kompression** | zlib (Deflate) | Reduziert Datengröße, dient als Integritäts-Check (HMAC-Ersatz). |
| **Verschlüsselung** | XOR Stream Cipher | `Plaintext XOR (GPU_Chaos(Seed))` |
| **Seed Schutz** | Hash-XOR | `Seed_Netzwerk = Seed_Echt XOR SHA256(Passwort)` |
| **Transport** | TCP Socket | Überträgt gerahmte Pakete (`Len + Payload`). |

## 6. Fazit

Mycelia Security ist mehr als Verschlüsselung. Es ist eine **Zugangskontrolle durch Technologie-Besitz**.

Nur wer:
1.  Teil der Organisation ist (Besitz der Software/Treiber)
2.  **UND** autorisiert ist (Kenntnis des Passworts)

...kann auf die Daten zugreifen. Fehlt eine der beiden Komponenten, ist der Datenstrom mathematisch und physikalisch wertloses Rauschen.
