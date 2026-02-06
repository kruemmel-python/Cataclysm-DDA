# mc_mycelia

Ein Paper/Spigot-Plugin, das neue Welten über den Mycelia-Treiber erzeugt. Der Fokus liegt darauf, Mycelia als Seed-Quelle zu nutzen (z.B. über SubQG + Mycel-Pipeline), sodass ein In-Game-Befehl reproduzierbare, deterministische Welten anlegt.

## Funktionsumfang
- `/myceliaworld <weltname> [--seed <zahl>]`: erzeugt eine neue Welt mit einem Mycelia-seeded Generator.
- Ein austauschbarer Seed-Provider, der den Mycelia-Treiber über einen konfigurierbaren Prozess-Aufruf nutzt.
- Fallback auf kryptografisch sichere Seeds, falls der Treiber nicht verfügbar ist, damit der Befehl immer funktioniert.

## Projektaufbau
```
mc_mycelia/
├── build.gradle           # Build-Konfiguration (Java 17, Paper/Spigot-API)
├── settings.gradle        # Setzt den Projektnamen
├── src/main/java
│   └── com/mycelia/mc
│       ├── MyceliaWorldPlugin.java       # Plugin-Entry-Point
│       ├── command/MyceliaWorldCommand.java
│       ├── driver/MyceliaDriver.java     # Seed-Beschaffung über Mycelia-Treiber
│       └── generation/MyceliaChunkGenerator.java
└── src/main/resources
    ├── config.yml         # Treiber-Aufruf + Default-Welteinstellungen
    └── plugin.yml         # Bukkit-Metadaten
```

## Mycelia-Integration
Der `MyceliaDriver` liest den Seed optional aus einem externen Befehl (konfigurierbar in `config.yml`). Damit kannst du deinen bestehenden SubQG/Mycel-Pipeline-Aufruf hinterlegen – z.B. ein Python-Skript, das den Mycelia-Treiber nutzt und einen Seed über STDOUT schreibt.

Beispiel-Konfiguration:
```yaml
driver:
  command: "python ../python/mein_subqg_seed_script.py"
  timeoutSeconds: 5
```
Der Prozess muss eine Integer-Zahl (long) in die erste Zeile von STDOUT schreiben. Bei Fehlern oder Timeouts fällt das Plugin automatisch auf einen kryptografisch sicheren Seed zurück und loggt den Grund.

## Bauen
Das Projekt ist als eigenständiges Gradle-Modul ausgelegt (Java 17):
```bash
cd mc_mycelia
./gradlew build
```
Das erzeugte JAR liegt anschließend unter `build/libs/mc_mycelia-<version>.jar` und kann direkt ins `plugins/`-Verzeichnis deines Paper/Spigot-Servers gelegt werden.

> Hinweis: Das Repository enthält keinen Gradle-Wrapper, damit der Host nicht unnötig wächst. Falls benötigt, kannst du ihn lokal mit `gradle wrapper` erzeugen.

## Nutzung In-Game
1. `config.yml` anpassen, sodass `driver.command` deinen SubQG/Mycel-Treiber startet.
2. Server starten/reloaden.
3. `/myceliaworld testwelt` ausführen – es wird eine neue Welt mit dem Mycelia-Seed erzeugt (oder mit `/myceliaworld testwelt --seed 12345` explizit vorgibt).

## Weiterer Ausbau
- Native JNI/JNA-Anbindung an `libCC_OpenCl` implementieren, um Seeds direkt aus dem Treiber ohne Subprozess zu beziehen.
- Einen dedizierten SubQG-Adapter schreiben, der statt eines generischen Kommandos die Datenströme direkt aus Mycelia entnimmt.
- Chunk-Generator um Biome/Strukturen erweitern, sobald die SubQG-Ausgabe für Höhen- und Feature-Maps verfügbar ist.
