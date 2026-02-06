# mc_mycelia – In-Game Befehle & Konfiguration

Diese Datei beschreibt alle ingame verfügbaren Kommandos (inkl. der jüngsten Funktionen) und die relevanten Konfigurationsoptionen aus `config.yml`, damit Admins die neuen Systeme steuern können.

## Befehle

### `/myceliaworld <weltname> [--seed <zahl>]`
Erzeugt eine neue Mycelia-Welt mit optionalem Seed. Nutzt den Treiber (oder Fallback) und speichert DNA/Metadaten.

### `/myceliaworld list`
Listet alle aktuell geladenen Welten.

### `/myceliaworld tp <weltname>`
Teleportiert den Spieler in die Zielwelt.

### `/myceliaworld remove <weltname> [--force]`
Entlädt und löscht die Welt. Mit `--force` auch wenn Spieler darin sind. Entfernt Einträge aus `worlds.yml`.

### `/myceliaworld info`
Zeigt Metadaten der aktuellen Mycelia-Welt (Seed, Blöcke, Scale, DNA, Meeresspiegel).

### `/mycelia attune`
Setzt die Evolutions-Bias der aktuellen Welt auf leichte Angleichung an den Kern (Bias 1) und regeneriert DNA/Seed-Drift.

### `/mycelia stabilize`
Neutralisiert die Evolution (Bias 0) und fixiert damit langsame Drift.

### `/mycelia corrupt`
Verstärkt Korruption (Bias 3) und mutiert DNA/Seed deutlicher.

### `/mycelia debug`
Zeigt Treiber-Telemetrie (letzter Call, Dauer, Fallbacks, Fehler, Persistent-Status).

### `/mycelia health`
Zeigt Warmup-Timeout, Treiberversion, Kernel-Fingerprint und Persistent-Status.

### `/mycelia lore`
Generiert eine aktuelle emergente Lore-Phrase aus dem Symbolic-Abstraction-Kernel und zeigt sie dem Spieler.

### `/mycelia dream [weltname]`
Aktualisiert den Dream-State-Gradienten (Ideal Gradient) für die angegebene Welt (Standard: eigene Welt) und speichert ihn im Cache; beeinflusst Dungeon-/Untergrund-Theming.

### `/mycelia otoc`
Ruft den globalen OTOC/Chaos-Faktor vom Treiber ab, aktualisiert den Plugin-Cache und zeigt den Wert an.

### `/mycelia timewarp`
Markiert den Chunk des Spielers als Zeit-Anomalie-Zone. Dort können Entitäten periodisch verlangsamt (Freeze-Ticks) werden, abhängig vom Chaos-Faktor.

## Rechte

- `mc_mycelia.world.generate`: Welten erstellen (`/myceliaworld <name>`)
- `mc_mycelia.world.remove`: Welten löschen
- `mc_mycelia.evolve`: Evolutions-Befehle (`attune`, `stabilize`, `corrupt`)
- `mc_mycelia.debug`: Diagnose (`debug`, `health`, `dream`, `otoc`, `timewarp`)
- `mc_mycelia.lore`: Lore/Dream/OTOC/Timewarp (falls getrennt gesteuert werden soll)

## Relevante Konfiguration (`src/main/resources/config.yml`)

```yaml
driver:
  command: "python path/to/mein_subqg_seed_script.py"
  timeoutSeconds: 35
  warmupTimeoutSeconds: 300
  version: "mycelia-driver 0.4.1"
  kernelFingerprint: "subqg_simulation_step@a91f"
  persistent:
    enabled: false            # true aktiviert den Persistent Driver Mode
    command: ""               # optional eigener Startbefehl für persistenten Treiber

world:
  baseBlock: DEEPSLATE
  surfaceBlock: MYCELIUM
  oreBlock: AMETHYST_BLOCK
  seaLevel: 45
  scale: 0.025

evolution:
  intervalMinutes: 5          # Intervall für Evolution/Spread
  mutationsPerWorld: 4        # Anzahl der Mutationen pro Zyklus

biomes:
  myzel-invasion:
    humidity: [0.6, 1.0]
    temperature: [0.3, 0.7]
    blocks:
      base: DEEPSLATE
      surface: MYCELIUM
      ore: AMETHYST_BLOCK
    mobs:
      hostileBoost: 1.2
  verdorbener-wald:
    humidity: [0.2, 0.5]
    temperature: [0.4, 0.9]
    blocks:
      base: MOSS_BLOCK
    surface: PODZOL
      ore: EMERALD_BLOCK
    mobs:
      hostileBoost: 1.1
```

### Beispiel-Anpassungen

- **Persistent Driver aktivieren**: `driver.persistent.enabled: true` und `driver.persistent.command` auf den Python-Treiber setzen, um GPU-Kontext warm zu halten.
- **Dungeon-Theming stärker beeinflussen**: `evolution.mutationsPerWorld` erhöhen oder zusätzliche Biome mit extremen Humidity/Temperature-Bereichen hinzufügen.
- **Lore/Chaos testen**: `/mycelia lore` für Text, `/mycelia otoc` für Chaos-Faktor, `/mycelia timewarp` für lokale Zeit-Anomalie in einem Chunk.

## Hinweise

- `worlds.yml` speichert Seeds, Blöcke, DNA und Fallback-Flags. Beim Neustart lädt das Plugin diese Welten automatisch.
- Dream-State-Gradient und OTOC werden periodisch aktualisiert (10 min bzw. 5 min) und können per Kommando manuell erneuert werden.
