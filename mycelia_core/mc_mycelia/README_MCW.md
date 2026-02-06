# Mycelia World Management (mc_mycelia)

Umfasst den `/myceliaworld`-Befehl, den Treiber-Workflow und die Mycelia-spezifische Weltgenerierung (Java 21 / Paper 1.21).

## Befehle (In-Game)
- `/myceliaworld list` – Zeigt alle geladenen Welten an.
- `/myceliaworld tp <world>` – Teleportiert Spieler zum Spawn einer geladenen Welt.
- `/myceliaworld remove <world> [--force]` – Evakuiert Spieler (oder erzwingt via `--force`), entlädt die Welt und löscht den Weltordner asynchron.
- `/myceliaworld <worldName> [--seed <long>]` – Erstellt eine neue Welt mit Daten aus dem Treiber; der optionale `--seed` übersteuert den Treiberseed.
- `/myceliaworld info` – Zeigt gespeicherte Mycelia-Metadaten der aktuellen Welt (Seed, Blöcke, Skalierung, Meeresspiegel).

## Treiber & Fallback
- Konfiguration: `mc_mycelia/src/main/resources/config.yml`
  - `driver.command`: Prozess, der Weltdaten ausgibt (s. Format unten). Beispiel (Windows): `"C:\\Users\\ralfk\\AppData\\Local\\Programs\\Python\\Python312\\python.exe D:/mc-test/paper/mein_subqg_seed_script.py"`.
  - `driver.timeoutSeconds`: Wartezeit pro Aufruf.
  - `world.*`: Fallback-Defaults, falls der Treiber keine gültigen Daten liefert (baseBlock, surfaceBlock, oreBlock, seaLevel, scale).
- Aufruf: Der Treiber wird asynchron gestartet. Ausgabe wird robust geparst (JSON, einfacher Seed, oder Key/Value-Liste). Bei Fehlern/Timeout greift ein kryptografisch sicherer Fallback.
- Unterstützte Ausgabeformate (eine Zeile):
  - JSON: `{ "seed":123, "baseBlock":"STONE", "surfaceBlock":"MYCELIUM", "oreBlock":"DIAMOND_ORE", "scale":0.03, "seaLevel":45 }`
  - Nur Seed: `123456789`
  - Key/Value: `seed=123 baseBlock=STONE surfaceBlock=MYCELIUM oreBlock=AMETHYST_BLOCK scale=0.03 seaLevel=45`

## Welt-Erzeugung (Schritte)
1) Spieler ruft `/myceliaworld <name> [--seed]` auf.
2) Treiber wird asynchron kontaktiert; bei Fehlern oder fehlenden Daten werden sichere Fallback-Daten genutzt.
3) Auf dem Main-Thread wird ein `WorldCreator` mit `MyceliaChunkGenerator` instanziiert und die Welt geladen/teleportiert.

## Terrain-Algorithmus (MyceliaChunkGenerator)
- Rauschbasis: Simplex Noise 3D, skaliert mit `scale` (Default 0.025) und vertikalem Gradienten.
- Dichte-Scan von oben (y=120) nach unten bis `world.getMinHeight()`:
  - Feste Blöcke, wenn `finalValue > 0.5` (Noise + Gradient).
  - Oberflächenschicht: erstes Festblock-Voxel oberhalb `seaLevel` wird zum `surfaceBlock` (Default MYCELIUM).
  - Unterboden: übrige Festblöcke werden `baseBlock` (Default DEEPSLATE).
  - Wasserfüllung: Bereiche unterhalb `seaLevel`, die nicht fest werden, erhalten `WATER`.
- Erz-Adern: Zweiter Simplex-Generator (`seed ^ 0xCAFEEBABEL`), skaliert 4×; ab Schwelle >0.8 wird `oreBlock` (Default AMETHYST_BLOCK) gesetzt.
- Material-Fallbacks: Ungültige Materialnamen werden automatisch auf Standard (STONE/MYCELIUM/AMETHYST_BLOCK) ersetzt.

## Strukturen (Populatoren)
- `MyceliaStructurePopulator` (2 % pro Chunk): erzeugt einen kleinen Pilz-Dungeon (3–5 Blöcke Kantenlänge), mit Myzel-Stämmen, deepslate Boden, Shroomlight-Decke und einer Loom-“Belohnung”. Nur auf festem Untergrund (y>45, kein Wasser).
- `MyceliaUndergroundPopulator` (3 % pro Chunk): sucht zwischen y=10–49 nach Hohlräumen auf solidem Boden und legt glühende Amethyst-Kammern mit Obsidian-Wänden an.

## Remove-Workflow
1) Prüft, ob die Welt geladen ist und ob Spieler online sind (`--force` überschreibt Blocker).
2) Teleportiert verbleibende Spieler in die erste geladene Welt.
3) Entlädt die Welt (`Bukkit.unloadWorld` ohne Speichern). Bei Fehlschlag Abbruch.
4) Löscht den Weltenordner rekursiv asynchron und informiert den Absender nach Abschluss.

## Warmup
- Beim Plugin-Start wird der Treiber einmal asynchron vorgewärmt, damit der erste Spielerbefehl nicht durch Prozessstart/IO gebremst wird.

## Hinweise für externe Treiber
- Eine (1) Zeile Output genügt; nur die letzte nicht-leere Zeile wird ausgewertet.
- Timeout oder Parser-Fehler triggern den Fallback, aber das Plugin bleibt funktionsfähig.
- Pfade mit Leerzeichen im `driver.command` können gequotet werden; das Command wird vor der Tokenisierung normalisiert.
- Dein Python-Skript kann thematische Paletten ausgeben; das Plugin übernimmt Materialnamen direkt via `Material.matchMaterial`.
- Achte darauf, dass Fehlermeldungen im Skript nur auf STDERR gehen; STDOUT muss eine einzige JSON-Zeile bleiben (sonst greift der Fallback).

## Python-Treiber (Beispiel: `mein_subqg_seed_script.py`)
- Liefert exakt eine JSON-Zeile mit den Keys: `seed`, `baseBlock`, `surfaceBlock`, `oreBlock`, `scale`, `seaLevel`.
- Wählt aus vordefinierten Paletten (z. B. Myzel-Invasion, Eis-Ödland, Vulkanisch, Verdorbener Wald, Überwuchert) anhand des Seeds passende Blöcke, um konsistente Themenwelten zu erzeugen.
- Beispiel-Aufruf (config `driver.command`): `python D:/mc-test/scripts/mein_subqg_seed_script.py`
- Loggt optional das gewählte Thema auf STDERR; der Treiber liest nur die letzte STDOUT-Zeile als Payload.
