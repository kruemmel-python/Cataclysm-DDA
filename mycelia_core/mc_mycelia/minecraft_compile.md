# mc_mycelia – Kompilationsanleitung für das Minecraft-Plugin

Diese Anleitung beschreibt Schritt für Schritt, wie du das Paper/Spigot-Plugin **mc_mycelia** baust. Sie richtet sich an Nutzer:innen, die den Mycelia-Treiber (z.B. über SubQG + Mycel) als Seed-Quelle für Weltgenerierung in Minecraft nutzen wollen.

## Voraussetzungen
- **Java 17+ JDK** auf dem `PATH`. Das Build zielt auf **Java 17 Bytecode** (damit es zu Paper 1.20 passt). Auch ein JDK 21 funktioniert – dank `--release 17` wird dennoch Java-17-kompatibler Bytecode erzeugt.
- **Gradle** lokal installiert **oder** ein vorhandener Gradle-Wrapper (im Repo nicht enthalten, siehe unten, wie du ihn erzeugen kannst).
- Internetzugang für den ersten Build, damit die Paper-/Spigot-Abhängigkeit aus dem Repository geladen werden kann.

> Hinweis: Das Projekt nutzt `paper-api:1.20.6-R0.1-SNAPSHOT` als `compileOnly`-Abhängigkeit. Dadurch bleiben die Paper/Spigot-Klassen zur Kompilezeit verfügbar, werden aber nicht ins JAR gepackt (Bukkit-Server liefert sie zur Laufzeit).

## Projektstruktur (Build-relevant)
- `mc_mycelia/build.gradle` – definiert Java 17, Paper-Abhängigkeit und JAR-Namen (`mc_mycelia-<version>.jar`).
- `mc_mycelia/settings.gradle` – setzt den Projektnamen.
- `mc_mycelia/src/main/resources/plugin.yml` – Bukkit-Metadaten (wird ins JAR kopiert).
- Quellcode unter `mc_mycelia/src/main/java/...`.

## Build mit vorhandenem Gradle (empfohlen, leichtgewichtig)
1. Wechsle ins Modulverzeichnis:
   ```bash
   cd mc_mycelia
   ```
2. Baue das Plugin:
   ```bash
  gradle clean build
  ```
  - Der Build lädt beim ersten Lauf die Paper-API aus `https://repo.papermc.io/repository/maven-public/`.
  - Das fertige JAR findest du unter `mc_mycelia/build/libs/mc_mycelia-0.1.0.jar`.

## Build mit lokal erzeugtem Gradle-Wrapper (falls Gradle nicht installiert ist)
1. Wechsle ins Modulverzeichnis und erzeuge den Wrapper:
   ```bash
   cd mc_mycelia
   gradle wrapper
   ```
   Dadurch werden `gradlew`, `gradlew.bat` und der Ordner `gradle/` angelegt.
2. Verwende den Wrapper für den Build (plattformunabhängig):
   ```bash
   ./gradlew clean build   # macOS/Linux
   # oder
   gradlew.bat clean build # Windows
   ```
3. Ergebnis: `mc_mycelia/build/libs/mc_mycelia-0.1.0.jar`.

## Deployment auf dem Minecraft-Server
1. Kopiere das JAR nach `plugins/` deines **Paper**- oder **Spigot**-Servers.
2. Starte den Server (oder führe `/reload` auf einer Testinstanz aus).
3. Passe bei Bedarf die `config.yml` im neu entstandenen Plugin-Ordner an, z.B. um den Mycelia-Treiber/ SubQG-Befehl zu hinterlegen:
   ```yaml
   driver:
     command: "python ../python/mein_subqg_seed_script.py"
     timeoutSeconds: 5
   ```
4. Teste den Befehl im Spiel (mit passenden Rechten):
   ```
   /myceliaworld testwelt
   # optional mit explizitem Seed:
   /myceliaworld testwelt --seed 12345
   ```

## Fehlersuche beim Build
- **"Could not resolve io.papermc.paper:paper-api"**: Prüfe die Internetverbindung oder ob Maven-Repositories blockiert sind.
- **Toolchain/Version-Fehler mit JDK 21**: Das Build nutzt jetzt keine Gradle-Toolchain mehr, sondern dein installiertes JDK. Achte nur darauf, dass `java -version` ein JDK ≥ 17 zeigt. Beispiel (JDK 21): `openjdk version "21.x" ...` ist okay, weil der Compiler trotzdem `--release 17` setzt.
- **Wrapper fehlt**: Falls `./gradlew` nicht existiert, zuvor `gradle wrapper` ausführen (siehe oben).

## Nützliche Build-Kommandos
- Nur kompilieren (ohne Tests, es gibt derzeit keine):
  ```bash
  gradle assemble
  ```
- Abhängigkeiten neu laden (bei Cache-Problemen):
  ```bash
  gradle --refresh-dependencies clean build
  ```

## Hinweise zu reproduzierbaren Seeds
- Der Plugin-Build selbst ist deterministisch; die Welt-Seed-Beschaffung erfolgt zur Laufzeit.
- Für deterministische Welten setze entweder den `--seed`-Parameter im Befehl oder konfiguriere den Treiber so, dass er für identische Eingaben den gleichen Seed ausgibt.
