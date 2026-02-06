package com.mycelia.mc.driver;

import com.mycelia.mc.generation.MyceliaBiomeProfile;
import com.mycelia.mc.generation.MyceliaBiomeRegistry;
import org.bukkit.Material;
import org.bukkit.configuration.file.FileConfiguration;
import org.bukkit.plugin.Plugin;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.StringJoiner;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

public class MyceliaDriver {

    private final Logger logger;
    private final String driverCommand;

    private final Duration timeoutRuntime;
    private final Duration timeoutWarmup;

    private final boolean persistentEnabled;
    private final MyceliaDriverService persistentService;

    private final SecureSeedFallback fallback;
    private final String defaultBaseBlock;
    private final String defaultSurfaceBlock;
    private final String defaultOreBlock;
    private final double defaultScale;
    private final int defaultSeaLevel;

    private final String driverVersion;
    private final String kernelFingerprint;
    private final List<MyceliaBiomeProfile> biomeProfiles;

    private volatile Instant lastDriverCall = Instant.EPOCH;
    private volatile long lastDriverDurationMs = 0L;
    private volatile int fallbackCount = 0;
    private volatile String lastDriverError = "";

    public MyceliaDriver(Plugin plugin, FileConfiguration config) {
        this.logger = plugin.getLogger();
        this.driverCommand = normalize(config.getString("driver.command", ""));

        int runtimeSeconds = config.getInt("driver.timeoutSeconds", 35);
        int warmupSeconds = config.getInt("driver.warmupTimeoutSeconds", 300);

        this.timeoutRuntime = Duration.ofSeconds(Math.max(runtimeSeconds, 1));
        this.timeoutWarmup = Duration.ofSeconds(Math.max(warmupSeconds, 1));

        this.persistentEnabled = config.getBoolean("driver.persistent.enabled", false);
        String persistentCmd = normalize(config.getString("driver.persistent.command", driverCommand));
        this.persistentService = persistentEnabled && !persistentCmd.isBlank()
                ? new MyceliaDriverService(tokenize(persistentCmd), timeoutRuntime, logger)
                : null;

        this.fallback = new SecureSeedFallback();

        this.defaultBaseBlock = config.getString("world.baseBlock", "STONE");
        this.defaultSurfaceBlock = config.getString("world.surfaceBlock", "MYCELIUM");
        this.defaultOreBlock = config.getString("world.oreBlock", "AMETHYST_BLOCK");
        this.defaultScale = config.getDouble("world.scale", 0.025D);
        this.defaultSeaLevel = config.getInt("world.seaLevel", 40);

        this.driverVersion = config.getString("driver.version", "mycelia-driver 0.0.0");
        this.kernelFingerprint = config.getString("driver.kernelFingerprint", "unknown-kernel");

        this.biomeProfiles = new MyceliaBiomeRegistry(config.getConfigurationSection("biomes")).getProfiles();
    }

    public CompletableFuture<MyceliaWorldData> resolveWorldDataAsync(Optional<Long> explicitSeed) {
        return CompletableFuture.supplyAsync(() -> {
            if (explicitSeed.isPresent()) {
                return createFallbackData(explicitSeed.get());
            }
            return requestWorldDataFromDriver(timeoutRuntime).orElseGet(() -> {
                logger.warning("Fallback auf sichere Welt-Daten, Treiber nicht erreichbar oder Antwort unbrauchbar.");
                return createFallbackData(fallback.nextSeed());
            });
        });
    }

    public CompletableFuture<Void> warmupAsync() {
        return CompletableFuture.runAsync(() -> {
            if (driverCommand == null || driverCommand.isBlank()) {
                return;
            }
            if (persistentEnabled && persistentService != null) {
                persistentService.start();
                persistentService.ping();
            }
        });
    }

    public String requestEmergentLorePhrase(org.bukkit.World world) {
        try {
            float[] noise = sampleSpawnField(world, 1024);
            float archetype = average(noise, 0, noise.length / 2);
            float energy = average(noise, noise.length / 2, noise.length);
            Optional<float[]> symbolic = persistentService != null
                    ? persistentService.requestSymbolicAbstraction(32, noise, noise)
                    : Optional.empty();
            float[] used = symbolic.orElse(noise);
            if (used.length >= 2) {
                archetype = used[0];
                energy = used[1];
            }
            return new com.mycelia.mc.lore.LoreEngine().generateLore(archetype, energy);
        } catch (Exception e) {
            logger.warning("Lore-Generator Fehler: " + e.getMessage());
            return "Wir sahen die Stille des Netzes und es war wie verblasste Erinnerung.";
        }
    }

    public float[] requestDreamState(int size) {
        int safeSize = Math.max(16, size);
        Optional<float[]> response = persistentService != null ? persistentService.requestDreamState(safeSize) : Optional.empty();
        return response.filter(arr -> arr.length > 0).orElse(generateDeterministicArray(safeSize, 0.13f));
    }

    public double requestGlobalOTOC() {
        Optional<Double> otoc = persistentService != null ? persistentService.requestOTOC() : Optional.empty();
        return otoc.orElse(0.5D);
    }

    private Optional<MyceliaWorldData> requestWorldDataFromDriver(Duration timeout) {
        Instant start = Instant.now();
        Optional<MyceliaWorldData> result = Optional.empty();

        try {
            if (persistentEnabled && persistentService != null) {
                // Nur den Persistent Service aufrufen. Er handhabt den Neustart selbst.
                result = persistentService.requestWorld(Optional.empty(), this);
                if (result.isPresent()) {
                    result.ifPresent(data -> lastDriverError = "");
                    return result;
                }
                logger.warning("Persistenter Treiber konnte keine Welt-Daten liefern – wechsle auf Einmal-Aufruf.");
            }

            // Fallback auf den Einmal-Prozess-Aufruf (langsamer, aber sicher)
            if (driverCommand == null || driverCommand.isBlank()) {
                return Optional.empty();
            }

            List<String> command = tokenize(driverCommand);
            if (command.isEmpty()) {
                return Optional.empty();
            }

            ProcessBuilder builder = new ProcessBuilder(command);
            builder.redirectErrorStream(false);

            File workDir = inferWorkDirFromCommand(command);
            if (workDir != null) {
                builder.directory(workDir);
            }

            builder.environment().put("MYCELIA_DRIVER_TIMEOUT", String.valueOf(timeout.toSeconds()));

            List<String> stdoutLines = Collections.synchronizedList(new ArrayList<>());
            StringBuilder stderrAll = new StringBuilder(4096);

            Process process = builder.start();

            Thread stdoutGobbler = new Thread(
                    () -> readLines(process.getInputStream(), stdoutLines, null),
                    "mycelia-stdout-gobbler"
            );
            Thread stderrGobbler = new Thread(
                    () -> readLines(process.getErrorStream(), null, stderrAll),
                    "mycelia-stderr-gobbler"
            );

            stdoutGobbler.setDaemon(true);
            stderrGobbler.setDaemon(true);
            stdoutGobbler.start();
            stderrGobbler.start();

            boolean finished = process.waitFor(timeout.toMillis(), TimeUnit.MILLISECONDS);
            if (!finished) {
                process.destroyForcibly();
                logger.warning("Mycelia-Treiber überschritt Timeout von " + timeout.toSeconds() + "s.");
                joinQuietly(stdoutGobbler, 200);
                joinQuietly(stderrGobbler, 200);
                logStderrSmart(stderrAll.toString());
                return Optional.empty();
            }

            joinQuietly(stdoutGobbler, 500);
            joinQuietly(stderrGobbler, 500);
            logStderrSmart(stderrAll.toString());

            String payload = lastNonBlank(stdoutLines);
            if (payload == null) {
                return Optional.empty();
            }

            return parsePayload(payload);

        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            logger.warning("Treiber-Aufruf unterbrochen: " + summarizeException(ex));
            lastDriverError = ex.getMessage();
            return Optional.empty();
        } catch (Exception ex) {
            logger.warning("Treiber-Aufruf fehlgeschlagen: " + summarizeException(ex));
            lastDriverError = ex.getMessage();
            return Optional.empty();
        } finally {
            lastDriverCall = Instant.now();
            lastDriverDurationMs = Duration.between(start, lastDriverCall).toMillis();
        }
    }

    /**
     * Loggt STDERR abhängig vom Inhalt:
     * - WARN: Timeout/Fehler/Exception/Fatal/Traceback/AccessViolation/etc.
     * - INFO: normale Statusmeldungen wie "Welt-Typ generiert ..."
     *
     * Zusätzlich: Kürzt sehr langen Text.
     */
    private void logStderrSmart(String stderrRaw) {
        if (stderrRaw == null) {
            return;
        }
        String stderr = stderrRaw.trim();
        if (stderr.isBlank()) {
            return;
        }

        String normalized = stderr.toLowerCase();

        boolean looksBad =
                normalized.contains("timeout")
                        || normalized.contains("fehler")
                        || normalized.contains("error")
                        || normalized.contains("exception")
                        || normalized.contains("traceback")
                        || normalized.contains("fatal")
                        || normalized.contains("access violation")
                        || normalized.contains("segmentation fault")
                        || normalized.contains("dll nicht gefunden")
                        || normalized.contains("context fail")
                        || normalized.contains("init fail")
                        || normalized.contains("process_buffer fail");

        // Log-Kürzung (falls z.B. C-Code viel ausgibt)
        final int maxLen = 1200;
        String toLog = stderr.length() > maxLen
                ? stderr.substring(0, maxLen) + " ...[truncated]"
                : stderr;

        if (looksBad) {
            logger.warning("Treiber-STDERR: " + toLog);
        } else {
            logger.info("Treiber-STDERR: " + toLog);
        }
    }

    private void readLines(InputStream stream, List<String> outLines, StringBuilder outText) {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) {
                if (outLines != null) {
                    if (!line.isBlank()) outLines.add(line.trim());
                }
                if (outText != null) {
                    if (!line.isBlank()) {
                        if (!outText.isEmpty()) outText.append(' ');
                        outText.append(line.trim());
                    }
                }
            }
        } catch (Exception ignored) {
            // best effort: hier nicht crashen
        }
    }

    private void joinQuietly(Thread t, long millis) {
        try {
            t.join(millis);
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
        }
    }

    private String lastNonBlank(List<String> lines) {
        if (lines == null || lines.isEmpty()) return null;
        for (int i = lines.size() - 1; i >= 0; i--) {
            String s = lines.get(i);
            if (s != null && !s.isBlank()) return s.trim();
        }
        return null;
    }

    private File inferWorkDirFromCommand(List<String> command) {
        if (command.size() < 2) return null;
        File script = new File(command.get(1));
        File parent = script.getParentFile();
        return (parent != null && parent.isDirectory()) ? parent : null;
    }

    private MyceliaWorldData parseJson(String json) {
        long seed = Long.parseLong(extract(json, "seed", String.valueOf(fallback.nextSeed())));
        String base = extract(json, "baseBlock", defaultBaseBlock);
        String surface = extract(json, "surfaceBlock", defaultSurfaceBlock);
        String ore = extract(json, "oreBlock", defaultOreBlock);
        double scale = Double.parseDouble(extract(json, "scale", String.valueOf(defaultScale)));
        int seaLevel = Integer.parseInt(extract(json, "seaLevel", String.valueOf(defaultSeaLevel)));
        boolean fromFallback = Boolean.parseBoolean(extract(json, "fallback", "false"));
        return sanitize(new MyceliaWorldData(seed, base, surface, ore, scale, seaLevel, biomeProfiles, MyceliaWorldDNA.compute(seed, base, surface, ore, scale, driverVersion, kernelFingerprint), fromFallback));
    }

    private String extract(String json, String key, String def) {
        int keyIndex = json.indexOf("\"" + key + "\"");
        if (keyIndex < 0) return def;

        String[] parts = json.substring(keyIndex).split(":", 2);
        if (parts.length < 2) return def;

        String valueAndRest = parts[1];
        int end = valueAndRest.indexOf(',');
        if (end < 0) end = valueAndRest.indexOf('}');
        if (end < 0) end = valueAndRest.length();

        String cleaned = valueAndRest.substring(0, end).replace("\"", "").trim();
        return cleaned.isEmpty() ? def : cleaned;
    }

    public MyceliaWorldData createFallbackData(long seed) {
        fallbackCount++;
        MyceliaWorldDNA dna = MyceliaWorldDNA.compute(seed, defaultBaseBlock, defaultSurfaceBlock, defaultOreBlock, defaultScale, driverVersion, kernelFingerprint);
        return new MyceliaWorldData(seed, defaultBaseBlock, defaultSurfaceBlock, defaultOreBlock, defaultScale, defaultSeaLevel, biomeProfiles, dna, true);
    }

    Optional<MyceliaWorldData> parsePayload(String payload) {
        String trimmed = payload.trim();
        if (trimmed.isEmpty()) return Optional.empty();

        try {
            if (trimmed.contains("{")) {
                return Optional.of(parseJson(trimmed));
            }

            if (trimmed.startsWith("[")) {
                // Status- oder Log-Zeile aus dem Treiber, keine Welt-Daten
                return Optional.empty();
            }

            if (trimmed.matches("-?\\d+")) {
                long seed = Long.parseLong(trimmed);
                return Optional.of(createFallbackData(seed));
            }

            Optional<MyceliaWorldData> kv = parseKeyValuePayload(trimmed);
            if (kv.isPresent()) return kv;

            if (!trimmed.startsWith("[")) {
                logger.warning("Treiber antwortete, aber Format nicht erkennbar: " + trimmed);
            }
            return Optional.empty();
        } catch (Exception ex) {
            logger.warning("Treiber-Antwort konnte nicht gelesen werden: " + summarizeException(ex));
            return Optional.empty();
        }
    }

    private Optional<MyceliaWorldData> parseKeyValuePayload(String payload) {
        String[] parts = payload.split("[,\\s]+");
        long seed = fallback.nextSeed();
        String base = defaultBaseBlock;
        String surface = defaultSurfaceBlock;
        String ore = defaultOreBlock;
        double scale = defaultScale;
        int seaLevel = defaultSeaLevel;
        boolean found = false;

        for (String part : parts) {
            if (!part.contains("=") && !part.contains(":")) continue;
            String[] kv = part.split("[:=]", 2);
            if (kv.length != 2) continue;

            String key = kv[0].trim();
            String value = kv[1].trim();

            switch (key) {
                case "seed" -> { seed = Long.parseLong(value); found = true; }
                case "baseBlock" -> { base = value; found = true; }
                case "surfaceBlock" -> { surface = value; found = true; }
                case "oreBlock" -> { ore = value; found = true; }
                case "scale" -> { scale = Double.parseDouble(value); found = true; }
                case "seaLevel" -> { seaLevel = Integer.parseInt(value); found = true; }
                default -> { }
            }
        }
        return found ? Optional.of(sanitize(new MyceliaWorldData(seed, base, surface, ore, scale, seaLevel, biomeProfiles, MyceliaWorldDNA.compute(seed, base, surface, ore, scale, driverVersion, kernelFingerprint), false))) : Optional.empty();
    }

    private MyceliaWorldData sanitize(MyceliaWorldData data) {
        Material base = Material.matchMaterial(data.baseBlock());
        Material surface = Material.matchMaterial(data.surfaceBlock());
        Material ore = Material.matchMaterial(data.oreBlock());
        if (base == null || surface == null || ore == null) {
            fallbackCount++;
            return createFallbackData(data.seed());
        }
        if (data.scale() <= 0.0D || data.scale() > 10.0D) {
            fallbackCount++;
            return createFallbackData(data.seed());
        }
        return data;
    }

    private float[] sampleSpawnField(org.bukkit.World world, int count) {
        float[] values = new float[count];
        org.bukkit.util.noise.SimplexNoiseGenerator sampler = new org.bukkit.util.noise.SimplexNoiseGenerator(world.getSeed());
        for (int i = 0; i < count; i++) {
            double x = world.getSpawnLocation().getX() + (i % 32);
            double z = world.getSpawnLocation().getZ() + (i / 32);
            values[i] = (float) sampler.noise(x * 0.05, z * 0.05);
        }
        return values;
    }

    private float average(float[] arr, int from, int to) {
        if (arr == null || arr.length == 0) return 0;
        int end = Math.min(arr.length, to);
        int start = Math.max(0, from);
        double sum = 0;
        int count = 0;
        for (int i = start; i < end; i++) {
            sum += arr[i];
            count++;
        }
        return count == 0 ? 0 : (float) (sum / count);
    }

    private float[] generateDeterministicArray(int size, float factor) {
        float[] arr = new float[size];
        java.util.Random r = new java.util.Random(Double.doubleToLongBits(factor));
        for (int i = 0; i < size; i++) {
            arr[i] = r.nextFloat();
        }
        return arr;
    }

    private String normalize(String value) {
        if (value == null) return "";
        String trimmed = value.trim();
        if (trimmed.length() >= 2 && trimmed.startsWith("\"") && trimmed.endsWith("\"")
                && trimmed.indexOf('"', 1) == trimmed.length() - 1) {
            return trimmed.substring(1, trimmed.length() - 1);
        }
        return trimmed;
    }

    private List<String> tokenize(String commandLine) {
        List<String> tokens = new ArrayList<>();
        boolean inQuotes = false;
        char quoteChar = 0;
        StringBuilder current = new StringBuilder();

        for (int i = 0; i < commandLine.length(); i++) {
            char c = commandLine.charAt(i);

            if (c == '"' || c == '\'') {
                if (inQuotes && quoteChar == c) {
                    inQuotes = false;
                    continue;
                }
                if (!inQuotes) {
                    inQuotes = true;
                    quoteChar = c;
                    continue;
                }
            }

            if (Character.isWhitespace(c) && !inQuotes) {
                if (current.length() > 0) {
                    tokens.add(current.toString());
                    current.setLength(0);
                }
            } else {
                current.append(c);
            }
        }

        if (current.length() > 0) tokens.add(current.toString());
        return tokens;
    }

    private String summarizeException(Exception ex) {
        StringJoiner joiner = new StringJoiner(" -> ");
        Throwable current = ex;
        while (current != null) {
            joiner.add(current.getClass().getSimpleName() + ": " + current.getMessage());
            current = current.getCause();
        }
        return joiner.toString();
    }

    public long getLastDriverDurationMs() {
        return lastDriverDurationMs;
    }

    public String getLastDriverError() {
        return lastDriverError;
    }

    public int getFallbackCount() {
        return fallbackCount;
    }

    public Instant getLastDriverCall() {
        return lastDriverCall;
    }

    public boolean isPersistentEnabled() {
        return persistentEnabled;
    }

    public Optional<MyceliaDriverService> getPersistentService() {
        return Optional.ofNullable(persistentService);
    }

    public List<MyceliaBiomeProfile> getBiomeProfiles() {
        return biomeProfiles;
    }

    public String getDriverVersion() {
        return driverVersion;
    }

    public String getKernelFingerprint() {
        return kernelFingerprint;
    }
}