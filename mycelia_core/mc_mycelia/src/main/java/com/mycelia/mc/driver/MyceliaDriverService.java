package com.mycelia.mc.driver;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.StringJoiner;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.logging.Logger;

public class MyceliaDriverService {

    private final List<String> command;
    private final Duration requestTimeout;
    private final Logger logger;

    private ExecutorService executor;

    private Process process;
    private BufferedWriter stdin;
    private BufferedReader stdout;

    public MyceliaDriverService(List<String> command, Duration requestTimeout, Logger logger) {
        this.command = command;
        this.requestTimeout = requestTimeout;
        this.logger = logger;
    }

    public synchronized void start() {
        if (process != null && process.isAlive()) {
            return;
        }

        executor = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "mycelia-persistent-driver");
            t.setDaemon(true);
            return t;
        });

        ProcessBuilder builder = new ProcessBuilder(command);
        builder.redirectErrorStream(true);

        File workDir = inferWorkDir();
        if (workDir != null) {
            builder.directory(workDir);
        }

        try {
            process = builder.start();
            stdin = new BufferedWriter(new OutputStreamWriter(process.getOutputStream(), StandardCharsets.UTF_8));
            stdout = new BufferedReader(new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));

            // Kleine Startverzögerung, um Warmup-Spikes abzufangen (GPU/OpenCL etc.)
            try {
                Thread.sleep(500);
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            }

            logger.info("Persistenter Treiber-Prozess erfolgreich gestartet. Warte auf Warmup-Abschluss...");

        } catch (IOException e) {
            logger.warning("Persistenter Treiber konnte nicht gestartet werden: " + e.getMessage());
            stop();
        }
    }

    public synchronized void stop() {
        if (process != null) {
            process.destroyForcibly();
        }
        process = null;
        stdin = null;
        stdout = null;

        if (executor != null) {
            executor.shutdownNow();
            executor = null;
        }
    }

    public synchronized boolean ping() {
        Optional<String> response = sendWithRestart(Map.of("cmd", "health"));
        return response
                .map(s -> s.toLowerCase().contains("ok") || s.toLowerCase().contains("warm"))
                .orElse(false);
    }

    public synchronized Optional<MyceliaWorldData> requestWorld(Optional<Long> seed, MyceliaDriver driver) {
        java.util.Map<String, Object> payload = new java.util.LinkedHashMap<>();
        payload.put("cmd", "world");
        seed.ifPresent(value -> payload.put("seed", value));
        return sendWithRestart(payload).flatMap(driver::parsePayload);
    }

    public synchronized Optional<String> requestNoise(int x, int z) {
        return send(Map.of("cmd", "noise", "x", x, "z", z));
    }

    public synchronized Optional<float[]> requestSymbolicAbstraction(int signalCount, float[] narrativeEmbeds, float[] weights) {
        return sendWithRestart(Map.of(
                        "cmd", "symbolic_abstract",
                        "signalCount", signalCount,
                        "narrativeEmbeds", narrativeEmbeds,
                        "weights", weights))
                .map(JsonUtil::parseFloatArray);
    }

    public synchronized Optional<float[]> requestDreamState(int size) {
        return sendWithRestart(Map.of("cmd", "dream_state", "size", size))
                .map(JsonUtil::parseFloatArray);
    }

    public synchronized Optional<Double> requestOTOC() {
        return sendWithRestart(Map.of("cmd", "otoc_chaos"))
                .map(resp -> {
                    try {
                        return Double.parseDouble(resp.trim());
                    } catch (NumberFormatException e) {
                        float[] arr = JsonUtil.parseFloatArray(resp);
                        return arr.length > 0 ? (double) arr[0] : null;
                    }
                });
    }

    // --- Robust: einmal neu starten, wenn der erste Versuch keine Antwort liefert ---
    private Optional<String> sendWithRestart(Map<String, Object> payload) {
        Optional<String> first = send(payload);
        if (first.isPresent()) return first;

        logger.warning("Persistenter Treiber reagierte nicht. Versuche Neustart und Wiederholung.");
        stop();
        start();
        return send(payload);
    }

    private Optional<String> send(Map<String, Object> payload) {
        if (stdin == null || stdout == null) {
            start();
        }
        if (stdin == null || stdout == null || process == null || !process.isAlive()) {
            return Optional.empty();
        }

        String json = JsonUtil.toJson(payload);

        try {
            stdin.write(json);
            stdin.write("\n");
            stdin.flush();
        } catch (IOException e) {
            logger.warning("Persistenter Treiber: Schreiben fehlgeschlagen: " + e.getMessage());
            stop();
            return Optional.empty();
        }

        try {
            if (executor == null) {
                executor = Executors.newSingleThreadExecutor(r -> {
                    Thread t = new Thread(r, "mycelia-persistent-driver-fallback");
                    t.setDaemon(true);
                    return t;
                });
            }

            long deadline = System.currentTimeMillis() + requestTimeout.toMillis();

            // Timeout-gesteuerter Read-Loop:
            // - Akzeptiert JSON-Objekte {...}
            // - Akzeptiert JSON-Arrays [...]
            // - Ignoriert nur echte Log-Zeilen ([C], [INFO], ...)
            while (System.currentTimeMillis() < deadline) {
                long remaining = deadline - System.currentTimeMillis();
                if (remaining <= 0) break;

                Future<String> future = executor.submit(() -> {
                    if (process.isAlive()) {
                        return stdout.readLine();
                    }
                    return null;
                });

                final String line;
                try {
                    line = future.get(remaining, TimeUnit.MILLISECONDS);
                } catch (TimeoutException te) {
                    // Wichtig: Read-Task abbrechen, sonst hängt der Thread ggf. weiter in readLine()
                    future.cancel(true);
                    continue;
                }

                if (line == null) {
                    if (!process.isAlive()) {
                        logger.warning("Persistenter Treiber hat Pipe geschlossen (Prozess beendet).");
                        stop();
                    }
                    return Optional.empty();
                }

                String trimmed = line.trim();
                if (trimmed.isEmpty()) {
                    continue;
                }

                // 1) JSON-Objekt: Nutzlast
                if (trimmed.startsWith("{")) {
                    return Optional.of(trimmed);
                }

                // 2) JSON-Array: Nutzlast (WICHTIG: NICHT pauschal wegfiltern!)
                if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
                    if (looksLikeDriverLog(trimmed)) {
                        continue;
                    }
                    return Optional.of(trimmed);
                }

                // 3) Log-Zeilen, die mit '[' anfangen, aber kein Array sind: ignorieren
                if (trimmed.startsWith("[")) {
                    continue;
                }

                // 4) Plaintext-Antwort: direkt zurückgeben
                return Optional.of(trimmed);
            }

            logger.warning("Persistenter Treiber: Lese-Timeout abgelaufen. Keine gültige Antwort.");
            stop();
            return Optional.empty();

        } catch (Exception e) {
            logger.warning("Persistenter Treiber: Kommunikationsfehler/Fehler beim Warten: " + summarizeException(e));
            stop();
            return Optional.empty();
        }
    }

    private boolean looksLikeDriverLog(String line) {
        // Nur “klassische” Log-Prefixe werden ignoriert. Arrays wie [0.1, 0.2, ...] bleiben gültige Antworten.
        String lower = line.toLowerCase();
        return lower.startsWith("[c]")
                || lower.startsWith("[info]")
                || lower.startsWith("[warn]")
                || lower.startsWith("[error]")
                || lower.startsWith("[debug]");
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

    private File inferWorkDir() {
        // In deinem Setup ist command typischerweise:
        // [python.exe, <script.py>, ...]
        // Wir versuchen deshalb “das erste existierende File nach dem executable” als Script zu nehmen.
        if (command == null || command.size() < 2) return null;

        for (int i = 1; i < command.size(); i++) {
            File candidate = new File(command.get(i));
            if (candidate.exists()) {
                File parent = candidate.getParentFile();
                return (parent != null && parent.isDirectory()) ? parent : null;
            }
        }
        return null;
    }
}
