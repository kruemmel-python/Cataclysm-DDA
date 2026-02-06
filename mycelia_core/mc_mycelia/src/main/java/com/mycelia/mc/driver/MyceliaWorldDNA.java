package com.mycelia.mc.driver;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.StringJoiner;

/**
 * Eindeutige Kennung einer generierten Mycelia-Welt. Hilft bei Reproduzierbarkeit und Debugging
 * (Kernel-/Treiber-Versionen, Parameter-Änderungen usw.).
 */
public record MyceliaWorldDNA(
        String hash,
        String driverVersion,
        String kernelFingerprint,
        String palette,
        double scale
) {

    public static MyceliaWorldDNA compute(long seed,
                                          String baseBlock,
                                          String surfaceBlock,
                                          String oreBlock,
                                          double scale,
                                          String driverVersion,
                                          String kernelFingerprint) {
        String palette = baseBlock + ":" + surfaceBlock + ":" + oreBlock;
        String input = new StringJoiner("|")
                .add(String.valueOf(seed))
                .add(palette)
                .add(String.valueOf(scale))
                .add(driverVersion)
                .add(kernelFingerprint)
                .toString();

        return new MyceliaWorldDNA("MYC-" + hash(input), driverVersion, kernelFingerprint, palette, scale);
    }

    private static String hash(String input) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-1");
            byte[] bytes = digest.digest(input.getBytes(StandardCharsets.UTF_8));
            String hex = HexFormat.of().formatHex(bytes);
            return hex.substring(0, 4) + "-" + hex.substring(4, 8) + "-" + hex.substring(8, 12) + "-" + hex.substring(12, 16);
        } catch (NoSuchAlgorithmException e) {
            return "unknown";
        }
    }
}
