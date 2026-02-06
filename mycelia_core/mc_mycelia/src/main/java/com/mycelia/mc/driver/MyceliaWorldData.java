package com.mycelia.mc.driver;

public record MyceliaWorldData(
        long seed,
        String baseBlock,
        String surfaceBlock,
        String oreBlock,
        double scale,
        int seaLevel,
        java.util.List<com.mycelia.mc.generation.MyceliaBiomeProfile> biomes,
        MyceliaWorldDNA dna,
        boolean fromFallback
) {
    public MyceliaWorldData {
        if (biomes == null) {
            biomes = java.util.List.of();
        }
    }
}
