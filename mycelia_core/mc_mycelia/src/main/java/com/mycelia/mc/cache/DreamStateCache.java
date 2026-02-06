package com.mycelia.mc.cache;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class DreamStateCache {
    private final Map<String, float[]> cache = new ConcurrentHashMap<>();

    public void put(String world, float[] gradient) {
        if (world == null || gradient == null) return;
        cache.put(world, gradient);
    }

    public float[] get(String world) {
        return cache.get(world);
    }

    public float getInfluence(String world, int chunkX, int chunkZ) {
        float[] gradient = cache.get(world);
        if (gradient == null || gradient.length == 0) return 0.5f;
        int idx = Math.floorMod(chunkX * 31 + chunkZ * 17, gradient.length);
        float val = gradient[idx];
        return Math.min(1f, Math.max(0f, val));
    }
}
