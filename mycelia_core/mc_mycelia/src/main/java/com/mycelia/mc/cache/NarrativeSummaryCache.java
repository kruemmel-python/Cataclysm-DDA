package com.mycelia.mc.cache;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class NarrativeSummaryCache {
    private final Map<String, float[]> cache = new ConcurrentHashMap<>();

    public void put(String world, float[] summary) {
        if (world == null || summary == null) return;
        cache.put(world, summary);
    }

    public float[] get(String world) {
        return cache.get(world);
    }

    public float getValueOrDefault(String world, int index, float def) {
        float[] arr = cache.get(world);
        if (arr == null || arr.length <= index) return def;
        return arr[index];
    }
}
