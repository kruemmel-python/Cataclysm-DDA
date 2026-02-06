package com.mycelia.mc.driver;

import java.util.Map;
import java.util.StringJoiner;

class JsonUtil {
    private JsonUtil() {}

    public static String toJson(Map<String, Object> map) {
        StringJoiner joiner = new StringJoiner(",", "{", "}");
        map.forEach((k, v) -> {
            if (v == null) {
                joiner.add("\"" + k + "\":null");
            } else if (v instanceof Number || v instanceof Boolean) {
                joiner.add("\"" + k + "\":" + v);
            } else if (v.getClass().isArray()) {
                joiner.add("\"" + k + "\":" + arrayToJson(v));
            } else {
                joiner.add("\"" + k + "\":\"" + v + "\"");
            }
        });
        return joiner.toString();
    }

    public static float[] parseFloatArray(String value) {
        if (value == null || value.isBlank()) return new float[0];
        String trimmed = value.trim();
        if (trimmed.startsWith("[")) {
            trimmed = trimmed.substring(1);
        }
        if (trimmed.endsWith("]")) {
            trimmed = trimmed.substring(0, trimmed.length() - 1);
        }
        String[] parts = trimmed.split("[,\\s]+");
        float[] out = new float[parts.length];
        int idx = 0;
        for (String p : parts) {
            if (p.isBlank()) continue;
            try {
                out[idx++] = Float.parseFloat(p);
            } catch (NumberFormatException ignored) {
            }
        }
        return idx == out.length ? out : java.util.Arrays.copyOf(out, idx);
    }

    private static String arrayToJson(Object arr) {
        int len = java.lang.reflect.Array.getLength(arr);
        StringJoiner joiner = new StringJoiner(",", "[", "]");
        for (int i = 0; i < len; i++) {
            Object element = java.lang.reflect.Array.get(arr, i);
            if (element instanceof Number || element instanceof Boolean) {
                joiner.add(String.valueOf(element));
            } else {
                joiner.add("\"" + element + "\"");
            }
        }
        return joiner.toString();
    }
}
