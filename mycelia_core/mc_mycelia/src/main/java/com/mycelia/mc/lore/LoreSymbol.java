package com.mycelia.mc.lore;

public enum LoreSymbol {
    DRIVE_INIT(0.0f, 0.5f, "das Erwachen des Hungers"),
    EQUILIBRIUM(0.5f, 1.0f, "die Stille des Netzes"),
    TENSION(1.0f, 1.5f, "der drohende Zusammenbruch"),
    CURIOSITY(1.5f, 2.0f, "die Suche nach Licht"),
    WEAK(0.0f, 0.2f, "verblasste Erinnerung"),
    STRONG(0.8f, 1.0f, "unerschütterlicher Wille");

    private final float min;
    private final float max;
    private final String description;

    LoreSymbol(float min, float max, String description) {
        this.min = min;
        this.max = max;
        this.description = description;
    }

    public boolean matches(float value) {
        return value >= min && value < max;
    }

    public String getDescription() {
        return description;
    }
}
