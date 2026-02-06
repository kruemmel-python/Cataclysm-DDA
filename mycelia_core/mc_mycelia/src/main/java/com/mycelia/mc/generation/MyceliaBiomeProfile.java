package com.mycelia.mc.generation;

import org.bukkit.Material;

public class MyceliaBiomeProfile {
    private final String name;
    private final double humidityMin;
    private final double humidityMax;
    private final double temperatureMin;
    private final double temperatureMax;
    private final Material base;
    private final Material surface;
    private final Material ore;
    private final double hostileBoost;

    public MyceliaBiomeProfile(String name,
                               double humidityMin,
                               double humidityMax,
                               double temperatureMin,
                               double temperatureMax,
                               Material base,
                               Material surface,
                               Material ore,
                               double hostileBoost) {
        this.name = name;
        this.humidityMin = humidityMin;
        this.humidityMax = humidityMax;
        this.temperatureMin = temperatureMin;
        this.temperatureMax = temperatureMax;
        this.base = base;
        this.surface = surface;
        this.ore = ore;
        this.hostileBoost = hostileBoost;
    }

    public String name() {
        return name;
    }

    public boolean matches(double humidity, double temperature) {
        return humidity >= humidityMin && humidity <= humidityMax
                && temperature >= temperatureMin && temperature <= temperatureMax;
    }

    public Material base() {
        return base;
    }

    public Material surface() {
        return surface;
    }

    public Material ore() {
        return ore;
    }

    public double hostileBoost() {
        return hostileBoost;
    }
}
