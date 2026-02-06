package com.mycelia.mc.generation;

import org.bukkit.Material;
import org.bukkit.configuration.ConfigurationSection;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class MyceliaBiomeRegistry {

    private final List<MyceliaBiomeProfile> profiles;

    public MyceliaBiomeRegistry(ConfigurationSection config) {
        this.profiles = Collections.synchronizedList(new ArrayList<>());
        load(config);
    }

    public List<MyceliaBiomeProfile> getProfiles() {
        return List.copyOf(profiles);
    }

    private void load(ConfigurationSection config) {
        if (config == null) return;
        for (String key : config.getKeys(false)) {
            ConfigurationSection biome = config.getConfigurationSection(key);
            if (biome == null) continue;

            java.util.List<Double> humidityList = biome.getDoubleList("humidity");
            double humidityMin = humidityList.size() > 0 ? humidityList.get(0) : biome.getDouble("humidity.0", 0.0);
            double humidityMax = humidityList.size() > 1 ? humidityList.get(1) : biome.getDouble("humidity.1", 1.0);

            java.util.List<Double> tempList = biome.getDoubleList("temperature");
            double temperatureMin = tempList.size() > 0 ? tempList.get(0) : biome.getDouble("temperature.0", 0.0);
            double temperatureMax = tempList.size() > 1 ? tempList.get(1) : biome.getDouble("temperature.1", 1.0);

            String base = biome.getString("blocks.base", "STONE");
            String surface = biome.getString("blocks.surface", "MYCELIUM");
            String ore = biome.getString("blocks.ore", "AMETHYST_BLOCK");
            double hostile = biome.getDouble("mobs.hostileBoost", 1.0);

            profiles.add(new MyceliaBiomeProfile(
                    key,
                    humidityMin,
                    humidityMax,
                    temperatureMin,
                    temperatureMax,
                    material(base, Material.STONE),
                    material(surface, Material.MYCELIUM),
                    material(ore, Material.AMETHYST_BLOCK),
                    hostile
            ));
        }
    }

    private Material material(String name, Material fallback) {
        Material mat = Material.matchMaterial(name);
        return mat != null ? mat : fallback;
    }
}
