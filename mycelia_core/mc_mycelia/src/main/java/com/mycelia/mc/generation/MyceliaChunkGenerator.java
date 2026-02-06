package com.mycelia.mc.generation;

import com.mycelia.mc.MyceliaWorldPlugin;
import com.mycelia.mc.driver.MyceliaWorldData;
import com.mycelia.mc.generation.MyceliaLorePopulator;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.generator.BlockPopulator;
import org.bukkit.generator.ChunkGenerator;
import org.bukkit.util.noise.SimplexNoiseGenerator;

import java.util.List;
import java.util.Random;

public class MyceliaChunkGenerator extends ChunkGenerator {

    private final MyceliaWorldData data;
    private final SimplexNoiseGenerator terrainNoise;
    private final SimplexNoiseGenerator oreNoise;
    private final SimplexNoiseGenerator humidityNoise;
    private final SimplexNoiseGenerator temperatureNoise;
    private final Material baseMat;
    private final Material surfaceMat;
    private final Material oreMat;
    private final int seaLevel;
    private final List<MyceliaBiomeProfile> biomes;
    private final MyceliaWorldPlugin plugin;

    public MyceliaChunkGenerator(MyceliaWorldPlugin plugin, MyceliaWorldData data) {
        this.data = data;
        this.plugin = plugin;
        this.terrainNoise = new SimplexNoiseGenerator(data.seed());
        this.oreNoise = new SimplexNoiseGenerator(data.seed() ^ 0xCAFEEBABEL);
        this.humidityNoise = new SimplexNoiseGenerator(data.seed() ^ 0xAA12BBL);
        this.temperatureNoise = new SimplexNoiseGenerator(data.seed() ^ 0xBB55DDL);
        this.baseMat = materialOrDefault(data.baseBlock(), Material.STONE);
        this.surfaceMat = materialOrDefault(data.surfaceBlock(), Material.MYCELIUM);
        this.oreMat = materialOrDefault(data.oreBlock(), Material.AMETHYST_BLOCK);
        this.seaLevel = data.seaLevel();
        this.biomes = data.biomes();
    }

    @Override
    public ChunkData generateChunkData(World world, Random random, int chunkX, int chunkZ, BiomeGrid biome) {
        ChunkData chunk = createChunkData(world);
        double scale = data.scale();

        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                boolean surfacePlaced = false;
                double worldX = (chunkX << 4) + x;
                double worldZ = (chunkZ << 4) + z;
                double humidity = normalize(humidityNoise.noise(worldX * scale, worldZ * scale));
                double temperature = normalize(temperatureNoise.noise(worldX * scale, worldZ * scale));
                MyceliaBiomeProfile biomeProfile = resolveBiome(humidity, temperature);
                Material base = biomeProfile != null ? biomeProfile.base() : baseMat;
                Material surface = biomeProfile != null ? biomeProfile.surface() : surfaceMat;
                Material ore = biomeProfile != null ? biomeProfile.ore() : oreMat;
                for (int y = 120; y >= world.getMinHeight(); y--) {
                    double density = terrainNoise.noise(worldX * scale, y * (scale * 1.5), worldZ * scale);
                    double finalValue = density + (1.0 - (y / 85.0));

                    if (finalValue > 0.5) {
                        double oreValue = oreNoise.noise(worldX * scale * 4, y * scale * 4, worldZ * scale * 4);
                        if (oreValue > 0.8) {
                            chunk.setBlock(x, y, z, ore);
                        } else if (!surfacePlaced && y > seaLevel) {
                            chunk.setBlock(x, y, z, surface);
                            surfacePlaced = true;
                        } else {
                            chunk.setBlock(x, y, z, base);
                        }
                    } else if (y < seaLevel) {
                        chunk.setBlock(x, y, z, Material.WATER);
                    }
                }
            }
        }
        return chunk;
    }

    @Override
    public List<BlockPopulator> getDefaultPopulators(World world) {
        return List.of(
                new MyceliaStructurePopulator(plugin),
                new MyceliaUndergroundPopulator(plugin),
                new MyceliaLorePopulator(plugin.getDriver())
        );
    }

    private Material materialOrDefault(String name, Material fallback) {
        Material material = Material.matchMaterial(name);
        return material != null ? material : fallback;
    }

    private double normalize(double noise) {
        return Math.min(1.0, Math.max(0.0, (noise + 1) / 2));
    }

    private MyceliaBiomeProfile resolveBiome(double humidity, double temperature) {
        if (biomes == null) {
            return null;
        }
        for (MyceliaBiomeProfile profile : biomes) {
            if (profile.matches(humidity, temperature)) {
                return profile;
            }
        }
        return null;
    }
}
