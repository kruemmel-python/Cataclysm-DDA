package com.mycelia.mc.generation;

import org.bukkit.Bukkit;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.block.Block;
import org.bukkit.plugin.Plugin;

import java.time.Duration;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;

public class MyceliaEvolutionManager {

    private final Plugin plugin;
    private final Map<String, Integer> worldEvolutionBias = new ConcurrentHashMap<>();
    private final Duration interval;
    private final int mutationsPerWorld;

    public MyceliaEvolutionManager(Plugin plugin, Duration interval, int mutationsPerWorld) {
        this.plugin = plugin;
        this.interval = interval;
        this.mutationsPerWorld = Math.max(1, mutationsPerWorld);
    }

    public void setBias(String worldName, int bias) {
        worldEvolutionBias.put(worldName, bias);
    }

    public void start() {
        long ticks = Math.max(1, interval.toSeconds() * 20);
        Bukkit.getScheduler().runTaskTimer(plugin, this::evolveAllWorlds, ticks, ticks);
    }

    private void evolveAllWorlds() {
        for (World world : Bukkit.getWorlds()) {
            if (!world.isChunkLoaded(world.getSpawnLocation().getBlockX() >> 4, world.getSpawnLocation().getBlockZ() >> 4)) {
                continue;
            }
            evolveWorld(world);
        }
    }

    private void evolveWorld(World world) {
        int bias = worldEvolutionBias.getOrDefault(world.getName(), 0);
        long period = System.currentTimeMillis() / interval.toMillis();
        Random deterministic = new Random(world.getSeed() ^ (period << 3) ^ bias);

        for (int i = 0; i < mutationsPerWorld; i++) {
            int x = deterministic.nextInt(256) - 128 + world.getSpawnLocation().getBlockX();
            int z = deterministic.nextInt(256) - 128 + world.getSpawnLocation().getBlockZ();
            int y = world.getHighestBlockYAt(x, z);
            Block target = world.getBlockAt(x, y, z);

            if (target.getType() == Material.GRASS_BLOCK || target.getType() == Material.DIRT) {
                target.setType(Material.MYCELIUM);
                spreadToNeighbors(world, x, y, z);
            } else if (target.getType() == Material.MYCELIUM) {
                maybeCorrupt(world, deterministic, x, y, z);
            }
        }
    }

    private void spreadToNeighbors(World world, int x, int y, int z) {
        int[][] offsets = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int[] o : offsets) {
            Block neighbor = world.getBlockAt(x + o[0], y, z + o[1]);
            if (neighbor.getType() == Material.GRASS_BLOCK) {
                neighbor.setType(Material.MYCELIUM);
            }
        }
    }

    private void maybeCorrupt(World world, Random deterministic, int x, int y, int z) {
        if (deterministic.nextInt(5) != 0) {
            return;
        }
        Block below = world.getBlockAt(x, y - 1, z);
        if (below.getType().isSolid()) {
            below.setType(Material.MAGMA_BLOCK);
        }
    }
}
