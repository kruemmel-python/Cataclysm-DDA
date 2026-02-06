package com.mycelia.mc.generation;

import com.mycelia.mc.MyceliaWorldPlugin;
import org.bukkit.Chunk;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.generator.BlockPopulator;

import java.util.Random;

public class MyceliaStructurePopulator extends BlockPopulator {

    private final MyceliaWorldPlugin plugin;

    public MyceliaStructurePopulator(MyceliaWorldPlugin plugin) {
        this.plugin = plugin;
    }

    @Override
    public void populate(World world, Random random, Chunk source) {
        if (random.nextInt(100) < 2) {
            int x = (source.getX() << 4) + random.nextInt(16);
            int z = (source.getZ() << 4) + random.nextInt(16);
            int y = world.getHighestBlockYAt(x, z);

            float influence = plugin.getDreamInfluence(world.getName(), source.getX(), source.getZ());
            if (influence > 0.8f && random.nextInt(100) < 5) {
                createMagmaCorruptedDungeon(world, x, y, z, random);
                return;
            } else if (influence < 0.2f && random.nextInt(100) < 5) {
                createAmethystSanctuary(world, x, y, z, random);
                return;
            }

            if (y > 45 && world.getBlockAt(x, y, z).getType() != Material.WATER) {
                createMushroomDungeon(world, x, y, z, random);
            }
        }
    }

    private void createMushroomDungeon(World world, int x, int y, int z, Random random) {
        int size = random.nextInt(3) + 3; // 3x3 bis 5x5

        for (int ox = -size; ox <= size; ox++) {
            for (int oz = -size; oz <= size; oz++) {
                for (int oy = 0; oy < 4; oy++) {
                    Material mat = Material.AIR;

                    if (Math.abs(ox) == size || Math.abs(oz) == size) {
                        mat = Material.MUSHROOM_STEM;
                    } else if (oy == 0) {
                        mat = Material.CHISELED_DEEPSLATE;
                    } else if (oy == 3) {
                        mat = Material.SHROOMLIGHT;
                    }

                    if (mat != Material.AIR) {
                        world.getBlockAt(x + ox, y + oy, z + oz).setType(mat);
                    }
                }
            }
        }

        world.getBlockAt(x, y + 1, z).setType(Material.LOOM);
    }

    private void createMagmaCorruptedDungeon(World world, int x, int y, int z, Random random) {
        int size = random.nextInt(2) + 2;
        for (int ox = -size; ox <= size; ox++) {
            for (int oz = -size; oz <= size; oz++) {
                for (int oy = 0; oy < 4; oy++) {
                    Material mat = (Math.abs(ox) == size || Math.abs(oz) == size) ? Material.MAGMA_BLOCK : Material.BASALT;
                    if (oy == 0) mat = Material.MAGMA_BLOCK;
                    if (random.nextInt(10) == 0) mat = Material.SOUL_FIRE;
                    world.getBlockAt(x + ox, y + oy, z + oz).setType(mat);
                }
            }
        }
    }

    private void createAmethystSanctuary(World world, int x, int y, int z, Random random) {
        int radius = random.nextInt(2) + 2;
        for (int ox = -radius; ox <= radius; ox++) {
            for (int oz = -radius; oz <= radius; oz++) {
                for (int oy = 0; oy < 3; oy++) {
                    double dist = Math.sqrt(ox * ox + oz * oz);
                    if (dist <= radius) {
                        Material mat = oy == 0 ? Material.AMETHYST_BLOCK : Material.CALCITE;
                        world.getBlockAt(x + ox, y + oy, z + oz).setType(mat);
                    }
                }
            }
        }
        world.getBlockAt(x, y + 1, z).setType(Material.ENCHANTING_TABLE);
    }
}
