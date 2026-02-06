package com.mycelia.mc.generation;

import com.mycelia.mc.driver.MyceliaDriver;
import org.bukkit.Chunk;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.block.Block;
import org.bukkit.block.Chest;
import org.bukkit.generator.BlockPopulator;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.meta.BookMeta;

import java.util.Random;

public class MyceliaLorePopulator extends BlockPopulator {

    private final MyceliaDriver driver;

    public MyceliaLorePopulator(MyceliaDriver driver) {
        this.driver = driver;
    }

    @Override
    public void populate(World world, Random random, Chunk source) {
        if (random.nextInt(100) > 4) {
            return;
        }
        int x = (source.getX() << 4) + random.nextInt(16);
        int z = (source.getZ() << 4) + random.nextInt(16);
        int y = world.getHighestBlockYAt(x, z);
        Block block = world.getBlockAt(x, y, z);
        if (block.getType() != Material.AIR) {
            return;
        }
        block.setType(Material.CHEST);
        if (block.getState() instanceof Chest chest) {
            ItemStack book = new ItemStack(Material.WRITTEN_BOOK);
            BookMeta meta = (BookMeta) book.getItemMeta();
            if (meta != null) {
                String lore = driver.requestEmergentLorePhrase(world);
                meta.setTitle("Myzel-Notizen");
                meta.setAuthor("Mycelia-Kern");
                meta.addPage(lore);
                book.setItemMeta(meta);
            }
            chest.getBlockInventory().addItem(book);
        }
    }
}
