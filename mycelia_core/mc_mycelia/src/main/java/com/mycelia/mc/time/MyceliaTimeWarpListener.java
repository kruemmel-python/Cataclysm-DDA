package com.mycelia.mc.time;

import com.mycelia.mc.MyceliaWorldPlugin;
import org.bukkit.Chunk;
import org.bukkit.entity.Entity;
import org.bukkit.entity.LivingEntity;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.world.ChunkLoadEvent;

public class MyceliaTimeWarpListener implements Listener {

    private final MyceliaWorldPlugin plugin;

    public MyceliaTimeWarpListener(MyceliaWorldPlugin plugin) {
        this.plugin = plugin;
    }

    @EventHandler
    public void onChunkLoad(ChunkLoadEvent event) {
        double chaos = plugin.getOtocChaosFactor();
        if (chaos > 0.9 && Math.random() < 0.05) {
            plugin.addTimeWarpZone(event.getChunk());
        }
    }

    public void tick() {
        double chaos = plugin.getOtocChaosFactor();
        for (var world : plugin.getServer().getWorlds()) {
            for (var player : world.getPlayers()) {
                Chunk chunk = player.getLocation().getChunk();
                if (plugin.isTimeWarpZone(chunk)) {
                    for (Entity entity : chunk.getEntities()) {
                        if (entity instanceof LivingEntity living && Math.random() < chaos) {
                            living.setFreezeTicks(living.getFreezeTicks() + 1);
                        }
                    }
                }
            }
        }
    }
}
