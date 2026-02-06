package com.mycelia.mc.interaction;

import com.mycelia.mc.driver.MyceliaDriver;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerInteractEntityEvent;

public class MyceliaNpcDialogue implements Listener {

    private final MyceliaDriver driver;

    public MyceliaNpcDialogue(MyceliaDriver driver) {
        this.driver = driver;
    }

    @EventHandler
    public void onNpcInteract(PlayerInteractEntityEvent event) {
        Entity entity = event.getRightClicked();
        Player player = event.getPlayer();
        if (entity.getCustomName() != null && entity.getCustomName().toLowerCase().contains("myzel")) {
            String lore = driver.requestEmergentLorePhrase(player.getWorld());
            player.sendMessage("§d" + entity.getCustomName() + "§7: " + lore);
        }
    }
}
