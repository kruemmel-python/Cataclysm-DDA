package com.mycelia.mc.command;

import com.mycelia.mc.MyceliaWorldPlugin;
import com.mycelia.mc.driver.MyceliaDriver;
import com.mycelia.mc.driver.MyceliaWorldData;
import com.mycelia.mc.driver.MyceliaWorldDNA;
import com.mycelia.mc.generation.MyceliaEvolutionManager;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;

import java.time.Duration;

public class MyceliaCommand implements CommandExecutor {

    private final MyceliaWorldPlugin plugin;
    private final MyceliaDriver driver;
    private final MyceliaEvolutionManager evolutionManager;

    public MyceliaCommand(MyceliaWorldPlugin plugin, MyceliaDriver driver, MyceliaEvolutionManager evolutionManager) {
        this.plugin = plugin;
        this.driver = driver;
        this.evolutionManager = evolutionManager;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        if (args.length == 0) {
            sender.sendMessage("§dMycelia§7: /mycelia attune|stabilize|corrupt|debug|health");
            return true;
        }

        switch (args[0].toLowerCase()) {
            case "attune" -> handleBias(sender, 1, "Welt wird an den Mycelia-Kern angepasst.");
            case "stabilize" -> handleBias(sender, 0, "Welt-Evolution wurde neutralisiert.");
            case "corrupt" -> handleBias(sender, 3, "Korruption verstärkt. Halte dich fest.");
            case "debug" -> handleDebug(sender);
            case "health" -> handleHealth(sender);
            case "lore" -> handleLore(sender);
            case "dream" -> handleDream(sender, args);
            case "otoc" -> handleOtoc(sender);
            case "timewarp" -> handleTimewarp(sender);
            default -> sender.sendMessage("§cUnbekannter Subcommand.");
        }
        return true;
    }

    private void handleBias(CommandSender sender, int bias, String message) {
        if (!sender.hasPermission("mc_mycelia.evolve")) {
            sender.sendMessage("§cKeine Berechtigung für Evolution.");
            return;
        }
        if (sender instanceof Player player) {
            String world = player.getWorld().getName();
            evolutionManager.setBias(world, bias);
            MyceliaWorldData worldData = plugin.getWorldData(world);
            if (worldData != null) {
                long mutatedSeed = worldData.seed() + bias * 13L;
                MyceliaWorldDNA dna = MyceliaWorldDNA.compute(mutatedSeed, worldData.baseBlock(), worldData.surfaceBlock(), worldData.oreBlock(), worldData.scale(), driver.getDriverVersion(), driver.getKernelFingerprint());
                plugin.updateWorldData(world, new MyceliaWorldData(
                        mutatedSeed,
                        worldData.baseBlock(),
                        worldData.surfaceBlock(),
                        worldData.oreBlock(),
                        worldData.scale(),
                        worldData.seaLevel(),
                        worldData.biomes(),
                        dna,
                        worldData.fromFallback()
                ));
            }
            sender.sendMessage("§a" + message);
        } else {
            sender.sendMessage("§cNur Spieler können Welt-Bias setzen.");
        }
    }

    private void handleDebug(CommandSender sender) {
        if (!sender.hasPermission("mc_mycelia.debug")) {
            sender.sendMessage("§cKeine Berechtigung für Debug.");
            return;
        }
        sender.sendMessage("§8§m-----§r §dMycelia Debug §8§m-----");
        sender.sendMessage("§7Letzter Treiber-Call: §f" + driver.getLastDriverCall());
        sender.sendMessage("§7Dauer: §f" + driver.getLastDriverDurationMs() + "ms");
        sender.sendMessage("§7Fallbacks: §f" + driver.getFallbackCount());
        sender.sendMessage("§7Fehler: §c" + (driver.getLastDriverError() == null ? "-" : driver.getLastDriverError()));
        boolean persistentOk = driver.getPersistentService().map(s -> {
            boolean ok = s.ping();
            return ok;
        }).orElse(false);
        sender.sendMessage("§7Persistent Mode: " + (driver.isPersistentEnabled() ? (persistentOk ? "§aok" : "§cdefekt") : "§8aus"));
    }

    private void handleHealth(CommandSender sender) {
        if (!sender.hasPermission("mc_mycelia.debug")) {
            sender.sendMessage("§cKeine Berechtigung.");
            return;
        }
        sender.sendMessage("§8§m-----§r §dMycelia Health §8§m-----");
        sender.sendMessage("§7Warmup-Timeout: §f" + Duration.ofSeconds(plugin.getConfig().getLong("driver.warmupTimeoutSeconds", 300)).toSeconds() + "s");
        sender.sendMessage("§7Treiber: §f" + driver.getDriverVersion());
        sender.sendMessage("§7Kernel: §f" + driver.getKernelFingerprint());
        sender.sendMessage("§7Persistent: §f" + (driver.isPersistentEnabled() ? "an" : "aus"));
    }

    private void handleLore(CommandSender sender) {
        if (!(sender instanceof Player player)) {
            sender.sendMessage("§cNur Spieler können Lore abrufen.");
            return;
        }
        if (!sender.hasPermission("mc_mycelia.debug")) {
            sender.sendMessage("§cKeine Berechtigung für Lore.");
            return;
        }
        float[] summary = plugin.getNarrativeSummaryCache().get(player.getWorld().getName());
        String lore = (summary != null && summary.length >= 2)
                ? new com.mycelia.mc.lore.LoreEngine().generateLore(summary[0], summary[1])
                : driver.requestEmergentLorePhrase(player.getWorld());
        player.sendMessage("§dMycelia-Lore§7: " + lore);
    }

    private void handleDream(CommandSender sender, String[] args) {
        if (!sender.hasPermission("mc_mycelia.debug")) {
            sender.sendMessage("§cKeine Berechtigung.");
            return;
        }
        String worldName = (sender instanceof Player p) ? p.getWorld().getName() : (args.length > 1 ? args[1] : null);
        if (worldName == null) {
            sender.sendMessage("§cBitte Welt angeben: /mycelia dream <world>");
            return;
        }
        float[] gradient = driver.requestDreamState(256);
        plugin.getDreamStateCache().put(worldName, gradient);
        sender.sendMessage("§7Dream-State aktualisiert für §e" + worldName + "§7 (" + gradient.length + " Werte).");
    }

    private void handleOtoc(CommandSender sender) {
        if (!sender.hasPermission("mc_mycelia.debug")) {
            sender.sendMessage("§cKeine Berechtigung.");
            return;
        }
        double chaos = driver.requestGlobalOTOC();
        plugin.setOtocChaosFactor(chaos);
        sender.sendMessage("§7Aktueller OTOC/Chaos-Faktor: §e" + chaos);
    }

    private void handleTimewarp(CommandSender sender) {
        if (!(sender instanceof Player player)) {
            sender.sendMessage("§cNur Spieler können Timewarp setzen.");
            return;
        }
        if (!sender.hasPermission("mc_mycelia.debug")) {
            sender.sendMessage("§cKeine Berechtigung.");
            return;
        }
        plugin.addTimeWarpZone(player.getLocation().getChunk());
        sender.sendMessage("§7Time-Warp-Zone für Chunk gesetzt.");
    }
}
