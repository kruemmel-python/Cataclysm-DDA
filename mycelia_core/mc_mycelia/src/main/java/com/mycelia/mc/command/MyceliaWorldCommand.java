package com.mycelia.mc.command;

import com.mycelia.mc.MyceliaWorldPlugin;
import com.mycelia.mc.driver.MyceliaDriver;
import com.mycelia.mc.driver.MyceliaWorldData;
import com.mycelia.mc.generation.MyceliaChunkGenerator;
import org.bukkit.Bukkit;
import org.bukkit.World;
import org.bukkit.WorldCreator;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.configuration.ConfigurationSection;
import org.bukkit.entity.Player;
import org.bukkit.scheduler.BukkitScheduler;

import java.io.File;
import java.time.Duration;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.stream.Collectors;

public class MyceliaWorldCommand implements CommandExecutor {

    private final MyceliaWorldPlugin plugin;
    private final MyceliaDriver driver;
    private final ConcurrentMap<String, Long> rateLimits = new ConcurrentHashMap<>();

    public MyceliaWorldCommand(MyceliaWorldPlugin plugin, MyceliaDriver driver) {
        this.plugin = plugin;
        this.driver = driver;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        if (args.length == 0) {
            sender.sendMessage("§e/myceliaworld list");
            sender.sendMessage("§e/myceliaworld tp <world>");
            sender.sendMessage("§e/myceliaworld remove <world> [--force]");
            sender.sendMessage("§e/myceliaworld info");
            return true;
        }

        switch (args[0].toLowerCase()) {
            case "list" -> handleList(sender);
            case "tp" -> handleTeleport(sender, args);
            case "remove" -> handleRemove(sender, args);
            case "info" -> handleInfo(sender);
            default -> handleCreate(sender, args);
        }
        return true;
    }

    private void handleList(CommandSender sender) {
        String worlds = Bukkit.getWorlds().stream()
                .map(World::getName)
                .collect(Collectors.joining(", "));

        sender.sendMessage("§aGeladene Welten:");
        sender.sendMessage("§7" + worlds);
    }

    private void handleTeleport(CommandSender sender, String[] args) {
        if (!(sender instanceof Player player)) {
            sender.sendMessage("§cNur Spieler können teleportiert werden.");
            return;
        }

        if (args.length < 2) {
            sender.sendMessage("§cUsage: /myceliaworld tp <world>");
            return;
        }

        World world = Bukkit.getWorld(args[1]);
        if (world == null) {
            sender.sendMessage("§cWelt nicht geladen oder existiert nicht.");
            return;
        }

        player.teleport(world.getSpawnLocation());
        sender.sendMessage("§aTeleportiert nach §e" + world.getName());
    }

    private void handleRemove(CommandSender sender, String[] args) {
        if (!sender.hasPermission("mc_mycelia.world.remove")) {
            sender.sendMessage("§cKeine Berechtigung zum Entfernen von Welten.");
            return;
        }
        if (args.length < 2) {
            sender.sendMessage("§cUsage: /myceliaworld remove <world> [--force]");
            return;
        }

        String worldName = args[1];
        boolean force = args.length >= 3 && args[2].equalsIgnoreCase("--force");

        World world = Bukkit.getWorld(worldName);
        if (world == null) {
            sender.sendMessage("§cWelt ist nicht geladen.");
            return;
        }

        if (!force && world.getPlayers().size() > 0) {
            sender.sendMessage("§cSpieler sind noch in der Welt. Nutze --force.");
            return;
        }

        world.getPlayers().forEach(p -> p.teleport(Bukkit.getWorlds().get(0).getSpawnLocation()));

        boolean unloaded = Bukkit.unloadWorld(world, false);
        if (!unloaded) {
            sender.sendMessage("§cWelt konnte nicht entladen werden.");
            return;
        }

        plugin.updateWorldData(worldName, null);

        File folder = world.getWorldFolder();
        Bukkit.getScheduler().runTaskAsynchronously(plugin, () -> {
            deleteDirectory(folder);
            plugin.deleteWorldMeta(worldName);
            Bukkit.getScheduler().runTask(plugin, () -> sender.sendMessage("§aWelt §e" + worldName + " §awurde gelöscht."));
        });
    }

    private void handleCreate(CommandSender sender, String[] args) {
        if (!sender.hasPermission("mc_mycelia.world.generate")) {
            sender.sendMessage("§cKeine Berechtigung.");
            return;
        }
        if (isRateLimited(sender)) {
            sender.sendMessage("§cRate-Limit aktiv. Bitte kurz warten.");
            return;
        }
        String worldName = args[0];
        Optional<Long> seed = parseSeed(args);
        sender.sendMessage("§7Hole Welt-Daten asynchron vom Mycelia-Treiber...");
        driver.resolveWorldDataAsync(seed)
                .whenComplete((data, error) -> {
                    MyceliaWorldData worldData = data;
                    boolean usedFallback = false;
                    if (error != null || data == null) {
                        plugin.getLogger().warning("Asynchroner Treiber-Aufruf fehlgeschlagen: " + (error != null ? error.getMessage() : "unbekannt"));
                        worldData = driver.createFallbackData(seed.orElseGet(() -> System.nanoTime()));
                        usedFallback = true;
                    }
                    BukkitScheduler scheduler = plugin.getServer().getScheduler();
                    MyceliaWorldData finalWorldData = worldData;
                    boolean finalUsedFallback = usedFallback;
                    scheduler.runTask(plugin, () -> createWorldSync(sender, worldName, finalWorldData, finalUsedFallback));
                });
    }

    private Optional<Long> parseSeed(String[] args) {
        for (int i = 0; i < args.length - 1; i++) {
            if ("--seed".equalsIgnoreCase(args[i]) || "-s".equalsIgnoreCase(args[i])) {
                try {
                    return Optional.of(Long.parseLong(args[i + 1]));
                } catch (NumberFormatException ignore) {
                    return Optional.empty();
                }
            }
        }
        return Optional.empty();
    }

    private void createWorldSync(CommandSender sender, String worldName, MyceliaWorldData data, boolean usedFallback) {
        java.util.List<com.mycelia.mc.generation.MyceliaBiomeProfile> biomes = new java.util.ArrayList<>(data.biomes());
        com.mycelia.mc.api.MyceliaAPI api = com.mycelia.mc.api.Mycelia.get();
        if (api instanceof com.mycelia.mc.api.Mycelia mycelia) {
            mycelia.getBiomeModifiers().forEach(mod -> mod.accept(biomes));
        }
        data = new MyceliaWorldData(
                data.seed(),
                data.baseBlock(),
                data.surfaceBlock(),
                data.oreBlock(),
                data.scale(),
                data.seaLevel(),
                biomes,
                data.dna(),
                data.fromFallback()
        );
        MyceliaChunkGenerator generator = new MyceliaChunkGenerator(plugin, data);

        WorldCreator creator = new WorldCreator(worldName);
        creator.generator(generator);
        creator.seed(data.seed());

        World world = Bukkit.createWorld(creator);
        if (world != null) {
            plugin.saveWorldMeta(worldName, data);
            plugin.registerActiveWorld(worldName, data);
            if (usedFallback) {
                sender.sendMessage("§eTreiber lieferte keine Daten, nutze sichere Fallback-Daten.");
            }
            sender.sendMessage("§aNeue Mycelia-Welt erzeugt: " + world.getName() + " (Seed: " + data.seed() + ") DNA: " + (data.dna() != null ? data.dna().hash() : "n/a"));
            if (sender instanceof Player player && player.isOnline()) {
                player.teleport(world.getSpawnLocation());
            }
        } else {
            sender.sendMessage("§cWelt konnte nicht erzeugt werden. Siehe Server-Logs für Details.");
        }
    }

    private void deleteDirectory(File file) {
        if (file.isDirectory()) {
            File[] files = file.listFiles();
            if (files != null) {
                for (File f : files) {
                    deleteDirectory(f);
                }
            }
        }
        file.delete();
    }

    private void handleInfo(CommandSender sender) {
        if (!(sender instanceof Player player)) {
            sender.sendMessage("§cNur Spieler können diesen Befehl nutzen.");
            return;
        }

        String worldName = player.getWorld().getName();
        ConfigurationSection meta = plugin.getWorldMeta(worldName);

        if (meta == null) {
            sender.sendMessage("§cKeine Mycelia-Metadaten für diese Welt gefunden.");
            return;
        }

        sender.sendMessage("§8§m      §r §d§lMycelia Welt-Info §8§m      ");
        sender.sendMessage("§7Welt: §f" + worldName);
        sender.sendMessage("§7Seed: §e" + meta.getLong("seed"));
        sender.sendMessage("§7Oberfläche: §a" + meta.getString("surfaceBlock"));
        sender.sendMessage("§7Basis: §7" + meta.getString("baseBlock"));
        sender.sendMessage("§7Erz-Adern: §d" + meta.getString("oreBlock"));
        sender.sendMessage("§7Skalierung: §b" + meta.getDouble("scale"));
        sender.sendMessage("§7Meeresspiegel: §3" + meta.getInt("seaLevel"));
        ConfigurationSection dna = meta.getConfigurationSection("dna");
        if (dna != null) {
            sender.sendMessage("§7DNA: §5" + dna.getString("hash") + " §7Treiber: §f" + dna.getString("driver") + " §7Kernel: §f" + dna.getString("kernel"));
        }
        sender.sendMessage("§8§m                             ");
    }

    private boolean isRateLimited(CommandSender sender) {
        long now = System.currentTimeMillis();
        long cooldownMs = Duration.ofSeconds(5).toMillis();
        String key = sender.getName();
        Long last = rateLimits.get(key);
        if (last != null && (now - last) < cooldownMs) {
            return true;
        }
        rateLimits.put(key, now);
        return false;
    }
}
