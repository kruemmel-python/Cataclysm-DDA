package com.mycelia.mc;

import com.mycelia.mc.api.Mycelia;
import com.mycelia.mc.cache.DreamStateCache;
import com.mycelia.mc.cache.NarrativeSummaryCache;
import com.mycelia.mc.command.MyceliaCommand;
import com.mycelia.mc.command.MyceliaWorldCommand;
import com.mycelia.mc.driver.MyceliaDriver;
import com.mycelia.mc.driver.MyceliaDriverService;
import com.mycelia.mc.driver.MyceliaWorldDNA;
import com.mycelia.mc.driver.MyceliaWorldData;
import com.mycelia.mc.generation.MyceliaBiomeProfile;
import com.mycelia.mc.generation.MyceliaChunkGenerator;
import com.mycelia.mc.generation.MyceliaEvolutionManager;
import com.mycelia.mc.interaction.MyceliaNpcDialogue;
import com.mycelia.mc.time.MyceliaTimeWarpListener;
import org.bukkit.Bukkit;
import org.bukkit.World;
import org.bukkit.WorldCreator;
import org.bukkit.command.PluginCommand;
import org.bukkit.configuration.ConfigurationSection;
import org.bukkit.configuration.file.FileConfiguration;
import org.bukkit.configuration.file.YamlConfiguration;
import org.bukkit.plugin.java.JavaPlugin;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

public class MyceliaWorldPlugin extends JavaPlugin {

    private MyceliaDriver myceliaDriver;
    private MyceliaEvolutionManager evolutionManager;
    private final java.util.Map<String, MyceliaWorldData> activeWorlds = new java.util.concurrent.ConcurrentHashMap<>();
    private final DreamStateCache dreamStateCache = new DreamStateCache();
    private final NarrativeSummaryCache narrativeSummaryCache = new NarrativeSummaryCache();
    private volatile double otocChaosFactor = 0.5D;
    private final Set<String> timeWarpZones = ConcurrentHashMap.newKeySet();
    private MyceliaTimeWarpListener timeWarpListener;

    @Override
    public void onEnable() {
        saveDefaultConfig();

        this.myceliaDriver = new MyceliaDriver(this, getConfig());
        long evolutionMinutes = getConfig().getLong("evolution.intervalMinutes", 5L);
        int mutations = getConfig().getInt("evolution.mutationsPerWorld", 3);
        this.evolutionManager = new MyceliaEvolutionManager(this, java.time.Duration.ofMinutes(Math.max(1, evolutionMinutes)), mutations);
        this.evolutionManager.start();

        org.bukkit.util.noise.SimplexNoiseGenerator apiNoise = new org.bukkit.util.noise.SimplexNoiseGenerator(42L);
        Mycelia.install(new Mycelia(
                (x, z) -> myceliaDriver.getPersistentService()
                        .flatMap(service -> service.requestNoise(x, z))
                        .map(str -> {
                            try {
                                return Double.parseDouble(str);
                            } catch (NumberFormatException e) {
                                return apiNoise.noise(x * 0.01, z * 0.01);
                            }
                        })
                        .orElseGet(() -> apiNoise.noise(x * 0.01, z * 0.01)),
                worldName -> java.util.Optional.ofNullable(activeWorlds.get(worldName)).map(MyceliaWorldData::dna)
        ));

        PluginCommand command = getCommand("myceliaworld");
        if (command != null) {
            command.setExecutor(new MyceliaWorldCommand(this, myceliaDriver));
            getLogger().info("mc_mycelia bereit. Nutze /myceliaworld für neue Welten.");
        } else {
            getLogger().severe("mc_mycelia konnte den Command 'myceliaworld' nicht registrieren.");
        }

        PluginCommand myceliaCmd = getCommand("mycelia");
        if (myceliaCmd != null) {
            myceliaCmd.setExecutor(new MyceliaCommand(this, myceliaDriver, evolutionManager));
        }

        loadPersistedWorlds();

        this.timeWarpListener = new MyceliaTimeWarpListener(this);
        getServer().getPluginManager().registerEvents(timeWarpListener, this);
        startDreamStateTask();
        startOtocTask();
        startTimeWarpTick();

        getServer().getPluginManager().registerEvents(new MyceliaNpcDialogue(myceliaDriver), this);

        // Warmup: nutzt eigenes Warmup-Timeout (driver.warmupTimeoutSeconds)
        getServer().getScheduler().runTaskAsynchronously(this, () -> {
            getLogger().info("[mc_mycelia] Warmup: Treiber wird vorgeladen...");
            myceliaDriver.warmupAsync().join();
            getLogger().info("[mc_mycelia] Warmup: abgeschlossen.");
        });
    }

    @Override
    public void onDisable() {
        myceliaDriver.getPersistentService().ifPresent(MyceliaDriverService::stop);
    }

    public void saveWorldMeta(String worldName, MyceliaWorldData data) {
        File file = new File(getDataFolder(), "worlds.yml");
        FileConfiguration config = YamlConfiguration.loadConfiguration(file);

        String path = "worlds." + worldName;
        config.set(path + ".seed", data.seed());
        config.set(path + ".baseBlock", data.baseBlock());
        config.set(path + ".surfaceBlock", data.surfaceBlock());
        config.set(path + ".oreBlock", data.oreBlock());
        config.set(path + ".scale", data.scale());
        config.set(path + ".seaLevel", data.seaLevel());
        config.set(path + ".fallback", data.fromFallback());
        MyceliaWorldDNA dna = data.dna();
        if (dna != null) {
            config.set(path + ".dna.hash", dna.hash());
            config.set(path + ".dna.driver", dna.driverVersion());
            config.set(path + ".dna.kernel", dna.kernelFingerprint());
            config.set(path + ".dna.palette", dna.palette());
            config.set(path + ".dna.scale", dna.scale());
        }

        try {
            config.save(file);
        } catch (IOException e) {
            getLogger().severe("Konnte Welt-Metadaten für " + worldName + " nicht speichern!");
        }
    }

    public ConfigurationSection getWorldMeta(String worldName) {
        File file = new File(getDataFolder(), "worlds.yml");
        FileConfiguration config = YamlConfiguration.loadConfiguration(file);
        return config.getConfigurationSection("worlds." + worldName);
    }

    public void deleteWorldMeta(String worldName) {
        File file = new File(getDataFolder(), "worlds.yml");
        FileConfiguration config = YamlConfiguration.loadConfiguration(file);
        config.set("worlds." + worldName, null);
        try {
            config.save(file);
        } catch (IOException e) {
            getLogger().severe("Konnte Welt-Metadaten für " + worldName + " nicht löschen!");
        }
    }

    public void registerActiveWorld(String name, MyceliaWorldData data) {
        activeWorlds.put(name, data);
        evolutionManager.setBias(name, data.fromFallback() ? 1 : 0);
        dreamStateCache.put(name, myceliaDriver.requestDreamState(256));
    }

    public MyceliaEvolutionManager getEvolutionManager() {
        return evolutionManager;
    }

    public MyceliaWorldData getWorldData(String worldName) {
        return activeWorlds.get(worldName);
    }

    public void updateWorldData(String worldName, MyceliaWorldData data) {
        if (data != null) {
            activeWorlds.put(worldName, data);
            saveWorldMeta(worldName, data);
        } else {
            activeWorlds.remove(worldName);
            dreamStateCache.put(worldName, null);
            narrativeSummaryCache.put(worldName, null);
            timeWarpZones.removeIf(key -> key.startsWith(worldName + ":"));
        }
    }

    private void loadPersistedWorlds() {
        File file = new File(getDataFolder(), "worlds.yml");
        if (!file.exists()) {
            return;
        }
        FileConfiguration config = YamlConfiguration.loadConfiguration(file);
        ConfigurationSection worldsSection = config.getConfigurationSection("worlds");
        if (worldsSection == null) {
            return;
        }

        for (String worldName : worldsSection.getKeys(false)) {
            if (Bukkit.getWorld(worldName) != null) {
                continue;
            }
            ConfigurationSection meta = worldsSection.getConfigurationSection(worldName);
            if (meta == null) {
                continue;
            }

            long seed = meta.getLong("seed", System.nanoTime());
            String base = meta.getString("baseBlock", getConfig().getString("world.baseBlock", "STONE"));
            String surface = meta.getString("surfaceBlock", getConfig().getString("world.surfaceBlock", "MYCELIUM"));
            String ore = meta.getString("oreBlock", getConfig().getString("world.oreBlock", "AMETHYST_BLOCK"));
            double scale = meta.getDouble("scale", getConfig().getDouble("world.scale", 0.025D));
            int seaLevel = meta.getInt("seaLevel", getConfig().getInt("world.seaLevel", 40));
            boolean fallback = meta.getBoolean("fallback", false);

            ConfigurationSection dnaSection = meta.getConfigurationSection("dna");
            MyceliaWorldDNA dna = null;
            if (dnaSection != null) {
                dna = new MyceliaWorldDNA(
                        dnaSection.getString("hash", ""),
                        dnaSection.getString("driver", myceliaDriver.getDriverVersion()),
                        dnaSection.getString("kernel", myceliaDriver.getKernelFingerprint()),
                        dnaSection.getString("palette", base + ":" + surface + ":" + ore),
                        dnaSection.getDouble("scale", scale)
                );
            }
            if (dna == null) {
                dna = MyceliaWorldDNA.compute(seed, base, surface, ore, scale, myceliaDriver.getDriverVersion(), myceliaDriver.getKernelFingerprint());
            }

            java.util.List<MyceliaBiomeProfile> biomes = myceliaDriver.getBiomeProfiles();
            MyceliaWorldData data = new MyceliaWorldData(seed, base, surface, ore, scale, seaLevel, biomes, dna, fallback);

            WorldCreator creator = new WorldCreator(worldName);
            creator.generator(new MyceliaChunkGenerator(this, data));
            creator.seed(seed);

            World world = Bukkit.createWorld(creator);
            if (world != null) {
                registerActiveWorld(worldName, data);
                getLogger().info("[mc_mycelia] Welt aus worlds.yml geladen: " + worldName);
            } else {
                getLogger().warning("[mc_mycelia] Welt konnte nicht geladen werden: " + worldName);
            }
        }
    }

    private void startDreamStateTask() {
        long periodTicks = 20L * 60L * 10L; // alle 10 Minuten
        getServer().getScheduler().runTaskTimerAsynchronously(this, () -> {
            for (String worldName : activeWorlds.keySet()) {
                float[] gradient = myceliaDriver.requestDreamState(256);
                dreamStateCache.put(worldName, gradient);
                narrativeSummaryCache.put(worldName, requestNarrativeSummary(worldName));
            }
        }, periodTicks, periodTicks);
    }

    private void startOtocTask() {
        long periodTicks = 20L * 60L * 5L; // alle 5 Minuten
        getServer().getScheduler().runTaskTimerAsynchronously(this, () -> {
            otocChaosFactor = myceliaDriver.requestGlobalOTOC();
        }, periodTicks, periodTicks);
    }

    private void startTimeWarpTick() {
        long tickInterval = 20L; // jede Sekunde
        getServer().getScheduler().runTaskTimer(this, () -> {
            if (timeWarpListener != null) {
                timeWarpListener.tick();
            }
        }, tickInterval, tickInterval);
    }

    public float getDreamInfluence(String worldName, int chunkX, int chunkZ) {
        return dreamStateCache.getInfluence(worldName, chunkX, chunkZ);
    }

    public double getOtocChaosFactor() {
        return otocChaosFactor;
    }

    public void setOtocChaosFactor(double chaos) {
        this.otocChaosFactor = chaos;
    }

    public void addTimeWarpZone(org.bukkit.Chunk chunk) {
        timeWarpZones.add(chunkKey(chunk));
    }

    public boolean isTimeWarpZone(org.bukkit.Chunk chunk) {
        return timeWarpZones.contains(chunkKey(chunk));
    }

    public DreamStateCache getDreamStateCache() {
        return dreamStateCache;
    }

    public NarrativeSummaryCache getNarrativeSummaryCache() {
        return narrativeSummaryCache;
    }

    private String chunkKey(org.bukkit.Chunk chunk) {
        return chunk.getWorld().getName() + ":" + chunk.getX() + ":" + chunk.getZ();
    }

    public MyceliaDriver getDriver() {
        return myceliaDriver;
    }

    private float[] requestNarrativeSummary(String worldName) {
        org.bukkit.World world = Bukkit.getWorld(worldName);
        if (world == null) return new float[0];
        float[] signature = sampleWorldSignature(world, 512);
        return myceliaDriver.getPersistentService()
                .flatMap(service -> service.requestSymbolicAbstraction(32, signature, signature))
                .orElse(signature);
    }

    private float[] sampleWorldSignature(org.bukkit.World world, int count) {
        float[] values = new float[count];
        org.bukkit.util.noise.SimplexNoiseGenerator sampler = new org.bukkit.util.noise.SimplexNoiseGenerator(world.getSeed());
        for (int i = 0; i < count; i++) {
            double x = world.getSpawnLocation().getX() + (i % 32);
            double z = world.getSpawnLocation().getZ() + (i / 32);
            values[i] = (float) sampler.noise(x * 0.05, z * 0.05);
        }
        return values;
    }
}
