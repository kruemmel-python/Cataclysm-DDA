package com.mycelia.mc.api;

import com.mycelia.mc.driver.MyceliaWorldDNA;
import com.mycelia.mc.generation.MyceliaBiomeProfile;
import org.bukkit.World;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Consumer;

public class Mycelia implements MyceliaAPI {

    private static Mycelia instance;

    public static void install(Mycelia backend) {
        instance = backend;
    }

    public static MyceliaAPI get() {
        return instance;
    }

    private final NoiseSampler sampler;
    private final WorldMetadataAdapter metadataAdapter;
    private final List<Consumer<List<MyceliaBiomeProfile>>> biomeModifiers = Collections.synchronizedList(new ArrayList<>());

    public Mycelia(NoiseSampler sampler, WorldMetadataAdapter metadataAdapter) {
        this.sampler = sampler;
        this.metadataAdapter = metadataAdapter;
    }

    @Override
    public MyceliaWorldDNA getWorldDNA(World world) {
        return metadataAdapter.resolveDNA(world.getName()).orElse(null);
    }

    @Override
    public double requestNoise(int x, int z) {
        return sampler.sample(x, z);
    }

    @Override
    public void registerBiomeModifier(Consumer<List<MyceliaBiomeProfile>> modifier) {
        Objects.requireNonNull(modifier, "modifier");
        biomeModifiers.add(modifier);
    }

    public List<Consumer<List<MyceliaBiomeProfile>>> getBiomeModifiers() {
        return List.copyOf(biomeModifiers);
    }

    @FunctionalInterface
    public interface NoiseSampler {
        double sample(int x, int z);
    }

    @FunctionalInterface
    public interface WorldMetadataAdapter {
        Optional<MyceliaWorldDNA> resolveDNA(String worldName);
    }
}
