package com.mycelia.mc.api;

import com.mycelia.mc.driver.MyceliaWorldDNA;
import com.mycelia.mc.generation.MyceliaBiomeProfile;
import org.bukkit.World;

import java.util.List;
import java.util.function.Consumer;

public interface MyceliaAPI {
    MyceliaWorldDNA getWorldDNA(World world);

    double requestNoise(int x, int z);

    void registerBiomeModifier(Consumer<List<MyceliaBiomeProfile>> modifier);
}
