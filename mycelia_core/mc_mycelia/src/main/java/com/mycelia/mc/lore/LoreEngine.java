package com.mycelia.mc.lore;

import java.util.Arrays;
import java.util.Comparator;

public class LoreEngine {

    public String generateLore(float avgArchetypeAxis, float avgPatternEnergy) {
        String arch = resolveSymbol(avgArchetypeAxis, LoreSymbol.DRIVE_INIT, LoreSymbol.EQUILIBRIUM, LoreSymbol.TENSION, LoreSymbol.CURIOSITY).getDescription();
        String energy = resolveSymbol(avgPatternEnergy, LoreSymbol.WEAK, LoreSymbol.STRONG).getDescription();
        return "Wir sahen " + arch + " und es war wie " + energy + ".";
    }

    private LoreSymbol resolveSymbol(float value, LoreSymbol... symbols) {
        return Arrays.stream(symbols)
                .filter(sym -> sym.matches(value))
                .findFirst()
                .orElse(Arrays.stream(symbols).max(Comparator.comparingDouble(s -> s.ordinal())).orElse(LoreSymbol.WEAK));
    }
}
