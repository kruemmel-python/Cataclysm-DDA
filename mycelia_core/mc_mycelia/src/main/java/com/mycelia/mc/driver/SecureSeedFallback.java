package com.mycelia.mc.driver;

import java.security.SecureRandom;

class SecureSeedFallback {
    private final SecureRandom secureRandom = new SecureRandom();

    long nextSeed() {
        return secureRandom.nextLong();
    }
}
