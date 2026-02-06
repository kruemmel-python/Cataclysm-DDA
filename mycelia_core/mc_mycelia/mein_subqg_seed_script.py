#!/usr/bin/env python3
"""
mein_subqg_seed_script.py - Persistent Mycelia Driver

Dieses Script unterstützt zwei Modi:
1. One-Shot: Wird mit Argumenten (z.B. --seed) aufgerufen, gibt JSON aus und endet.
2. Persistent: Ohne Argumente gestartet, wartet es auf STDIN auf JSON-Befehle.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import pathlib
import random
import secrets
import struct
import sys
import time
from typing import Final, Any

RESULT_SUCCESS: Final[int] = 0
DLL_NAME: Final[str] = "CC_OpenCl.dll"

# --- Globale Palette ---
PALETTES = [
    {"name": "Myzel-Invasion", "base": "DEEPSLATE", "surface": "MYCELIUM", "ore": "AMETHYST_BLOCK", "scale": 0.02},
    {"name": "Eis-Oede", "base": "PACKED_ICE", "surface": "SNOW_BLOCK", "ore": "BLUE_ICE", "scale": 0.035},
    {"name": "Vulkanisch", "base": "BLACKSTONE", "surface": "BASALT", "ore": "MAGMA_BLOCK", "scale": 0.015},
    {"name": "Verdorbener Wald", "base": "NETHERRACK", "surface": "WARPED_NYLIUM", "ore": "NETHER_WART_BLOCK", "scale": 0.025},
    {"name": "Ueberwuchert", "base": "MOSSY_COBBLESTONE", "surface": "MOSS_BLOCK", "ore": "RAW_GOLD_BLOCK", "scale": 0.03},
]

# --- Hilfsfunktionen für Datentypen ---
def _u64(n: int) -> int:
    return n & 0xFFFFFFFFFFFFFFFF

def _i64_from_u64(n: int) -> int:
    raw = _u64(n)
    return struct.unpack("<q", struct.pack("<Q", raw))[0]

def _prepare_windows_dll_search(dll_dir: pathlib.Path) -> None:
    if os.name != "nt":
        return
    try:
        os.add_dll_directory(str(dll_dir))
    except Exception:
        pass
    os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")

def _load_library(script_dir: pathlib.Path) -> ctypes.CDLL:
    dll_dir = (script_dir / "bin").resolve()
    candidate = (dll_dir / DLL_NAME).resolve()
    if not candidate.exists():
        # Fallback auf CWD/bin
        candidate = (pathlib.Path.cwd() / "bin" / DLL_NAME).resolve()
        
    if not candidate.exists():
        raise FileNotFoundError(f"DLL nicht gefunden: {candidate}")

    if os.name == "nt":
        _prepare_windows_dll_search(candidate.parent)
        return ctypes.WinDLL(str(candidate))
    return ctypes.CDLL(str(candidate))

# --- DLL Binding ---
class MyceliaCore:
    def __init__(self, lib: ctypes.CDLL):
        self.lib = lib
        self._bind()

    def _bind(self):
        self.lib.myc_init.restype = ctypes.c_int
        self.lib.myc_create_context.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.myc_create_context.restype = ctypes.c_int
        self.lib.myc_set_seed.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self.lib.myc_set_seed.restype = ctypes.c_int
        self.lib.myc_process_buffer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t, ctypes.c_size_t]
        self.lib.myc_process_buffer.restype = ctypes.c_int
        self.lib.myc_destroy_context.argtypes = [ctypes.c_void_p]
        self.lib.myc_get_last_error.restype = ctypes.c_char_p

    def get_last_error(self) -> str:
        msg = self.lib.myc_get_last_error()
        return msg.decode("utf-8", errors="replace") if msg else "Unknown Error"

# --- Treiber Logik ---
class DriverInstance:
    def __init__(self, gpu_index: int):
        script_dir = pathlib.Path(__file__).resolve().parent
        self.lib_raw = _load_library(script_dir)
        self.core = MyceliaCore(self.lib_raw)
        self.gpu_index = gpu_index
        self.ctx = ctypes.c_void_p()
        
        if self.core.lib.myc_init() != RESULT_SUCCESS:
            raise RuntimeError(f"GPU Init fehlgeschlagen: {self.core.get_last_error()}")
            
        if self.core.lib.myc_create_context(self.gpu_index, ctypes.byref(self.ctx)) != RESULT_SUCCESS:
            raise RuntimeError(f"GPU Context fehlgeschlagen: {self.core.get_last_error()}")

    def generate_seed(self, base_seed: int) -> int:
        self.core.lib.myc_set_seed(self.ctx, ctypes.c_uint64(_u64(base_seed)))
        buf = (ctypes.c_uint8 * 8)()
        if self.core.lib.myc_process_buffer(self.ctx, buf, 8, 0) != RESULT_SUCCESS:
            sys.stderr.write(f"[Mycelia] Buffer-Fehler: {self.core.get_last_error()}\n")
            return base_seed ^ secrets.randbits(64)
        return struct.unpack("<Q", bytes(buf))[0]

    def shutdown(self):
        if self.ctx:
            self.core.lib.myc_destroy_context(self.ctx)

# --- Command Handler ---
def handle_world_cmd(driver: DriverInstance, base_seed: int, unsigned: bool) -> dict:
    final_seed = driver.generate_seed(base_seed)
    random.seed(final_seed)
    theme = random.choice(PALETTES)
    
    return {
        "seed": final_seed if unsigned else _i64_from_u64(final_seed),
        "baseBlock": theme["base"],
        "surfaceBlock": theme["surface"],
        "oreBlock": theme["ore"],
        "scale": theme["scale"],
        "seaLevel": 62,
        "fallback": False
    }

def run_persistent(gpu_index: int, unsigned: bool):
    sys.stderr.write(f"[Mycelia] Starte Persistent Mode auf GPU {gpu_index}...\n")
    try:
        driver = DriverInstance(gpu_index)
        sys.stderr.write("[Mycelia] Treiber bereit. Warte auf STDIN...\n")
        sys.stderr.flush()

        while True:
            line = sys.stdin.readline()
            if not line:
                break
            
            try:
                data = json.loads(line)
                cmd = data.get("cmd")
                
                if cmd == "health":
                    print("ok", flush=True)
                
                elif cmd == "world":
                    base = data.get("seed", secrets.randbits(64))
                    result = handle_world_cmd(driver, base, unsigned)
                    print(json.dumps(result), flush=True)
                
                elif cmd == "otoc_chaos":
                    # Simulierter Wert basierend auf GPU Drift
                    print(json.dumps([0.1 + (random.random() * 0.4)]), flush=True)
                
                elif cmd == "dream_state":
                    size = data.get("size", 256)
                    # Erzeugt ein künstliches Gradienten-Array
                    gradient = [random.random() for _ in range(size)]
                    print(json.dumps(gradient), flush=True)
                
                elif cmd == "symbolic_abstract":
                    # Gibt zwei Concept-Werte zurück (Archetype, Energy)
                    concepts = [random.uniform(0, 2.0), random.uniform(0, 1.0)]
                    print(json.dumps(concepts), flush=True)
                
                elif cmd == "noise":
                    # Einfaches deterministisches Rauschen
                    x, z = data.get("x", 0), data.get("z", 0)
                    random.seed(x ^ z)
                    print(str(random.random()), flush=True)

            except json.JSONDecodeError:
                sys.stderr.write("[Mycelia] Ungültiges JSON empfangen.\n")
            except Exception as e:
                sys.stderr.write(f"[Mycelia] Fehler bei Befehlsverarbeitung: {e}\n")

    except Exception as e:
        sys.stderr.write(f"[Mycelia] Fataler Treiberfehler: {e}\n")
    finally:
        sys.stderr.write("[Mycelia] Beende persistenten Treiber.\n")

def run_oneshot(args):
    try:
        driver = DriverInstance(args.gpu)
        base = args.seed if args.seed is not None else secrets.randbits(64)
        result = handle_world_cmd(driver, base, args.unsigned)
        print(json.dumps(result), flush=True)
        driver.shutdown()
    except Exception as e:
        # Fallback im One-Shot Modus
        sys.stderr.write(f"[Mycelia] One-Shot fehlgeschlagen: {e}. Nutze Python-Fallback.\n")
        theme = random.choice(PALETTES)
        fallback_seed = args.seed if args.seed is not None else secrets.randbits(64)
        result = {
            "seed": fallback_seed if args.unsigned else _i64_from_u64(fallback_seed),
            "baseBlock": theme["base"],
            "surfaceBlock": theme["surface"],
            "oreBlock": theme["ore"],
            "scale": theme["scale"],
            "seaLevel": 62,
            "fallback": True
        }
        print(json.dumps(result), flush=True)

# --- Main Entry ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, help="Basis-Seed (int)")
    ap.add_argument("--gpu", type=int, default=0, help="GPU-Index")
    ap.add_argument("--unsigned", action="store_true", help="Unsigned Output")
    ap.add_argument("--persistent", action="store_true", help="Erzwinge Persistent Mode")
    ap.add_argument("--oneshot", action="store_true", help="Erzwinge One-Shot Mode (überschreibt persistent)")

    args = ap.parse_args()

    # Regelwerk:
    # 1) --oneshot erzwingt One-Shot
    # 2) --persistent erzwingt persistent
    # 3) Wenn ein Seed explizit übergeben wurde -> One-Shot (klassischer CLI-Aufruf)
    # 4) Sonst -> persistent (Plugin-Standardfall: startet mit --gpu/--unsigned etc.)
    if args.oneshot:
        run_oneshot(args)
        return

    if args.persistent:
        run_persistent(args.gpu, args.unsigned)
        return

    if args.seed is not None:
        run_oneshot(args)
        return

    run_persistent(args.gpu, args.unsigned)


if __name__ == "__main__":
    main()