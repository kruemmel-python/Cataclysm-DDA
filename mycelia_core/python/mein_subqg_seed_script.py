#!/usr/bin/env python3
"""
mein_subqg_seed_script.py (hardened)

Garantiert:
- Gibt innerhalb von --timeout Sekunden einen Seed auf STDOUT aus (auch Fallback).
- Nutzt CC_OpenCl.dll relativ zu ../bin.
- Loggt Fehler kurz auf STDERR.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import pathlib
import secrets
import struct
import sys
from multiprocessing import Process, Queue
from typing import Final

RESULT_SUCCESS: Final[int] = 0


def _load_library(script_dir: pathlib.Path) -> ctypes.CDLL:
    dll_name = "CC_OpenCl.dll"
    candidate = (script_dir.parent / "bin" / dll_name).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"DLL nicht gefunden: {candidate}")
    return ctypes.WinDLL(str(candidate)) if os.name == "nt" else ctypes.CDLL(str(candidate))


def _bind_functions(lib: ctypes.CDLL) -> None:
    lib.myc_init.restype = ctypes.c_int
    lib.myc_get_device_count.restype = ctypes.c_int

    lib.myc_create_context.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
    lib.myc_create_context.restype = ctypes.c_int

    lib.myc_set_seed.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.myc_set_seed.restype = ctypes.c_int

    lib.myc_process_buffer.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.myc_process_buffer.restype = ctypes.c_int

    lib.myc_destroy_context.argtypes = [ctypes.c_void_p]
    lib.myc_destroy_context.restype = None

    lib.myc_get_last_error.restype = ctypes.c_char_p


def _last_error(lib: ctypes.CDLL) -> str:
    try:
        msg = lib.myc_get_last_error()
        return msg.decode("utf-8", errors="replace") if msg else ""
    except Exception:
        return ""


def _u64(n: int) -> int:
    return n & 0xFFFFFFFFFFFFFFFF


def _i64_from_u64(n: int) -> int:
    raw = _u64(n)
    return struct.unpack("<q", struct.pack("<Q", raw))[0]


def _generate_seed_inner(dll_path_dir: pathlib.Path, gpu_index: int, base_seed: int, unsigned: bool, q: Queue) -> None:
    """
    Läuft in separatem Prozess, damit wir harte Timeouts erzwingen können.
    """
    try:
        lib = _load_library(dll_path_dir)
        _bind_functions(lib)

        if lib.myc_init() != RESULT_SUCCESS:
            raise RuntimeError(f"myc_init fehlgeschlagen: {_last_error(lib)}")

        device_count = lib.myc_get_device_count()
        if device_count <= 0:
            raise RuntimeError("Keine GPUs verfügbar")

        ctx = ctypes.c_void_p()
        rc = lib.myc_create_context(gpu_index, ctypes.byref(ctx))
        if rc != RESULT_SUCCESS or not ctx:
            raise RuntimeError(f"myc_create_context fehlgeschlagen (Code {rc}): {_last_error(lib)}")

        try:
            rc = lib.myc_set_seed(ctx, ctypes.c_uint64(_u64(base_seed)))
            if rc != RESULT_SUCCESS:
                raise RuntimeError(f"myc_set_seed fehlgeschlagen (Code {rc}): {_last_error(lib)}")

            buf = (ctypes.c_uint8 * 8)()
            rc = lib.myc_process_buffer(ctx, buf, ctypes.c_size_t(8), ctypes.c_size_t(0))
            if rc != RESULT_SUCCESS:
                raise RuntimeError(f"myc_process_buffer fehlgeschlagen (Code {rc}): {_last_error(lib)}")

            raw = bytes(buf)
            if unsigned:
                seed = struct.unpack("<Q", raw)[0]
            else:
                seed = struct.unpack("<q", raw)[0]

            q.put(("ok", int(seed)))
        finally:
            lib.myc_destroy_context(ctx)

    except Exception as exc:
        q.put(("err", str(exc)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, help="Basis-Seed (64-bit)")
    ap.add_argument("--gpu", type=int, default=0, help="GPU-Index (Default: 0)")
    ap.add_argument("--timeout", type=float, default=2.0, help="Max. Sekunden für Treiber/Seed (Default: 2.0)")
    ap.add_argument("--unsigned", action="store_true", help="Seed als unsigned 64-bit ausgeben (Default: signed)")
    args = ap.parse_args()

    script_dir = pathlib.Path(__file__).resolve().parent
    base_seed = args.seed if args.seed is not None else secrets.randbits(64)

    q: Queue = Queue()
    p = Process(
        target=_generate_seed_inner,
        args=(script_dir, args.gpu, base_seed, args.unsigned, q),
        daemon=True,
    )
    p.start()
    p.join(timeout=args.timeout)

    if p.is_alive():
        # Hard timeout: kill process and fallback
        p.kill()
        p.join(timeout=0.2)
        sys.stderr.write(f"[mein_subqg_seed_script] timeout>{args.timeout}s -> fallback\n")
        fallback = secrets.randbits(64)
        print(fallback if args.unsigned else _i64_from_u64(fallback))
        return 0

    if not q.empty():
        status, payload = q.get()
        if status == "ok":
            print(payload)
            return 0
        sys.stderr.write(f"[mein_subqg_seed_script] driver_error -> fallback: {payload}\n")

    fallback = secrets.randbits(64)
    print(fallback if args.unsigned else _i64_from_u64(fallback))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
