import ctypes
import struct
import os
import random
import sys
import zlib
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# --- Funktion zum Finden von Ressourcen in der EXE ---
def resource_path(relative_path):
    """ Ermittelt den absoluten Pfad, egal ob Skript oder EXE """
    try:
        # PyInstaller speichert den Pfad im _MEIPASS Attribut
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# --- C-Treiber Setup ---
DLL_NAME = "./bin/CC_OpenCl.dll" if os.name == 'nt' else "./bin/CC_OpenCl.so"

# 1. Versuch: Pfad in der EXE (oder lokal root)
LIB_PATH = resource_path(DLL_NAME)

# 2. Fallback: Entwicklungsumgebung (build Ordner)
if not os.path.exists(LIB_PATH):
    if os.path.exists(os.path.join("build", DLL_NAME)):
        LIB_PATH = os.path.join("build", DLL_NAME)
    elif os.path.exists(DLL_NAME): # Direkt im Ordner ohne resource_path logik
        LIB_PATH = os.path.abspath(DLL_NAME)

try:
    # Wir nutzen LoadLibrary explizit für Windows, um Pfadprobleme zu minimieren
    if os.name == 'nt':
        cl = ctypes.CDLL(LIB_PATH, winmode=0) 
    else:
        cl = ctypes.CDLL(LIB_PATH)
except OSError as e:
    # Wir geben den Fehler aus, aber in einer GUI ohne Konsole sieht man das nicht.
    # Daher schreiben wir es in eine Fehlerdatei, falls es crasht.
    with open("mycelia_error.log", "w") as f:
        f.write(f"CRITICAL: Konnte {LIB_PATH} nicht laden.\nFehler: {e}\nSuchpfade: {sys.path}")
    sys.exit(1)

# Strukturen & Signaturen
class HPIOAgent(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("energy", ctypes.c_float), ("coupling", ctypes.c_float)]

cl.initialize_gpu.argtypes = [ctypes.c_int]
cl.subqg_set_deterministic_mode.argtypes = [ctypes.c_int, ctypes.c_ulonglong]
cl.subqg_initialize_state.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
cl.subqg_inject_agents.argtypes = [ctypes.c_int, ctypes.POINTER(HPIOAgent), ctypes.c_int]
cl.subqg_simulation_step.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
cl.subqg_debug_read_channel.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
# cl.cc_get_last_error gibt es evtl nicht in jedem build, daher optional:
try:
    cl.cc_get_last_error.restype = ctypes.c_char_p
except:
    pass

# --- Konstanten V4 ---
HEADER_MAGIC = b'MYZ4' 
VERSION = 4
GRID_SIZE = 256 * 256 # 65536 Zellen
CHUNK_SIZE = GRID_SIZE # 1:1 Mapping: 1 Feld-Zustand verschlüsselt 64KB Daten
QUEUE_SIZE = 4 # Kleinerer Puffer für mehr Stabilität

# GLOBAL LOCK für C-Zugriffe (Verhindert Race Conditions im VRAM Treiber)
C_LOCK = threading.Lock()

class GPUSlot:
    """Verwaltet eine einzelne GPU Instanz."""
    def __init__(self, index):
        self.index = index
        print(f"[System] Initialisiere GPU {index}...")
        res = cl.initialize_gpu(index)

    def generate_key_block(self, seed, block_index):
        """
        Erzeugt einen Key-Block. 
        WICHTIG: Durch C_LOCK abgesichert, damit globale C-Variablen nicht korrupt werden.
        """
        import numpy as np
        
        with C_LOCK:
            # 1. Deterministischer Seed für diesen Block
            # Wir nutzen einen simplen linearen Offset für Stabilität
            block_seed = seed + (block_index * 7919) 
            
            cl.subqg_set_deterministic_mode(1, ctypes.c_ulonglong(block_seed))
            
            # 2. Reset Physics
            cl.subqg_initialize_state(self.index, 0.5, 0.5, 0.005, 0.5)
            
            # 3. Evolution
            cl.subqg_simulation_step(self.index, 0.5, 0.5, 0.5, None, None, None, None, None, None, None, 0)
            
            # 4. Readback
            raw_buffer = np.zeros(GRID_SIZE, dtype=np.float32)
            cl.subqg_debug_read_channel(
                self.index, 0, 
                raw_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 
                GRID_SIZE
            )
            
            # 5. Hashing (Float -> Byte)
            key_int = raw_buffer.view(np.uint32)
            key_int = (key_int ^ (key_int >> 16)) * 0x45d9f3b
            key_bytes = (key_int & 0xFF).astype(np.uint8)
            
            return key_bytes

class MyceliaVaultV4:
    def __init__(self):
        self.gpus = []
        self._detect_gpus()
        
    def _detect_gpus(self):
        # FIX: Wir nutzen strikt NUR GPU 0 um Race Conditions zu vermeiden
        # Multi-Threading im Python-Teil ist okay, aber der C-State muss atomar sein.
        self.gpus.append(GPUSlot(0))
        print(f"[System] Mycelia Engine aktiv (High-Precision Mode).")

    def _process_chunk_task(self, gpu_slot, data_chunk, master_seed, block_index):
        """Worker Funktion für ThreadPool"""
        key_bytes = gpu_slot.generate_key_block(master_seed, block_index)
        
        # Key auf Datenlänge zuschneiden
        import numpy as np
        n = len(data_chunk)
        current_key = key_bytes[:n]
        
        # Numpy XOR
        np_data = np.frombuffer(data_chunk, dtype=np.uint8)
        xor_result = np.bitwise_xor(np_data, current_key)
        
        return xor_result.tobytes()

    def process_stream(self, input_path, output_path, master_seed, mode='encrypt'):
        file_size = os.path.getsize(input_path)
        
        # Integritäts-Check (Keyed Hash)
        hasher = hashlib.blake2b(key=struct.pack("Q", master_seed)[:32]) 
        
        # Wir nutzen 2 Worker (I/O und Compute parallel), aber C-Calls sind serialized via C_LOCK
        executor = ThreadPoolExecutor(max_workers=2)
        
        futures = {} 
        block_index = 0
        
        with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
            # Header schreiben (nur bei Encrypt)
            if mode == 'encrypt':
                fn = os.path.basename(input_path).encode('utf-8')
                # Container Format V4: [Magic 4][Ver 4][Seed 8][FnLen 2][Fn Bytes...][Content...]
                header = HEADER_MAGIC + struct.pack('I', VERSION) + struct.pack('Q', master_seed)
                header += struct.pack('H', len(fn)) + fn
                fout.write(header)
                hasher.update(header)
            
            # Reader Loop
            while True:
                # Puffer füllen
                while len(futures) < QUEUE_SIZE:
                    chunk = fin.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    
                    gpu = self.gpus[0] # Immer GPU 0
                    
                    # Submit task
                    ft = executor.submit(self._process_chunk_task, gpu, chunk, master_seed, block_index)
                    futures[block_index] = ft
                    block_index += 1
                
                if not futures:
                    break
                
                # In Reihenfolge schreiben
                next_write_idx = min(futures.keys())
                result_chunk = futures[next_write_idx].result()
                del futures[next_write_idx]
                
                # Schreiben & Hashen
                fout.write(result_chunk)
                hasher.update(result_chunk)
                
                # UI Progress
                if next_write_idx % 10 == 0:
                    processed_mb = (next_write_idx * CHUNK_SIZE) / 1024 / 1024
                    sys.stdout.write(f"\r[Vault] Verarbeite: {processed_mb:.1f} MB ...")
                    sys.stdout.flush()

            sys.stdout.write("\n")
            
            if mode == 'encrypt':
                # Tag anhängen
                tag = hasher.digest()
                fout.write(tag)
                print(f"[Vault] Integrity Tag generiert: {tag.hex()[:16]}...")
            
    def encrypt(self, input_file, output_file):
        seed = random.randint(0, 2**64 - 1)
        print(f"[Vault] Neuer Master-Seed: {seed}")
        self.process_stream(input_file, output_file, seed, 'encrypt')

    def decrypt(self, input_file, output_folder="."):
        # Header lesen um Seed zu bekommen
        with open(input_file, 'rb') as f:
            magic = f.read(4)
            if magic != HEADER_MAGIC:
                print("FEHLER: Kein Mycelia V4 Format.")
                return
            version = struct.unpack('I', f.read(4))[0]
            seed = struct.unpack('Q', f.read(8))[0]
            fn_len = struct.unpack('H', f.read(2))[0]
            fn_bytes = f.read(fn_len)
            original_filename = fn_bytes.decode('utf-8')
            
            data_start = f.tell()
            
            # Dateigröße für Tag-Handling
            f.seek(0, 2)
            total_size = f.tell()
            tag_size = 64 # Blake2b digest size
            payload_size = total_size - data_start - tag_size
            
            if payload_size < 0:
                print("FEHLER: Datei zu kurz oder beschädigt.")
                return

        print(f"[Vault] Erkannt: '{original_filename}' (Seed {seed})")
        
        # Zielpfad
        out_path = os.path.join(output_folder, original_filename)
        
        # Exaktes Decrypten der Payload
        self._decrypt_stream_bounded(input_file, out_path, seed, data_start, payload_size, tag_size)

    def _decrypt_stream_bounded(self, input_path, output_path, master_seed, start_offset, payload_len, tag_size):
        executor = ThreadPoolExecutor(max_workers=2)
        futures = {}
        block_index = 0
        bytes_processed = 0
        
        hasher = hashlib.blake2b(key=struct.pack("Q", master_seed)[:32])
        
        with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
            # Header hashen (für Validierung)
            fin.seek(0)
            header_data = fin.read(start_offset)
            hasher.update(header_data)
            
            # Payload Loop
            while bytes_processed < payload_len:
                while len(futures) < QUEUE_SIZE:
                    # Berechne wie viel wir noch lesen dürfen (stoppt VOR dem Tag)
                    remaining = payload_len - bytes_processed - (len(futures) * CHUNK_SIZE) 
                    # Korrektur: bytes_processed ist das was schon geschrieben wurde.
                    # Wir müssen wissen, wieviel schon in futures steckt.
                    # Einfacher: wir schauen auf fin position.
                    
                    pos = fin.tell()
                    remaining_payload = (start_offset + payload_len) - pos
                    
                    if remaining_payload <= 0:
                        break
                    
                    to_read = min(CHUNK_SIZE, remaining_payload)
                    chunk = fin.read(to_read)
                    if not chunk: break
                    
                    # Hash update mit Ciphertext
                    hasher.update(chunk)
                    
                    gpu = self.gpus[0]
                    ft = executor.submit(self._process_chunk_task, gpu, chunk, master_seed, block_index)
                    futures[block_index] = ft
                    block_index += 1
                
                if not futures: break
                
                next_idx = min(futures.keys())
                decrypted_chunk = futures[next_idx].result()
                del futures[next_idx]
                
                fout.write(decrypted_chunk)
                bytes_processed += len(decrypted_chunk)
                
                if next_idx % 10 == 0:
                    sys.stdout.write(f"\r[Vault] Entschlüssele: {bytes_processed/1024/1024:.1f} MB")

            # Tag prüfen
            print("\n[Vault] Prüfe Integrität...")
            fin.seek(start_offset + payload_len)
            file_tag = fin.read(tag_size)
            calc_tag = hasher.digest()
            
            if file_tag == calc_tag:
                print("✅ INTEGRITÄT BESTÄTIGT. Datei ist authentisch.")
            else:
                print("❌ WARNUNG: INTEGRITÄTSFEHLER!")
                print("   Die Datei wurde manipuliert oder die Physik-Engine war nicht deterministisch.")
                fout.close()
                os.rename(output_path, output_path + ".CORRUPT")
                print(f"   Datei umbenannt in: {os.path.basename(output_path)}.CORRUPT")

if __name__ == "__main__":
    vault = MyceliaVaultV4()
    
    if len(sys.argv) < 3:
        print("Usage: python mycelia_vault_v4.py [encrypt|decrypt] [input] [output]")
        sys.exit()
        
    mode = sys.argv[1]
    inp = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "."
    
    if mode == 'encrypt':
        vault.encrypt(inp, out)
    elif mode == 'decrypt':
        vault.decrypt(inp, out)