import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import sys
import os
import time
import queue

# --- WICHTIG: Import der Backend-Logik ---
# Die Datei 'mycelia_vault_v4.py' muss im selben Ordner liegen!
try:
    from mycelia_vault_v4 import MyceliaVaultV4, HEADER_MAGIC
except ImportError:
    # Falls wir in einer EXE sind, hilft dieser Block bei der Fehlersuche
    try:
        # Versuch, den Fehler in ein Log zu schreiben, da keine Konsole sichtbar ist
        with open("gui_import_error.log", "w") as f:
            f.write(f"Pfad: {sys.path}\nFehler: mycelia_vault_v4 nicht gefunden.")
    except:
        pass
    print("CRITICAL: mycelia_vault_v4.py nicht gefunden!")
    sys.exit(1)

# --- Design Konstanten (Dark Mode) ---
COLOR_BG = "#121212"
COLOR_PANEL = "#1e1e1e"
COLOR_TEXT = "#e0e0e0"
COLOR_ACCENT = "#00ff99" # Bio-Cyberpunk Grün
COLOR_ERROR = "#ff5555"
FONT_MAIN = ("Segoe UI", 10)
FONT_MONO = ("Consolas", 9)

class IORedirector(object):
    """
    Fängt print-Ausgaben (stdout) vom Backend ab 
    und sendet sie sicher an die GUI-Queue.
    """
    def __init__(self, text_queue):
        self.text_queue = text_queue

    def write(self, string):
        self.text_queue.put(string)

    def flush(self):
        pass

class MyceliaVaultGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Mycelia Vault V4 [Enterprise]")
        self.root.geometry("700x550")
        self.root.configure(bg=COLOR_BG)
        # Fenstergröße fixieren für sauberes Layout
        self.root.resizable(False, False)

        # Queue für Thread-sichere Kommunikation
        self.msg_queue = queue.Queue()
        self.vault_instance = None
        self.is_processing = False
        self.total_size_mb = 1 # Default um Division durch Null zu vermeiden

        self._setup_styles()
        self._build_ui()
        
        # Start Routine: Engine laden (verzögert, damit GUI erst sichtbar wird)
        self.log_manual("Systemstart...", "info")
        self.root.after(100, self._init_engine_thread)
        self.root.after(100, self._process_queue)

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure("TFrame", background=COLOR_BG)
        style.configure("Card.TFrame", background=COLOR_PANEL, relief="flat")
        
        style.configure("TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT, font=FONT_MAIN)
        style.configure("Header.TLabel", background=COLOR_BG, foreground=COLOR_ACCENT, font=("Segoe UI", 16, "bold"))
        style.configure("Status.TLabel", background=COLOR_PANEL, foreground="#888", font=("Segoe UI", 9))
        
        style.configure("TButton", 
                        background="#333", 
                        foreground="#fff", 
                        font=("Segoe UI", 10, "bold"), 
                        borderwidth=0, 
                        focuscolor=COLOR_PANEL)
        style.map("TButton", 
                  background=[('active', COLOR_ACCENT), ('disabled', '#2a2a2a')], 
                  foreground=[('active', '#000'), ('disabled', '#555')])
        
        style.configure("Horizontal.TProgressbar", 
                        background=COLOR_ACCENT, 
                        troughcolor="#333", 
                        bordercolor="#333", 
                        lightcolor=COLOR_ACCENT, 
                        darkcolor=COLOR_ACCENT)

    def _build_ui(self):
        # Header Bereich
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill="x", padx=20, pady=15)
        lbl_title = ttk.Label(header_frame, text="MYCELIA VAULT", style="Header.TLabel")
        lbl_title.pack(side="left")
        self.lbl_gpu_status = ttk.Label(header_frame, text="Lade Engine...", background=COLOR_BG, foreground="#666")
        self.lbl_gpu_status.pack(side="right", pady=5)

        # Haupt-Karte
        card = ttk.Frame(self.root, style="Card.TFrame", padding=20)
        card.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # Input
        lbl_in = ttk.Label(card, text="QUELLDATEI:")
        lbl_in.pack(anchor="w")
        frm_in = ttk.Frame(card, style="Card.TFrame")
        frm_in.pack(fill="x", pady=(5, 15))
        
        self.ent_input = tk.Entry(frm_in, bg="#2b2b2b", fg="#fff", insertbackground="#fff", relief="flat", font=FONT_MONO)
        self.ent_input.pack(side="left", fill="x", expand=True, ipady=4, padx=(0, 5))
        btn_browse = ttk.Button(frm_in, text="...", width=4, command=self._browse_input)
        btn_browse.pack(side="right")

        # Output
        lbl_out = ttk.Label(card, text="ZIEL (OPTIONAL):")
        lbl_out.pack(anchor="w")
        frm_out = ttk.Frame(card, style="Card.TFrame")
        frm_out.pack(fill="x", pady=(5, 15))
        
        self.ent_output = tk.Entry(frm_out, bg="#2b2b2b", fg="#fff", insertbackground="#fff", relief="flat", font=FONT_MONO)
        self.ent_output.pack(side="left", fill="x", expand=True, ipady=4, padx=(0, 5))
        btn_browse_out = ttk.Button(frm_out, text="...", width=4, command=self._browse_output)
        btn_browse_out.pack(side="right")

        # Buttons
        btn_frame = ttk.Frame(card, style="Card.TFrame")
        btn_frame.pack(fill="x", pady=10)
        
        self.btn_encrypt = ttk.Button(btn_frame, text="VERSCHLÜSSELN (STREAM)", command=lambda: self._start_task('encrypt'))
        self.btn_encrypt.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        self.btn_decrypt = ttk.Button(btn_frame, text="ENTSCHLÜSSELN (AUTO)", command=lambda: self._start_task('decrypt'))
        self.btn_decrypt.pack(side="right", fill="x", expand=True, padx=(5, 0))

        # Progress & Status
        self.progress = ttk.Progressbar(card, style="Horizontal.TProgressbar", mode='determinate', length=100)
        self.progress.pack(fill="x", pady=(20, 5))
        
        self.lbl_status = ttk.Label(card, text="Warte auf Eingabe...", style="Status.TLabel")
        self.lbl_status.pack(anchor="w")

        # Log Terminal
        log_frame = tk.Frame(card, bg="#000", bd=1, relief="solid")
        log_frame.pack(fill="both", expand=True, pady=(10, 0))
        
        self.txt_log = tk.Text(log_frame, bg="#0c0c0c", fg=COLOR_ACCENT, font=FONT_MONO, state="disabled", bd=0, height=8)
        self.txt_log.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        # Log Farben
        self.txt_log.tag_config("info", foreground="#888")
        self.txt_log.tag_config("success", foreground=COLOR_ACCENT)
        self.txt_log.tag_config("error", foreground=COLOR_ERROR)
        self.txt_log.tag_config("vault", foreground="#ffffff")

    def _browse_input(self):
        path = filedialog.askopenfilename()
        if path:
            self.ent_input.delete(0, tk.END)
            self.ent_input.insert(0, path)

    def _browse_output(self):
        path = filedialog.asksaveasfilename()
        if path:
            self.ent_output.delete(0, tk.END)
            self.ent_output.insert(0, path)

    def log_manual(self, message, tag="vault"):
        """Schreibt manuell in das GUI-Log"""
        self.txt_log.config(state="normal")
        self.txt_log.insert(tk.END, f"> {message}\n", tag)
        self.txt_log.see(tk.END)
        self.txt_log.config(state="disabled")

    def _process_queue(self):
        """Liest Ausgaben vom Worker-Thread und aktualisiert die UI"""
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                
                # Check ob es eine Fortschrittsmeldung ist
                if "\r" in msg or msg.startswith("[Vault] Verarbeite:") or msg.startswith("[Vault] Entschlüssele:"):
                    clean_msg = msg.replace("\r", "").strip()
                    self.lbl_status.config(text=clean_msg)
                    
                    # Versuch, MB Zahl zu parsen für den Balken
                    try:
                        parts = clean_msg.split(" ")
                        for part in parts:
                            if "." in part and part.replace(".", "").isdigit():
                                current_mb = float(part)
                                if self.total_size_mb > 0:
                                    perc = (current_mb / self.total_size_mb) * 100
                                    self.progress['value'] = perc
                    except:
                        pass
                else:
                    # Normale Log-Nachricht
                    clean_msg = msg.strip()
                    if clean_msg:
                        tag = "vault"
                        if "FEHLER" in clean_msg or "CRITICAL" in clean_msg or "WARNUNG" in clean_msg:
                            tag = "error"
                        elif "Erfolg" in clean_msg or "BESTÄTIGT" in clean_msg:
                            tag = "success"
                        
                        self.log_manual(clean_msg, tag)
                        
        except queue.Empty:
            pass
        
        # Loop alle 50ms
        self.root.after(50, self._process_queue)

    def _init_engine_thread(self):
        """Initialisiert die OpenCL Engine im Hintergrund"""
        def run():
            try:
                self.vault_instance = MyceliaVaultV4()
                count = len(self.vault_instance.gpus)
                
                self.root.after(0, lambda: self.lbl_gpu_status.config(
                    text=f"ENGINE AKTIV", foreground=COLOR_ACCENT))
                self.msg_queue.put(f"[System] OpenCL Engine ({count} GPU) bereit.")
            except Exception as e:
                self.msg_queue.put(f"[System] FEHLER bei GPU Init: {e}")
                self.root.after(0, lambda: self.lbl_gpu_status.config(
                    text="GPU FEHLER", foreground=COLOR_ERROR))

        threading.Thread(target=run, daemon=True).start()

    def _toggle_controls(self, enable):
        state = "normal" if enable else "disabled"
        self.btn_encrypt.config(state=state)
        self.btn_decrypt.config(state=state)
        self.ent_input.config(state=state)
        
        if enable:
            self.progress['value'] = 0
            self.lbl_status.config(text="Bereit.")
            self.is_processing = False
        else:
            self.is_processing = True

    def _start_task(self, mode):
        if self.is_processing: return
        if not self.vault_instance:
            messagebox.showerror("Fehler", "Engine noch nicht bereit.")
            return

        inp = self.ent_input.get()
        outp = self.ent_output.get()

        if not inp or not os.path.exists(inp):
            messagebox.showerror("Fehler", "Eingabedatei existiert nicht.")
            return

        # Dateigröße für Progressbar merken
        try:
            size_bytes = os.path.getsize(inp)
            self.total_size_mb = size_bytes / (1024 * 1024)
        except:
            self.total_size_mb = 1

        self._toggle_controls(False)
        threading.Thread(target=self._worker, args=(mode, inp, outp), daemon=True).start()

    def _worker(self, mode, inp, outp):
        # Stdout umleiten, damit print() aus V4 in der GUI landet
        original_stdout = sys.stdout
        sys.stdout = IORedirector(self.msg_queue)
        
        try:
            start_t = time.time()
            if mode == 'encrypt':
                if not outp:
                    outp = inp + ".box"
                self.msg_queue.put(f"[Task] Starte Verschlüsselung -> {os.path.basename(outp)}")
                self.vault_instance.encrypt(inp, outp)
                
            elif mode == 'decrypt':
                target_dir = os.path.dirname(inp)
                if outp:
                    if os.path.isdir(outp):
                        target_dir = outp
                    else:
                        # Fallback: Ordner der Zieldatei nutzen
                        target_dir = os.path.dirname(outp)
                
                self.msg_queue.put(f"[Task] Starte Entschlüsselung...")
                self.vault_instance.decrypt(inp, target_dir)

            duration = time.time() - start_t
            self.msg_queue.put(f"[Success] Vorgang beendet in {duration:.2f}s")
            self.root.after(0, lambda: self.progress.configure(value=100))

        except Exception as e:
            self.msg_queue.put(f"CRITICAL ERROR: {str(e)}")
            # Optional: Traceback in die Konsole (für Debug)
            import traceback
            traceback.print_exc(file=original_stdout) 
        finally:
            sys.stdout = original_stdout
            self.root.after(0, lambda: self._toggle_controls(True))

if __name__ == "__main__":
    root = tk.Tk()
    # Icon laden falls vorhanden
    # try: root.iconbitmap("icon.ico")
    # except: pass
    app = MyceliaVaultGUI(root)
    root.mainloop()