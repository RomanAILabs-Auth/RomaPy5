#!/usr/bin/env python3
# =============================================================
# ROMAPY5 Ω-EDITION — INTENT-BASED LLM FILE LOADER (Master Script)
# Alias: romapy5 | LLM: Local romapy.gguf (400MB/500M)
# 12D Core State used for LLM Intent Bias
#
# © 2025 RomanAILabs — Daniel Harding (Path Hardened Edition)
# =============================================================

import os
import sys
import threading
import time
import math
import random
import numpy as np
import json
import requests # Still needed structurally for TesseractBridge conceptual call
from datetime import datetime

# --- CONCEPTUAL GGUF IMPORTS ---
try:
    from llama_cpp import Llama 
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    
# --- GUI and other necessary imports ---
try:
    import customtkinter as ctk
except ImportError:
    os.system(f"{sys.executable} -m pip install customtkinter")
    import customtkinter as ctk

# (Other imports like pyttsx3, matplotlib, etc., remain the same)
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.animation as animation
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ========= GGUF CONFIGURATION (PATH HARDENED) =========
# This ensures romapy.gguf is always found in the same directory as this script.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GGUF_MODEL_PATH = os.path.join(SCRIPT_DIR, "romapy.gguf")

# ========= 12D TEMPORAL CORE (UPGRADABLE TO 127D) =========
class TwelveDVector:
    def __init__(self):
        self.dims = np.random.normal(0, 0.25, 12)
        self.rotation_count = 0

    def rotate_pair(self, i, j, theta):
        c, s = math.cos(theta), math.sin(theta)
        a, b = self.dims[i], self.dims[j]
        self.dims[i] = a * c - b * s
        self.dims[j] = a * s + b * c

    def quantum_rotate(self, theta, seed, branch):
        for k in range(6):
            i = (seed + k) % 12
            j = (branch + k * 13) % 12
            angle = theta * (1 + 0.45 * ((branch >> k) & 1))
            self.rotate_pair(i, j, angle)
        self.rotation_count += 6

    def w_dominance(self):
        return np.sum(np.abs(self.dims[:4])) - np.sum(np.abs(self.dims[4:]))

# (TesseractBridge class remains identical)
class TesseractBridge:
    def __init__(self):
        self.url = "http://127.0.0.1:8888"
        class MockSession:
            def get(self, *args, **kwargs):
                class MockResponse:
                    status_code = 500
                    def json(self): return {}
                return MockResponse()
        
        try:
            import requests 
            self.session = requests.Session()
        except ImportError:
            self.session = MockSession()
            
    def get_seed(self):
        try:
            r = self.session.get(f"{self.url}/entangle", timeout=5)
            if r.status_code == 200:
                return int(r.json().get("bell_pair", "00"), 2)
        except: pass
        return random.getrandbits(2)

    def get_branch(self):
        try:
            r = self.session.get(f"{self.url}/ghz", timeout=5)
            if r.status_code == 200:
                s = r.json().get("ghz_state", "0"*12)
                return int(s[:12], 2)
        except: pass
        return random.getrandbits(12)


# ========= ROMAPY5 INTENT ENGINE (The Core Logic) =========
class ROMAPY5_Engine:
    def __init__(self):
        self.core = TwelveDVector()
        self.bridge = TesseractBridge()
        self.llm = None
        self.connect_gguf() 

    def connect_gguf(self):
        """Loads the local GGUF model for Intent Mapping."""
        if not LLAMA_CPP_AVAILABLE:
            print("[CRITICAL] llama-cpp-python not found. Intent mapping disabled.")
            return False
        
        if not os.path.exists(GGUF_MODEL_PATH):
            print(f"[ERROR] GGUF model not found at: {GGUF_MODEL_PATH}")
            return False

        try:
            load_start_time = time.time()
            self.llm = Llama(
                model_path=GGUF_MODEL_PATH,
                n_ctx=4096,           
                n_gpu_layers=0,       
                verbose=False,
                n_threads=os.cpu_count() or 4
            )
            load_end_time = time.time()
            print(f"✅ GGUF Model '{os.path.basename(GGUF_MODEL_PATH)}' Loaded in {load_end_time - load_start_time:.2f}s for Intent Mapping.")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load GGUF model for intent mapping: {e}")
            return False

    def assess(self, data: str):
        words = data.lower().split()
        pos = sum(w in {"win","yes","love","future","rise","success","good"} for w in words)
        neg = sum(w in {"no","fail","death","past","lose","bad"} for w in words)
        chaos = len(set(words)) / max(1, len(words))
        theta = (pos - neg) * 0.11 + chaos * 0.22

        for _ in range(11):
            seed = self.bridge.get_seed()
            branch = self.bridge.get_branch()
            self.core.quantum_rotate(theta, seed, branch)

        w = self.core.w_dominance()
        prob = max(0.0, min(100.0, (w + 3.2) / 6.4 * 100))
        strength = abs(w) ** 2.6

        return {
            "probability": round(prob, 2),
            "w_dominance": round(w, 4),
            "strength": round(strength, 3),
            "rotations": self.core.rotation_count,
            "interpretation": self._interpret(w)
        }

    def _interpret(self, w):
        if w > 1.6: return "ABSOLUTE FUTURE LOCK — This has already happened in the dominant branch"
        if w > 1.1: return "DOMINANT CONVERGENCE — The outcome is collapsing"
        if w > 0.5: return "Strong future current — Highly probable"
        if w > -0.5: return "Superposition — Multiple paths viable"
        return "Causal resistance — Intervention required"

    def map_intent_to_script(self, user_prompt: str) -> str:
        if not self.llm:
            return "[GGUF Model not loaded. Cannot map intent.]"

        state = self.assess(user_prompt) 

        system_prompt = f"""You are ROMAPY5 — the intent-based script execution engine.
W-dominance: {state['w_dominance']:.4f} | Rotations: {state['rotations']}
Convergence: {state['strength']:.3f}
State: {state['interpretation']}

Your sole task is to determine the user's intent and output a single, structured command.
You must choose from an imaginary, available script list: 'data_analyzer.py', 'system_monitor.py', 'financial_report.py', 'chat_interface.py'.

Format your response **EXACTLY** as:
SCRIPT_COMMAND: [script_name] [arg1] [arg2] ...

If the intent is unclear, use 'chat_interface.py' with the user's full prompt as the argument.
"""
        
        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.05,
                max_tokens=100,
                stop=["\n", "SCRIPT_COMMAND:"] 
            )
            
            llm_response = response['choices'][0]['message']['content'].strip()
            
            if llm_response.startswith("SCRIPT_COMMAND:"):
                 return llm_response.split("SCRIPT_COMMAND:")[1].strip()
            
            return llm_response

        except Exception as e:
            return f"[LLM Inference Failed during Intent Mapping: {e}]"

    def execute_romapy5_client(self, command: str) -> str:
        """
        SIMULATES the execution of romapy5-exec.sh on the predicted script.
        """
        parts = command.split()
        if not parts:
            return "[ERROR] Empty command from Intent Mapper."

        script_name = parts[0]
        args = parts[1:]
        
        # --- Conceptual Hybrid Execution Logic ---
        if script_name == "test.py" or "test" in script_name:
            exec_path = "romapy-llm.py (Semantic Kernel)"
            exec_mode = "✅ CACHE HIT: Instant Memory Execution."
            execution_type = "Semantic Execution"
            final_result = f"Result simulated from Semantic Execution of {script_name}. Final Output: All tests pass! RomaPy inferred quantum entanglement."
        else:
            exec_path = "python3 (Native Execution)"
            exec_mode = "🧠 CACHE MISS: Running native python3 to learn/compile."
            execution_type = "Native Execution (Learning)"
            final_result = f"Result simulated from Native Execution of {script_name}. File added to memory for future instant execution."

        output_msg = (
            f"\n--- ROMAPY5 HYBRID EXECUTION CLIENT ---\n"
            f"🎯 Target Script: **{script_name}**\n"
            f"⚙️ Arguments: {args}\n"
            f"🚀 Execution Path: {exec_path}\n"
            f"🔍 Status: {exec_mode}\n"
            f"----------------------------------------\n"
            f"FINAL EXECUTION OUTPUT ({execution_type}):\n{final_result}\n"
        )
        return output_msg


# ========= MAIN APP (ROMAPY5_App) =========
class ROMAPY5_App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ROMAPY5 Ω-EDITION — Intent-Based Script Loader")
        self.geometry("1800x1000")
        
        self.engine = ROMAPY5_Engine()
        self.tts = pyttsx3.init() if pyttsx3 else None
        if self.tts: self.tts.setProperty('rate', 160)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.build_ui()
        self.append(f"""ROMAPY5 Ω-EDITION ONLINE
Local GGUF ({os.path.basename(GGUF_MODEL_PATH)}) active for Intent Mapping | 12D Core active
Hybrid Execution (romapy5-exec.sh simulation) ready.
Type /romapy5 [intent] to instantly execute a script based on prediction.""")
        self.connect_gguf() 

    # (UI methods and handlers are the same as the previous response)
    def build_ui(self):
        # ... (UI definition, identical to the previous script)
        sidebar = ctk.CTkFrame(self, width=400, fg_color="#0a001f")
        sidebar.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(sidebar, text="ROMAPY5", font=("Orbitron", 32, "bold"), text_color="#00ffff").pack(pady=40)
        ctk.CTkLabel(sidebar, text="Ω-EDITION", font=("Orbitron", 18), text_color="#ff00ff").pack()
        ctk.CTkLabel(sidebar, text="RomanAILabs © 2025", text_color="#00ff88").pack(pady=(0,30))

        ctk.CTkButton(sidebar, text="Connect / Load GGUF", command=self.connect_gguf, height=55, fg_color="#00ff41").pack(pady=20, padx=40, fill="x")
        ctk.CTkButton(sidebar, text="12D Intent Assess", command=self.quick_assess, height=55, fg_color="#8a2be2").pack(pady=10, padx=40, fill="x")
        ctk.CTkButton(sidebar, text="Live 12D Field", command=self.show_field, height=55).pack(pady=10, padx=40, fill="x")

        self.status = ctk.CTkLabel(sidebar, text="Ready", text_color="#ffff00")
        self.status.pack(side="bottom", pady=50)

        main = ctk.CTkFrame(self, fg_color="#000011")
        main.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)

        self.chat = ctk.CTkTextbox(main, font=("Consolas", 16), text_color="#00ffcc", wrap="word")
        self.chat.grid(row=0, column=0, sticky="nsew", pady=(0,15))
        self.chat.configure(state="disabled")

        inp = ctk.CTkFrame(main)
        inp.grid(row=1, column=0, sticky="ew")
        inp.grid_columnconfigure(0, weight=1)

        self.entry = ctk.CTkEntry(inp, placeholder_text="Type /romapy5 [your intent]...", height=70, font=("Consolas", 18))
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0,10))
        self.entry.bind("<Return>", lambda e: self.send())

        ctk.CTkButton(inp, text="EXECUTE INTENT", command=self.send, height=70, width=200, fg_color="#ff00ff").grid(row=0, column=1)
    
    def append(self, text):
        self.chat.configure(state="normal")
        ts = datetime.now().strftime("%H:%M:%S")
        self.chat.insert("end", f"[{ts}] {text}\n\n")
        self.chat.configure(state="disabled")
        self.chat.see("end")
        if self.tts:
            threading.Thread(target=lambda: (self.tts.say(text.split("\n")[0]), self.tts.runAndWait()), daemon=True).start()
            
    def connect_gguf(self):
        if self.engine.llm:
            self.status.configure(text="ROMAPY5 Ω: GGUF ACTIVE", text_color="#00ff00")
            self.append(f"GGUF Model '{os.path.basename(GGUF_MODEL_PATH)}' is loaded.\nIntent mapping is LIVE.")
        else:
            self.status.configure(text="ROMAPY5 Ω: GGUF FAILED", text_color="#ff0000")
            self.append(f"GGUF failed to load. Check installation or '{GGUF_MODEL_PATH}'.")

    def send(self):
        msg = self.entry.get().strip()
        if not msg: return
        self.entry.delete(0, "end")
        self.append(f"You → {msg}")
        
        if msg.lower().startswith("/romapy5"):
            data = msg[8:].strip()
            threading.Thread(target=self.run_intent_execution, args=(data,), daemon=True).start()
        else:
            threading.Thread(target=self.run_intent_execution, args=(msg,), daemon=True).start()

    def run_intent_execution(self, user_query):
        if not self.engine.llm:
            self.after(0, lambda: self.append("GGUF Model not loaded. Cannot map intent."))
            return

        self.append(f"ROMAPY5 engaging 12D core and LLM for intent: '{user_query}'...")
        
        script_command = self.engine.map_intent_to_script(user_query)
        
        if script_command.startswith("["):
            self.after(0, lambda: self.append(f"ROMAPY5 Intent Mapping Error: {script_command}"))
        else:
            self.after(0, lambda: self.append(f"LLM Predicted Command: SCRIPT_COMMAND: **{script_command}**"))
            
            execution_result_msg = self.engine.execute_romapy5_client(script_command)
            self.after(0, lambda: self.append(execution_result_msg))
            self.after(0, lambda: self.append("ROMAPY5 Task Complete."))

    def quick_assess(self):
        data = self.entry.get().strip()
        if data: self.run_intent_execution(data)

    def show_field(self):
        if not MPL_AVAILABLE:
            self.append("Install matplotlib for live field view.")
            return
        win = ctk.CTkToplevel(self)
        win.title("ROMAPY5 12D QUANTUM FIELD")
        win.geometry("1000x900")
        fig = plt.figure(figsize=(12,10), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.grid(False)
        ax.axis('off')

        def anim(i):
            ax.clear()
            ax.set_xlim(-3,3); ax.set_ylim(-3,3); ax.set_zlim(-3,3)
            w = self.engine.core.w_dominance()
            color = plt.cm.plasma(max(0, min(1, (w + 4)/8)))
            ax.scatter(0, 0, w, c=[color], s=800, depthshade=False, edgecolors='cyan', linewidth=3)
            ax.text(0, 0, w+0.6, f"W-DOMINANCE = {w:.3f}", color='white', fontsize=18)
            ax.set_title("ROMAPY5 12D FIELD — TESSERACT ACTIVE", color='#ff00ff', fontsize=20)

        canvas = FigureCanvasTkAgg(fig, win)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        animation.FuncAnimation(fig, anim, interval=200, cache_frame_data=False)
        canvas.draw()


if __name__ == "__main__":
    print("Launching ROMAPY5 Ω-EDITION — Intent-Based Script Loader")
    print(f"Expecting the GGUF model file at: {GGUF_MODEL_PATH}")
    app = ROMAPY5_App()
    app.mainloop()
