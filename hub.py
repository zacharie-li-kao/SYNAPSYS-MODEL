# Console encoding must be set before anything prints: the suite emits
# non-ASCII physics notation, which aborts print() on a cp1252 Windows
# console. See console_io for the failures this caused.
from console_io import enable_utf8_console
enable_utf8_console()

import customtkinter as ctk
import subprocess
import sys
import os
import tkinter as tk
from tkinter import  messagebox
from PIL import Image, ImageTk

# --- Configuration ---
SCRIPT_CHARACTERISATION = "keithley_analyser.py"
SCRIPT_DATA_CONVERTER = "data_converter.py"
SCRIPT_MODELLING = "Main.py"
SCRIPT_TEST_DATA = "generate_test_data.py"
LOGO_FILE = "logo.png"
# ---------------------

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class SynapseSuiteHub(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("SYNAPSYS - Main Hub")
        self.geometry("1100x700")

        # --- Title Frame ---
        self.title_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.title_frame.pack(fill="x", pady=(20, 10))

        if os.path.exists(LOGO_FILE):
            try:
                logo_img = Image.open(LOGO_FILE)
                max_width, max_height = 200, 200  # your intended bounding box
        
                # Compute scale preserving aspect ratio
                w, h = logo_img.size
                scale = min(max_width / w, max_height / h)
                new_size = (int(w * scale), int(h * scale))
        
                logo_img = logo_img.resize(new_size, Image.Resampling.LANCZOS)
                self.logo_ctk = ctk.CTkImage(light_image=logo_img, dark_image=logo_img, size=new_size)
        
                self.logo_label = ctk.CTkLabel(self.title_frame, image=self.logo_ctk, text="")
                self.logo_label.pack(side="left", padx=(20, 20))
            except Exception as e:
                print(f"Warning: could not load logo.png ({e})")

        # Title text
        self.title_label = ctk.CTkLabel(
            self.title_frame,
            text="SYNAPSYS: Synaptic Systems Characterisation and Modelling",
            font=ctk.CTkFont(size=26, weight="bold")
        )
        self.title_label.pack(anchor="center", pady=(10, 5))

        self.subtitle_label = ctk.CTkLabel(
            self.title_frame,
            text="A unified pipeline for experimental characterisation, physical model extraction, and network simulation.",
            font=ctk.CTkFont(size=14, slant="italic"),
            text_color="gray50"
        )
        self.subtitle_label.pack(anchor="center", pady=(0, 20))

        # --- Main Frame ---
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=10)
        self.main_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # --- Tile 1: Characterisation ---
        self.create_tile(
            self.main_frame,
            column=0,
            icon="🔬",
            title="Experimental Control",
            description=(
                "Launch the sourcemeter control GUI. This module performs "
                "in-situ electrical and optical characterisation of synaptic devices. "
                "It runs pulse-read sequences (LTP/LTD, SRDP, STDP, cycling) "
                "and exports standardized JSON characterisation suites."
            ),
            button_text="Launch Characterisation Tool",
            button_color="#C62828",
            hover_color="#8E0000",
            script_name=SCRIPT_CHARACTERISATION
        )

        # --- Tile 2: Data Conversion ---
        tile2_frame = self.create_tile_with_extra_button(
            self.main_frame,
            column=1,
            icon="🗂️",
            title="Data Conversion Pipeline",
            description=(
                "For data acquired from other setups. This tool loads raw CSV or text files "
                "and guides you through metadata tagging (e.g., 'potentiation', 'retention'). "
                "It processes the raw data and generates the same standardized JSON "
                "file required by the simulation module."
            ),
            button_text="Launch Data Converter",
            button_color="#1565C0",
            hover_color="#0D47A1",
            script_name=SCRIPT_DATA_CONVERTER
        )

        # Extra: Generate test data button
        self.test_data_button = ctk.CTkButton(
            tile2_frame,
            text="✨ Generate Test Data",
            fg_color="gray60",
            hover_color="gray50",
            text_color="#FFFFFF",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=30,
            command=lambda: self.launch_script(SCRIPT_TEST_DATA)
        )
        self.test_data_button.grid(row=4, column=0, padx=20, pady=(0, 20), sticky="sew")

        # --- Tile 3: Modelling ---
        self.create_tile(
            self.main_frame,
            column=2,
            icon="🧠",
            title="Model & Network Simulation",
            description=(
                "Load a JSON characterisation suite (experimental or synthetic). "
                "This module fits physical models (nonlinearity, spectral response) "
                "to extract parameters (α, β, λ-peaks). "
                "It then uses these parameters to simulate a full synaptic network."
            ),
            button_text="Launch Simulation",
            button_color="#2E7D32",
            hover_color="#1B5E20",
            script_name=SCRIPT_MODELLING
        )

        # --- Footer / Watermark ---
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", pady=(5, 15))

        footer_label = ctk.CTkLabel(
            footer,
            text="Developer: Zacharie Jehl Li-Kao    •    Contact: zacharie.jehl@upc.edu",
            font=ctk.CTkFont(size=12, slant="italic"),
            text_color="gray60"
        )
        footer_label.pack(side="right", padx=(0, 20))

    # --- Helper Methods ---

    def create_tile_with_extra_button(self, parent, column, icon, title, description, button_text, button_color, hover_color, script_name):
        tile_frame = ctk.CTkFrame(parent, border_width=1, corner_radius=10)
        tile_frame.grid(row=0, column=column, sticky="nsew", padx=10, pady=10)
        tile_frame.grid_columnconfigure(0, weight=1)
        tile_frame.grid_rowconfigure(2, weight=1)
        tile_frame.grid_rowconfigure(3, weight=0)
        tile_frame.grid_rowconfigure(4, weight=0)

        title_label = ctk.CTkLabel(tile_frame, text=f"{icon}  {title}", font=ctk.CTkFont(size=20, weight="bold"))
        title_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        separator = ctk.CTkFrame(tile_frame, height=2, fg_color="gray80")
        separator.grid(row=1, column=0, padx=20, pady=(0, 15), sticky="ew")

        desc_label = ctk.CTkLabel(tile_frame, text=description, font=ctk.CTkFont(size=13), wraplength=280, justify="left")
        desc_label.grid(row=2, column=0, padx=20, pady=10, sticky="nw")

        button = ctk.CTkButton(
            tile_frame,
            text=button_text,
            fg_color=button_color,
            hover_color=hover_color,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=45,
            command=lambda: self.launch_script(script_name)
        )
        button.grid(row=3, column=0, padx=20, pady=(15, 10), sticky="sew")
        return tile_frame

    def create_tile(self, parent, column, icon, title, description, button_text, button_color, hover_color, script_name):
        tile_frame = ctk.CTkFrame(parent, border_width=1, corner_radius=10)
        tile_frame.grid(row=0, column=column, sticky="nsew", padx=10, pady=10)
        tile_frame.grid_columnconfigure(0, weight=1)
        tile_frame.grid_rowconfigure(2, weight=1)
        tile_frame.grid_rowconfigure(3, weight=0)

        title_label = ctk.CTkLabel(tile_frame, text=f"{icon}  {title}", font=ctk.CTkFont(size=20, weight="bold"))
        title_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")

        separator = ctk.CTkFrame(tile_frame, height=2, fg_color="gray80")
        separator.grid(row=1, column=0, padx=20, pady=(0, 15), sticky="ew")

        desc_label = ctk.CTkLabel(tile_frame, text=description, font=ctk.CTkFont(size=13), wraplength=280, justify="left")
        desc_label.grid(row=2, column=0, padx=20, pady=10, sticky="nw")

        button = ctk.CTkButton(
            tile_frame,
            text=button_text,
            fg_color=button_color,
            hover_color=hover_color,
            font=ctk.CTkFont(size=14, weight="bold"),
            height=45,
            command=lambda: self.launch_script(script_name)
        )
        button.grid(row=3, column=0, padx=20, pady=(20, 20), sticky="sew")

    def launch_script(self, script_name):
        """
        Launches either a .exe (preferred) or a .py (fallback).
        Behavior:
          - If <script>.exe exists in the same directory as the hub, run it directly.
          - Else if running from source (not frozen) and a .py exists, run it with sys.executable.
          - Else show an explanatory error (frozen hub cannot run .py via sys.executable).
        """
        # Determine base directory where hub and modules live:
        # - If frozen (PyInstaller/cx_Freeze), the executables will be next to sys.executable
        # - Else (running from source) use the script directory
        is_frozen = getattr(sys, "frozen", False)
        if is_frozen:
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
    
        # Support absolute or relative script_name
        if os.path.isabs(script_name):
            requested_path = script_name
        else:
            requested_path = os.path.join(base_dir, script_name)
    
        # Normalize path and names
        requested_path = os.path.normpath(requested_path)
        root, ext = os.path.splitext(requested_path)
        exe_candidate = root + ".exe"
    
        # If the user passed a bare name like "generate_test_data.py" but your exe uses different
        # casing or suffix, you could extend matching logic here (not included to keep predictable).
    
        # File-not-found handling (we will check both .exe and .py before giving up)
        if not (os.path.exists(exe_candidate) or os.path.exists(requested_path)):
            # try also the case where user passed .py but exe is in same dir with same basename
            alt_py = root + ".py"
            if not os.path.exists(exe_candidate) and not os.path.exists(alt_py):
                tk.messagebox.showerror(
                    "Error: File Not Found",
                    f"Could not find the module:\n{script_name}\n\n"
                    f"Searched:\n  {exe_candidate}\n  {requested_path}\n\n"
                    "Please ensure the hub and modules are in the same folder, or provide absolute paths."
                )
                return
    
        print(f"--- [SYNAPSYS HUB] Launching '{script_name}'... ---")
    
        try:
            if os.path.exists(exe_candidate):
                # Launch the compiled exe if present (preferred)
                subprocess.Popen([exe_candidate], cwd=base_dir)
            elif ext.lower() == ".exe" and os.path.exists(requested_path):
                # User already passed an .exe full path or name
                subprocess.Popen([requested_path], cwd=base_dir)
            elif ext.lower() == ".py" and os.path.exists(requested_path):
                # We have a .py to run. Only possible when hub is NOT frozen.
                if is_frozen:
                    # frozen hub cannot use sys.executable to run .py (sys.executable is hub.exe)
                    tk.messagebox.showerror(
                        "Cannot Launch Python Script",
                        "This hub is running as a frozen executable and cannot directly run .py modules.\n\n"
                        "Options:\n"
                        "  • Compile the module(s) to .exe (recommended), or\n"
                        "  • Run the hub as hub.py with a Python interpreter so it can launch .py modules."
                    )
                    return
                # Running from source — invoke python interpreter with the .py file
                subprocess.Popen([sys.executable, requested_path], cwd=base_dir)
            else:
                # As a last resort, if an unexpected file exists, try launching it directly
                subprocess.Popen([requested_path], cwd=base_dir)
            
            # Keep your special test-data informational message (match by basename)
            if os.path.basename(root).lower().startswith("generate_test_data"):
                tk.messagebox.showinfo(
                    "Test Data Generation",
                    "The 'generate_test_data' module has created:\n\n"
                    "1. CSV files of the synthetic data.\n"
                    "2. A 'METADATA_INSTRUCTIONS.txt' file for use with the Data Conversion Pipeline."
                )
    
        except Exception as e:
            tk.messagebox.showerror("Error Launching Script", f"Failed to launch '{script_name}'.\n\nError: {e}")
            print(f"--- [SYNAPSYS HUB] ERROR launching '{script_name}': {e} ---")


if __name__ == "__main__":
    app = SynapseSuiteHub()
    app.mainloop()
