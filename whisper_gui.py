import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import whisper
import os
import threading
import webbrowser

# Load the model once (change "small" to your preferred model)
MODEL = whisper.load_model("small")
OUTPUT_FOLDER = "transcripts"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".flac", ".mp3", ".wav")

def transcribe_files(filepaths, status_label, progress_bar):
    total = len(filepaths)
    for idx, filepath in enumerate(filepaths, 1):
        filename = os.path.basename(filepath)
        status_label.config(text=f"Transcribing {filename}...")
        progress_bar['value'] = (idx - 1) / total * 100
        progress_bar.update_idletasks()
        result = MODEL.transcribe(filepath)
        txt_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
    status_label.config(text="All done! Transcripts saved.")
    progress_bar['value'] = 100
    progress_bar.update_idletasks()
    messagebox.showinfo("Done", "Transcription complete!")

def select_files(status_label, progress_bar):
    files = filedialog.askopenfilenames(
        title="Select video/audio files",
        filetypes=[("Media files", "*.mp4 *.mkv *.avi *.mov *.flac *.mp3 *.wav")]
    )
    if files:
        progress_bar['value'] = 0
        threading.Thread(target=transcribe_files, args=(files, status_label, progress_bar), daemon=True).start()

def open_output_folder():
    webbrowser.open(os.path.abspath(OUTPUT_FOLDER))

def center_window(win, width, height):
    win.update_idletasks()
    screen_width = win.winfo_screenwidth()
    screen_height = win.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    win.geometry(f'{width}x{height}+{x}+{y}')

root = tk.Tk()
root.title("Whisper Batch Transcriber")
window_width, window_height = 600, 350
center_window(root, window_width, window_height)
root.configure(bg="#23272f")

frame = tk.Frame(root, bg="#2c313c")
frame.pack(expand=True, fill="both", padx=30, pady=30)

header = tk.Label(frame, text="GameScribe: Whisper Batch Transcriber", font=("Segoe UI", 20, "bold"), fg="#f5c518", bg="#2c313c")
header.pack(pady=(0, 18))

label = tk.Label(frame, text="Drag and drop files or click below to select", font=("Segoe UI", 14), fg="#ffffff", bg="#2c313c")
label.pack(pady=10)

status_label = tk.Label(frame, text="", fg="#00bfff", bg="#2c313c", font=("Segoe UI", 12, "italic"))
status_label.pack(pady=5)

progress_bar = ttk.Progressbar(frame, orient="horizontal", length=400, mode="determinate")
progress_bar.pack(pady=10)

select_btn = tk.Button(frame, text="Select Files", command=lambda: select_files(status_label, progress_bar), font=("Segoe UI", 12, "bold"), bg="#f5c518", fg="#23272f", activebackground="#ffe066", activeforeground="#23272f", bd=0, padx=10, pady=6)
select_btn.pack(pady=8)

open_btn = tk.Button(frame, text="Open Transcript Folder", command=open_output_folder, font=("Segoe UI", 12), bg="#393e46", fg="#f5c518", activebackground="#23272f", activeforeground="#f5c518", bd=0, padx=10, pady=6)
open_btn.pack(pady=8)

root.mainloop()