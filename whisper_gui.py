import tkinter as tk
from tkinter import filedialog, messagebox
import whisper
import os
import threading
import webbrowser

# Load the model once (change "small" to your preferred model)
MODEL = whisper.load_model("small")
OUTPUT_FOLDER = "transcripts"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".flac", ".mp3", ".wav")

def transcribe_files(filepaths, status_label):
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        status_label.config(text=f"Transcribing {filename}...")
        result = MODEL.transcribe(filepath)
        txt_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
    status_label.config(text="All done! Transcripts saved.")
    messagebox.showinfo("Done", "Transcription complete!")

def select_files(status_label):
    files = filedialog.askopenfilenames(
        title="Select video/audio files",
        filetypes=[("Media files", "*.mp4 *.mkv *.avi *.mov *.flac *.mp3 *.wav")]
    )
    if files:
        threading.Thread(target=transcribe_files, args=(files, status_label), daemon=True).start()

def open_output_folder():
    webbrowser.open(os.path.abspath(OUTPUT_FOLDER))

root = tk.Tk()
root.title("Whisper Batch Transcriber")
root.geometry("400x200")

frame = tk.Frame(root)
frame.pack(expand=True, fill="both", padx=20, pady=20)

label = tk.Label(frame, text="Drag and drop files or click below to select", font=("Arial", 12))
label.pack(pady=10)

status_label = tk.Label(frame, text="", fg="blue")
status_label.pack(pady=5)

select_btn = tk.Button(frame, text="Select Files", command=lambda: select_files(status_label))
select_btn.pack(pady=5)

open_btn = tk.Button(frame, text="Open Transcript Folder", command=open_output_folder)
open_btn.pack(pady=5)

root.mainloop()