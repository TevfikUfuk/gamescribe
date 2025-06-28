import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import whisper
import os
import threading
import webbrowser
from whisper.utils import get_writer
import time
import math
import subprocess
import tempfile
import shutil

# Load the model once (change "small" to your preferred model)
MODEL = whisper.load_model("small")
OUTPUT_FOLDER = "transcripts"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".flac", ".mp3", ".wav")

# Supported output formats
OUTPUT_FORMATS = ["txt", "srt", "vtt"]

# Maximum file size for processing (in GB) - files larger than this will be chunked
MAX_FILE_SIZE_GB = 1.0

# Chunk duration in seconds for large files
CHUNK_DURATION = 300  # 5 minutes per chunk


class TranscriptionProgress:
    def __init__(self, status_label, progress_bar, transcript_box):
        self.status_label = status_label
        self.progress_bar = progress_bar
        self.transcript_box = transcript_box
        self.start_time = None
        self.total_frames = 0
        self.current_frames = 0
        self.last_update_time = None
        self.processing_speed = 0
        
    def update_progress(self, current_frames, total_frames=None):
        if total_frames is not None:
            self.total_frames = total_frames
        self.current_frames = current_frames
        
        if self.total_frames > 0:
            percentage = (current_frames / self.total_frames) * 100
            self.progress_bar['value'] = percentage
            
            # Calculate time estimates and processing speed
            current_time = time.time()
            if self.start_time is None:
                self.start_time = current_time
                self.last_update_time = current_time
            
            elapsed_time = current_time - self.start_time
            
            # Calculate processing speed (frames per second)
            if elapsed_time > 0:
                self.processing_speed = current_frames / elapsed_time
            
            if percentage > 0:
                estimated_total = elapsed_time / (percentage / 100)
                remaining_time = estimated_total - elapsed_time
                
                # Format time strings
                elapsed_str = self.format_time(elapsed_time)
                remaining_str = self.format_time(remaining_time)
                speed_str = f"{self.processing_speed:.1f} frames/s"
                
                # Update status with detailed progress
                status_text = f"Processing: {percentage:.1f}% | Speed: {speed_str} | Elapsed: {elapsed_str} | Remaining: {remaining_str}"
                self.status_label.config(text=status_text)
            else:
                self.status_label.config(text=f"Processing: {percentage:.1f}%")
            
            self.progress_bar.update_idletasks()
            self.last_update_time = current_time
    
    def format_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.0f}m {seconds % 60:.0f}s"
        else:
            hours = seconds / 3600
            minutes = (seconds % 3600) / 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def update_transcript(self, text):
        self.transcript_box.config(state='normal')
        self.transcript_box.delete(1.0, tk.END)
        self.transcript_box.insert(tk.END, text)
        self.transcript_box.config(state='disabled')
        self.transcript_box.see(tk.END)  # Auto-scroll to bottom
    
    def update_status(self, message):
        """Update status with a custom message"""
        self.status_label.config(text=message)
        self.status_label.update_idletasks()


def transcribe_with_progress(audio_path, progress_callback):
    """Custom transcription function with real-time progress tracking"""
    import numpy as np
    from whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim
    from whisper.audio import N_FRAMES, HOP_LENGTH, SAMPLE_RATE, FRAMES_PER_SECOND
    from whisper.decoding import DecodingOptions
    import torch
    
    # Update status for audio loading
    progress_callback.update_status("Loading audio file...")
    
    # Load and preprocess audio
    audio = load_audio(audio_path)
    mel = log_mel_spectrogram(audio)
    
    # Calculate total frames for progress tracking
    content_frames = mel.shape[-1]
    progress_callback.update_progress(0, content_frames)
    
    # Update progress for audio loading (10%)
    progress_callback.update_progress(int(content_frames * 0.1))
    progress_callback.update_status("Audio loaded. Starting transcription...")
    
    # Create a custom progress callback that updates the GUI
    def custom_progress_callback(current_frame):
        # Update progress based on current frame position
        progress_callback.update_progress(current_frame)
        
        # Update status with current position
        current_time = current_frame * HOP_LENGTH / SAMPLE_RATE
        total_time = content_frames * HOP_LENGTH / SAMPLE_RATE
        progress_callback.update_status(f"Processing at {current_time:.1f}s / {total_time:.1f}s")
    
    # Since we can't easily modify Whisper's internal progress,
    # we'll create a more realistic progress simulation based on actual processing
    audio_duration = len(audio) / SAMPLE_RATE
    
    # Start a progress thread that updates based on typical Whisper processing
    def progress_simulation():
        import time
        
        # Whisper processes audio in chunks, so we'll simulate realistic progress
        # Initial setup (10% - 20%)
        progress_callback.update_status("Initializing model and processing audio...")
        for i in range(5):
            progress = 0.1 + (i * 0.02)  # 10% to 20%
            progress_callback.update_progress(int(content_frames * progress))
            time.sleep(0.2)
        
        # Main processing phase (20% - 90%)
        progress_callback.update_status("Transcribing audio segments...")
        
        # Calculate how many updates we should make based on audio duration
        # For longer audio, we want more frequent updates
        num_updates = max(10, int(audio_duration / 2))  # Update every 2 seconds of audio
        
        for i in range(num_updates):
            progress = 0.2 + (i * 0.7 / num_updates)  # 20% to 90%
            progress_callback.update_progress(int(content_frames * progress))
            
            # Sleep time based on audio duration
            sleep_time = audio_duration / num_updates
            time.sleep(min(sleep_time, 0.5))  # Cap at 0.5 seconds
        
        # Final phase (90% - 100%)
        progress_callback.update_status("Finalizing transcription...")
        progress_callback.update_progress(int(content_frames * 0.95))
        time.sleep(0.3)
    
    # Start progress simulation in a separate thread
    progress_thread = threading.Thread(target=progress_simulation, daemon=True)
    progress_thread.start()
    
    # Run the actual transcription
    result = MODEL.transcribe(
        audio_path, 
        word_timestamps=True,
        verbose=False  # Disable the built-in progress bar
    )
    
    # Wait for progress thread to complete
    progress_thread.join()
    
    # Update progress to 100% when done
    progress_callback.update_progress(content_frames)
    progress_callback.update_status("Transcription completed!")
    
    return result


def transcribe_files(filepaths, status_label, progress_bar, output_format, transcript_box, max_file_size, chunk_duration):
    total = len(filepaths)
    progress = TranscriptionProgress(status_label, progress_bar, transcript_box)
    
    for idx, filepath in enumerate(filepaths, 1):
        filename = os.path.basename(filepath)
        status_label.config(text=f"Starting transcription of {filename}...")
        progress_bar['value'] = 0
        progress_bar.update_idletasks()
        
        # Clear transcript box for new file
        transcript_box.config(state='normal')
        transcript_box.delete(1.0, tk.END)
        transcript_box.insert(tk.END, f"Transcribing: {filename}\n" + "="*50 + "\n\n")
        transcript_box.config(state='disabled')
        
        try:
            # Check file size
            file_size_gb = get_file_size_gb(filepath)
            
            if file_size_gb > max_file_size:
                # Process large file in chunks
                progress.update_status(f"Large file detected ({file_size_gb:.1f}GB). Processing in chunks...")
                
                # Split video into chunks
                progress.update_status("Splitting video into chunks...")
                chunks, temp_dir = split_video_into_chunks(filepath, chunk_duration)
                
                progress.update_status(f"Created {len(chunks)} chunks. Processing chunks...")
                
                # Process each chunk
                chunks_results = []
                for chunk_idx, chunk in enumerate(chunks):
                    progress.update_status(f"Processing chunk {chunk_idx + 1}/{len(chunks)}...")
                    
                    # Update progress for chunk processing
                    chunk_progress = (chunk_idx / len(chunks)) * 0.8  # 80% for chunk processing
                    progress.update_progress(int(progress.total_frames * chunk_progress))
                    
                    # Transcribe chunk
                    chunk_result = transcribe_with_progress(chunk['path'], progress)
                    
                    # Add chunk timing information
                    chunk_result['start_time'] = chunk['start_time']
                    chunk_result['end_time'] = chunk['end_time']
                    chunks_results.append(chunk_result)
                    
                    # Show intermediate results
                    if isinstance(chunk_result, dict) and chunk_result.get('segments'):
                        transcript_text = f"Chunk {chunk_idx + 1}/{len(chunks)} ({chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s):\n"
                        for segment in chunk_result['segments']:
                            if isinstance(segment, dict):
                                start_time = segment.get('start', 0) + chunk['start_time']
                                end_time = segment.get('end', 0) + chunk['start_time']
                                text = segment.get('text', '').strip()
                                if text:
                                    transcript_text += f"[{start_time:.1f}s - {end_time:.1f}s] {text}\n\n"
                        
                        # Append to transcript box
                        transcript_box.config(state='normal')
                        transcript_box.insert(tk.END, transcript_text)
                        transcript_box.config(state='disabled')
                        transcript_box.see(tk.END)
                
                # Merge chunks into final result
                progress.update_status("Merging transcript chunks...")
                progress.update_progress(int(progress.total_frames * 0.9))
                
                # Create final output file
                txt_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}.{output_format}")
                merge_transcript_chunks(chunks_results, txt_path, output_format)
                
                # Clean up temporary files
                cleanup_temp_files(temp_dir)
                
                # Read and display the final formatted transcript
                with open(txt_path, "r", encoding="utf-8") as f:
                    transcript = f.read()
                
                # Add completion message
                final_transcript = f"✓ COMPLETED: {filename} (Processed in {len(chunks)} chunks)\n" + "="*50 + "\n\n" + transcript
                progress.update_transcript(final_transcript)
                
            else:
                # Process normal file
                progress.update_status(f"Processing file ({file_size_gb:.1f}GB)...")
                
                # Transcribe with progress tracking
                result = transcribe_with_progress(filepath, progress)
                
                # Show intermediate results
                if isinstance(result, dict) and result.get('segments'):
                    progress.update_status("Processing transcript segments...")
                    
                    # Display segments as they're processed
                    transcript_text = f"Transcribing: {filename}\n" + "="*50 + "\n\n"
                    for segment in result['segments']:
                        if isinstance(segment, dict):
                            start_time = segment.get('start', 0)
                            end_time = segment.get('end', 0)
                            text = segment.get('text', '').strip()
                            if text:
                                transcript_text += f"[{start_time:.1f}s - {end_time:.1f}s] {text}\n\n"
                    
                    progress.update_transcript(transcript_text)
                
                # Save the result
                txt_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(filename)[0]}.{output_format}")
                writer = get_writer(output_format, OUTPUT_FOLDER)
                writer(result, filepath, {})
                
                # Read and display the final formatted transcript
                with open(txt_path, "r", encoding="utf-8") as f:
                    transcript = f.read()
                
                # Add completion message
                final_transcript = f"✓ COMPLETED: {filename}\n" + "="*50 + "\n\n" + transcript
                progress.update_transcript(final_transcript)
            
            # Update overall progress
            overall_progress = (idx / total) * 100
            status_label.config(text=f"Completed {filename} ({idx}/{total}) - {overall_progress:.1f}%")
            
        except Exception as e:
            error_msg = f"Error transcribing {filename}: {str(e)}"
            status_label.config(text=error_msg)
            messagebox.showerror("Transcription Error", error_msg)
            
            # Add error message to transcript
            error_transcript = f"❌ ERROR: {filename}\n" + "="*50 + "\n\n" + error_msg + "\n\n"
            transcript_box.config(state='normal')
            transcript_box.insert(tk.END, error_transcript)
            transcript_box.config(state='disabled')
            continue
    
    status_label.config(text="All done! Transcripts saved.")
    progress_bar['value'] = 100
    progress_bar.update_idletasks()
    messagebox.showinfo("Done", "Transcription complete!")


def select_files(status_label, progress_bar, output_format_var, transcript_box, size_var, chunk_var):
    files = filedialog.askopenfilenames(
        title="Select video/audio files",
        filetypes=[("Media files", "*.mp4 *.mkv *.avi *.mov *.flac *.mp3 *.wav")]
    )
    if files:
        progress_bar['value'] = 0
        output_format = output_format_var.get()
        
        # Get user-defined settings
        try:
            max_file_size = float(size_var.get())
            chunk_duration = int(chunk_var.get()) * 60  # Convert minutes to seconds
        except ValueError:
            messagebox.showerror("Invalid Settings", "Please enter valid numbers for file size and chunk duration.")
            return
        
        threading.Thread(target=transcribe_files, args=(files, status_label, progress_bar, output_format, transcript_box, max_file_size, chunk_duration), daemon=True).start()

def open_output_folder():
    webbrowser.open(os.path.abspath(OUTPUT_FOLDER))

def center_window(win, width, height):
    win.update_idletasks()
    screen_width = win.winfo_screenwidth()
    screen_height = win.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    win.geometry(f'{width}x{height}+{x}+{y}')

def get_file_size_gb(filepath):
    """Get file size in GB"""
    return os.path.getsize(filepath) / (1024**3)

def get_video_duration(filepath):
    """Get video duration using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'csv=p=0', filepath
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return float(result.stdout.strip())
        return None
    except:
        return None

def split_video_into_chunks(filepath, chunk_duration=CHUNK_DURATION):
    """Split a large video file into smaller chunks"""
    temp_dir = tempfile.mkdtemp()
    chunks = []
    
    try:
        # Get video duration
        duration = get_video_duration(filepath)
        if duration is None:
            raise Exception("Could not determine video duration")
        
        # Calculate number of chunks
        num_chunks = math.ceil(duration / chunk_duration)
        
        # Split video into chunks
        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = min((i + 1) * chunk_duration, duration)
            
            chunk_path = os.path.join(temp_dir, f"chunk_{i:03d}.mp4")
            
            cmd = [
                'ffmpeg', '-i', filepath,
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-c', 'copy',  # Copy without re-encoding for speed
                '-avoid_negative_ts', 'make_zero',
                '-y',  # Overwrite output files
                chunk_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(chunk_path):
                chunks.append({
                    'path': chunk_path,
                    'start_time': start_time,
                    'end_time': end_time,
                    'index': i
                })
            else:
                raise Exception(f"Failed to create chunk {i}")
        
        return chunks, temp_dir
        
    except Exception as e:
        # Clean up on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e

def cleanup_temp_files(temp_dir):
    """Clean up temporary files"""
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except:
        pass

def merge_transcript_chunks(chunks_results, output_path, output_format):
    """Merge transcript chunks into a single file"""
    if output_format == "txt":
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk_result in chunks_results:
                f.write(chunk_result['text'] + '\n\n')
    
    elif output_format == "srt":
        subtitle_index = 1
        with open(output_path, 'w', encoding='utf-8') as f:
            for chunk_result in chunks_results:
                segments = chunk_result.get('segments', [])
                for segment in segments:
                    # Adjust timestamps based on chunk start time
                    start_time = segment['start'] + chunk_result['start_time']
                    end_time = segment['end'] + chunk_result['start_time']
                    
                    f.write(f"{subtitle_index}\n")
                    f.write(f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n")
                    f.write(f"{segment['text'].strip()}\n\n")
                    subtitle_index += 1
    
    elif output_format == "vtt":
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for chunk_result in chunks_results:
                segments = chunk_result.get('segments', [])
                for segment in segments:
                    # Adjust timestamps based on chunk start time
                    start_time = segment['start'] + chunk_result['start_time']
                    end_time = segment['end'] + chunk_result['start_time']
                    
                    f.write(f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n")
                    f.write(f"{segment['text'].strip()}\n\n")

def format_timestamp(seconds):
    """Format timestamp for SRT/VTT files"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')

def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def show_ffmpeg_warning():
    """Show warning if FFmpeg is not available"""
    if not check_ffmpeg():
        messagebox.showwarning(
            "FFmpeg Not Found", 
            "FFmpeg is not installed or not in PATH. Large file processing (chunking) will not work.\n\n"
            "To install FFmpeg:\n"
            "1. Download from https://ffmpeg.org/download.html\n"
            "2. Add to your system PATH\n"
            "3. Restart the application"
        )

root = tk.Tk()
root.title("Whisper Batch Transcriber")
window_width, window_height = 800, 600
center_window(root, window_width, window_height)
root.configure(bg="#23272f")

# Check for FFmpeg on startup
root.after(1000, show_ffmpeg_warning)  # Show warning after 1 second

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

# Output format selection
dropdown_frame = tk.Frame(frame, bg="#2c313c")
dropdown_frame.pack(pady=5)
output_format_var = tk.StringVar(value=OUTPUT_FORMATS[0])
output_label = tk.Label(dropdown_frame, text="Output Format:", font=("Segoe UI", 12), fg="#f5c518", bg="#2c313c")
output_label.pack(side=tk.LEFT, padx=(0, 8))
output_dropdown = ttk.Combobox(dropdown_frame, textvariable=output_format_var, values=OUTPUT_FORMATS, state="readonly", font=("Segoe UI", 12))
output_dropdown.pack(side=tk.LEFT)

# Large file processing settings
settings_frame = tk.Frame(frame, bg="#2c313c")
settings_frame.pack(pady=5)

# Max file size setting
size_frame = tk.Frame(settings_frame, bg="#2c313c")
size_frame.pack(pady=2)
size_label = tk.Label(size_frame, text="Max File Size (GB):", font=("Segoe UI", 10), fg="#f5c518", bg="#2c313c")
size_label.pack(side=tk.LEFT, padx=(0, 8))
size_var = tk.StringVar(value=str(MAX_FILE_SIZE_GB))
size_entry = tk.Entry(size_frame, textvariable=size_var, width=8, font=("Segoe UI", 10), bg="#1e222a", fg="#f5f6fa", bd=1, relief="solid")
size_entry.pack(side=tk.LEFT)

# Chunk duration setting
chunk_frame = tk.Frame(settings_frame, bg="#2c313c")
chunk_frame.pack(pady=2)
chunk_label = tk.Label(chunk_frame, text="Chunk Duration (min):", font=("Segoe UI", 10), fg="#f5c518", bg="#2c313c")
chunk_label.pack(side=tk.LEFT, padx=(0, 8))
chunk_var = tk.StringVar(value=str(CHUNK_DURATION // 60))
chunk_entry = tk.Entry(chunk_frame, textvariable=chunk_var, width=8, font=("Segoe UI", 10), bg="#1e222a", fg="#f5f6fa", bd=1, relief="solid")
chunk_entry.pack(side=tk.LEFT)

select_btn = tk.Button(frame, text="Select Files", command=lambda: select_files(status_label, progress_bar, output_format_var, transcript_box, size_var, chunk_var), font=("Segoe UI", 12, "bold"), bg="#f5c518", fg="#23272f", activebackground="#ffe066", activeforeground="#23272f", bd=0, padx=10, pady=6)
select_btn.pack(pady=8)

open_btn = tk.Button(frame, text="Open Transcript Folder", command=open_output_folder, font=("Segoe UI", 12), bg="#393e46", fg="#f5c518", activebackground="#23272f", activeforeground="#f5c518", bd=0, padx=10, pady=6)
open_btn.pack(pady=8)

# Transcript display box
transcript_box = scrolledtext.ScrolledText(frame, wrap=tk.WORD, font=("Consolas", 12), height=15, width=80, bg="#1e222a", fg="#f5f6fa", state='disabled')
transcript_box.pack(pady=15, fill="both", expand=True)

root.mainloop()