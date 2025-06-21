# thisisberk.py
from flask import Flask, send_file, Response
from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
import numpy as np
from scipy.io.wavfile import write
import os

app = Flask(__name__)

def midi_to_freq(midi_note):
    return 440.0 * (2 ** ((midi_note - 69) / 12))

def generate_tone(freq, duration, sample_rate=44100):
    if freq <= 0:
        return np.zeros(int(sample_rate * duration))
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    return tone

def transcribe_and_generate(audio_file):
    (audio, _) = load_audio(audio_file, sr=sample_rate, mono=True)
    transcriptor = PianoTranscription(device='cpu')
    notes = transcriptor.transcribe(audio)

    beep_lines = []
    wave_chunks = []

    for note in notes:
        freq = midi_to_freq(note['midi_note'])
        duration = note['offset_time'] - note['onset_time']
        if duration <= 0:
            continue
        beep_lines.append(f"beep({round(freq, 2)}, {round(duration, 2)})")
        wave_chunks.append(generate_tone(freq, duration, sample_rate))

    # Write output files
    with open("notes.txt", "w") as f:
        for line in beep_lines:
            f.write(line + "\n")

    if wave_chunks:
        waveform = np.concatenate(wave_chunks)
        waveform /= np.max(np.abs(waveform))  # normalize
        write("notes.wav", sample_rate, np.int16(waveform * 32767))
    else:
        # Write an empty wav if no notes detected
        write("notes.wav", sample_rate, np.zeros(1, dtype=np.int16))

@app.route("/")
def index():
    audio_file = "thisisberk.wav"
    if not os.path.isfile(audio_file):
        return "Audio file not found.", 404

    transcribe_and_generate(audio_file)

    # Prepare response HTML with links to download files
    html = """
    <h1>Transcription complete</h1>
    <p><a href="/download/notes.txt">Download notes.txt</a></p>
    <p><a href="/download/notes.wav">Download notes.wav</a></p>
    """
    return Response(html, mimetype="text/html")

@app.route("/download/<filename>")
def download_file(filename):
    if filename not in ("notes.txt", "notes.wav"):
        return "File not found", 404
    return send_file(filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

