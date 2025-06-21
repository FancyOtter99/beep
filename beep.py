from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
import numpy as np
from scipy.io.wavfile import write

def midi_to_freq(midi_note):
    return 440.0 * (2 ** ((midi_note - 69) / 12))

def generate_tone(freq, duration, sample_rate=44100):
    if freq <= 0:
        return np.zeros(int(sample_rate * duration))
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    return tone

def main(audio_file):
    print("Transcribing piano audio...")

    # Step 1: Transcribe using deep model
    (audio, _) = load_audio(audio_file, sr=sample_rate, mono=True)
    transcriptor = PianoTranscription(device='cpu')
    notes = transcriptor.transcribe(audio)

    print(f"Detected {len(notes)} notes")

    # Step 2: Process notes → beep() format
    beep_lines = []
    wave_chunks = []

    for note in notes:
        freq = midi_to_freq(note['midi_note'])
        duration = note['offset_time'] - note['onset_time']
        if duration <= 0: continue  # skip bad notes
        beep_lines.append(f"beep({round(freq, 2)}, {round(duration, 2)})")
        wave_chunks.append(generate_tone(freq, duration, sample_rate))

    # Step 3: Write .txt file
    with open("notes.txt", "w") as f:
        for line in beep_lines:
            f.write(line + "\n")
    print("Saved notes.txt")

    # Step 4: Generate .wav file
    if wave_chunks:
        waveform = np.concatenate(wave_chunks)
        waveform /= np.max(np.abs(waveform))  # normalize
        write("notes.wav", sample_rate, np.int16(waveform * 32767))
        print("Saved notes.wav")
    else:
        print("No audio generated — no valid notes found")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python piano_to_beeps.py your_song.wav")
        sys.exit(1)
    main(sys.argv[1])
