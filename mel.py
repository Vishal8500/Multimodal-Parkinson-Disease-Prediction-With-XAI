import os
import torchaudio
import torch
import matplotlib.pyplot as plt

def save_melspectrogram(audio_path, output_dir='mel_outputs', sample_rate=16000, n_mels=128, fixed_length=5):
    """
    Generates and saves a Mel spectrogram from an audio file.
    
    Args:
        audio_path (str): Path to the input WAV file.
        output_dir (str): Directory to save the spectrogram image.
        sample_rate (int): Target sampling rate.
        n_mels (int): Number of Mel bands.
        fixed_length (float): Length of audio in seconds (pads/truncates if needed).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    waveform = waveform.mean(dim=0)  # mono

    fixed_samples = int(sample_rate * fixed_length)
    if waveform.shape[0] < fixed_samples:
        pad = fixed_samples - waveform.shape[0]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    else:
        waveform = waveform[:fixed_samples]

    # Mel spectrogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=1024,
        hop_length=256
    )
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    
    mel_spec = mel_spectrogram(waveform.unsqueeze(0))
    mel_spec_db = amplitude_to_db(mel_spec)
    
    # Normalize
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
    
    # Plot and save
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec_db[0].numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(format="%+2.f dB")
    plt.title(f"Mel Spectrogram: {os.path.basename(audio_path)}")
    save_path = os.path.join(output_dir, f"{os.path.basename(audio_path).split('.')[0]}_mel.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

# ========================
# Example usage
# ========================
audio_file = r"D:\SSDM\ssdm da-2\Speech\spontaneous\HC\ID15_hc_0_0_0.wav"
save_melspectrogram(audio_file)
