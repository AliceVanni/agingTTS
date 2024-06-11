import numpy as np
import argparse
import matplotlib.pyplot as plt
import librosa.display

# Function to load and visualize mel spectrogram
def visualize_mel_spectrogram(args):
    mel_spectrogram = np.load(args.numpy_mel)
    output_path = f'mel_plot.png'
    title = f'{args.numpy_mel} Mel Spectrogram'
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram, sr=args.sample_rate, hop_length=args.hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-nm",
        "--numpy_mel",
        type=str,
        required=True,
        help="Name of the file with mel spectrogram as np array",
    )
    parser.add_argument(
        "-sr",
        "--sample_rate",
        type=int,
        required=True,
        help="Sampling rate",
    ) 
    parser.add_argument(
        "-hl",
        "--hop_length",
        type=int,
        required=True,
        default=512,
        help="hop length"
    )       
    args = parser.parse_args()
    
    visualize_mel_spectrogram(args)