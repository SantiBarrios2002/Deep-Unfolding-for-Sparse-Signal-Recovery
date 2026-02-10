"""Experiment 4: Plot audio waveforms (original, corrupted, recovered)."""

import numpy as np
import matplotlib.pyplot as plt
import sys

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


def main(original: str, corrupted: str, recovered_ista: str,
         recovered_lista: str = None):
    if not HAS_SOUNDFILE:
        print("soundfile not installed. Install with: pip install soundfile")
        return

    orig, sr = sf.read(original)
    corr, _ = sf.read(corrupted)
    rec_ista, _ = sf.read(recovered_ista)

    t = np.arange(len(orig)) / sr

    n_plots = 4 if recovered_lista else 3
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2.5 * n_plots), sharex=True)

    axes[0].plot(t, orig, "b", linewidth=0.3)
    axes[0].set_title("Original")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(t, corr, "r", linewidth=0.3)
    axes[1].set_title("Corrupted (50% samples dropped)")
    axes[1].set_ylabel("Amplitude")

    axes[2].plot(t, rec_ista, "g", linewidth=0.3)
    axes[2].set_title("Recovered (ISTA, 500 iterations)")
    axes[2].set_ylabel("Amplitude")

    if recovered_lista and n_plots == 4:
        rec_lista, _ = sf.read(recovered_lista)
        axes[3].plot(t[:len(rec_lista)], rec_lista, "m", linewidth=0.3)
        axes[3].set_title("Recovered (LISTA, 16 layers)")
        axes[3].set_ylabel("Amplitude")

    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig("data/results/audio_waveforms.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: plot_audio_waveforms.py <original.wav> <corrupted.wav> "
              "<recovered_ista.wav> [recovered_lista.wav]")
    else:
        lista = sys.argv[4] if len(sys.argv) > 4 else None
        main(sys.argv[1], sys.argv[2], sys.argv[3], lista)
