"""Master script: generate all figures for the presentation."""

import subprocess
import sys
import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPTS_DIR, "..", "data", "results")


def run_script(name: str):
    path = os.path.join(SCRIPTS_DIR, name)
    print(f"Running {name}...")
    try:
        subprocess.run([sys.executable, path], check=True, timeout=60)
        print(f"  Done.")
    except FileNotFoundError:
        print(f"  Script not found: {path}")
    except subprocess.CalledProcessError as e:
        print(f"  Failed with return code {e.returncode}")
    except subprocess.TimeoutExpired:
        print(f"  Timed out")


def main():
    print("=== Generating Presentation Figures ===\n")

    scripts = [
        "plot_nmse_vs_iterations.py",
        "plot_doa_spectrum.py",
        "plot_doa_rmse_vs_snr.py",
        "plot_speed_accuracy.py",
        "plot_audio_waveforms.py",
        "plot_ablation.py",
    ]

    for script in scripts:
        run_script(script)

    print(f"\nFigures saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
