#!/usr/bin/env python3
"""
Porównanie błędów regulatorów PID i Pure Pursuit na podstawie
plików all_trajectories_summary_pid*.txt oraz all_trajectories_summary_pp*.txt.

Umieść ten skrypt w tym samym katalogu co oba pliki TXT i uruchom:
    python3 plot_pid_vs_pp.py
"""

from pathlib import Path
import re
import matplotlib.pyplot as plt


HERE = Path("/home/developer/ros2_ws/src/roboracer_sim/config/trajectory_comparison")

PID_FILE = HERE / "all_trajectories_summary_pid.txt"
PP_FILE = HERE / "all_trajectories_summary_pp.txt"

METRICS = {
    "MAE": ("MAE [m]", "MAE – średni błąd bezwzględny"),
    "RMSE": ("RMSE [m]", "RMSE – pierwiastek błędu średniokwadratowego"),
    "Mediana": ("Mediana [m]", "Mediana błędu"),
    "P95": ("P95 [m]", "95. percentyl błędu"),
    "Max": ("Max [m]", "Maksymalny błąd"),
}


def read_summary(path):
    """Odczytuje wiersze tabeli z pliku TXT."""
    results = {}
    row_pattern = re.compile(
        r"^\s*(\d+(?:[.,]\d+)?)\s*\|\s*\d+\s*\|\s*"
        r"(\d+(?:[.,]\d+)?)\s*\|\s*"
        r"(\d+(?:[.,]\d+)?)\s*\|\s*"
        r"(\d+(?:[.,]\d+)?)\s*\|\s*"
        r"(\d+(?:[.,]\d+)?)\s*\|\s*"
        r"(\d+(?:[.,]\d+)?)\s*\|\s*"
        r"(\d+(?:[.,]\d+)?)"
    )

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            match = row_pattern.match(line)
            if not match:
                continue

            values = [
                float(value.replace(",", "."))
                for value in match.groups()
            ]
            speed = values[0]
            results[round(speed, 4)] = {
                "Długość": values[1],
                "MAE": values[2],
                "RMSE": values[3],
                "Mediana": values[4],
                "P95": values[5],
                "Max": values[6],
            }

    if not results:
        raise ValueError(f"Nie znaleziono danych tabelarycznych w pliku: {path}")
    return results


def plot_metric(pid, pp, metric, ylabel, title):
    """Rysuje jeden wykres porównujący PID i Pure Pursuit."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for name, results, marker in (
        ("PID", pid, "o"),
        ("Pure Pursuit", pp, "s"),
    ):
        speeds = sorted(results)
        values = [results[speed][metric] for speed in speeds]
        ax.plot(speeds, values, marker=marker, linewidth=2, label=name)

    ax.set_title(title)
    ax.set_xlabel("Prędkość zadana [m/s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()


def main():
    if PID_FILE is None or PP_FILE is None:
        print("Nie znaleziono plików TXT.")
        print("Wymagane pliki: all_trajectories_summary_pid*.txt")
        print("                all_trajectories_summary_pp*.txt")
        print("Umieść je w katalogu skryptu.")
        return

    pid = read_summary(PID_FILE)
    pp = read_summary(PP_FILE)

    print(f"PID: {PID_FILE.name} ({len(pid)} pomiarów)")
    print(f"Pure Pursuit: {PP_FILE.name} ({len(pp)} pomiarów)")
    print("Wyświetlam wykresy porównawcze...")

    # Osobny wykres dla każdego rodzaju błędu.
    for metric, (ylabel, title) in METRICS.items():
        plot_metric(pid, pp, metric, ylabel, title)

    # Dodatkowy wykres długości przejazdu – pomaga zauważyć nadmierne
    # zygzakowanie lub wydłużenie drogi względem toru referencyjnego.
    plot_metric(
        pid, pp, "Długość", "Długość przejazdu [m]",
        "Długość przejazdu – PID vs Pure Pursuit"
    )

    plt.show()


if __name__ == "__main__":
    main()
