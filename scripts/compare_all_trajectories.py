#!/usr/bin/env python3
"""
Porownanie wielu trajektorii rzeczywistych z trajektoria referencyjna.

Pliki wejsciowe w katalogu CONFIG_DIR:
  track_trajectory.csv
  trajectory_actual_pid_0_22.csv
  trajectory_actual_pid_0_25.csv
  ...
  trajectory_actual_pid_1_3.csv

Wynik:
  trajectory_comparison/all_trajectories_summary.txt

Wykresy sa wyswietlane przez plt.show() i NIE sa zapisywane na dysku.
Na kazdym wykresie: trajektoria referencyjna + maksymalnie 3 przejazdy.
"""

import csv
import math
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# KONFIGURACJA
# ============================================================

CONFIG_DIR = Path("/home/developer/ros2_ws/src/roboracer_sim/config")

IDEAL_CSV = CONFIG_DIR / "track_trajectory.csv"
#ACTUAL_PATTERN = "trajectory_actual_pid_*.csv"
ACTUAL_PATTERN = "trajectory_actual_pp_*.csv"

OUTPUT_DIR = CONFIG_DIR / "trajectory_comparison"
#SUMMARY_TXT = OUTPUT_DIR / "all_trajectories_summary_pid.txt"
SUMMARY_TXT = OUTPUT_DIR / "all_trajectories_summary_pp.txt"

# Ile trajektorii rzeczywistych pokazywac na jednym wykresie
TRAJECTORIES_PER_PLOT = 3

# Kolumny w plikach:
IDEAL_X_COLUMN = "x"
IDEAL_Y_COLUMN = "y"
ACTUAL_X_COLUMN = "x_m"
ACTUAL_Y_COLUMN = "y_m"

MAX_PLOT_POINTS = 12000


# ============================================================
# WCZYTYWANIE DANYCH
# ============================================================

def load_xy(path: Path, x_column: str, y_column: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")

    points = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames or x_column not in reader.fieldnames or y_column not in reader.fieldnames:
            raise ValueError(
                f"Plik {path.name} musi zawierac kolumny "
                f"'{x_column}' i '{y_column}'. Znalezione: {reader.fieldnames}"
            )

        for row in reader:
            try:
                x = float(row[x_column])
                y = float(row[y_column])
            except (TypeError, ValueError):
                continue

            if math.isfinite(x) and math.isfinite(y):
                points.append((x, y))

    data = np.asarray(points, dtype=float)

    if len(data) < 2:
        raise ValueError(f"{path.name}: za malo poprawnych punktow XY.")

    # Usuwamy kolejne identyczne punkty.
    keep = np.ones(len(data), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(data, axis=0), axis=1) > 1e-10
    data = data[keep]

    if len(data) < 2:
        raise ValueError(f"{path.name}: za malo roznych punktow XY.")

    return data


def get_speed_from_filename(path: Path) -> float:
    """
    Odczytuje predkosc z koncowki nazwy:
      trajectory_actual_pid_0_22.csv -> 0.22
      trajectory_actual_pid_1_3.csv  -> 1.3
    """
    #match = re.search(r"trajectory_actual_pid_(\d+)_(\d+)\.csv$", path.name)
    match = re.search(r"trajectory_actual_pp_(\d+)_(\d+)\.csv$", path.name)

    if not match:
        raise ValueError(
            f"Nie mozna odczytac predkosci z nazwy pliku: {path.name}. "
            "Oczekiwany format np. trajectory_actual_pid_0_22.csv"
        )

    return float(f"{match.group(1)}.{match.group(2)}")


# ============================================================
# OBLICZANIE BLEDU WZGledem TRAJEKTORII REFERENCYJNEJ
# ============================================================

def nearest_distances_to_polyline(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Odleglosc kazdego punktu query od najblizszego odcinka reference."""
    starts = reference[:-1]
    vectors = reference[1:] - reference[:-1]
    lengths_sq = np.einsum("ij,ij->i", vectors, vectors)

    # Ochrona przed ewentualnym odcinkiem o zerowej dlugosci.
    valid = lengths_sq > 1e-20
    starts = starts[valid]
    vectors = vectors[valid]
    lengths_sq = lengths_sq[valid]

    if len(starts) == 0:
        raise ValueError("Trajektoria referencyjna nie zawiera poprawnych odcinkow.")

    distances = np.empty(len(query), dtype=float)

    for i, point in enumerate(query):
        rel = point - starts
        t = np.einsum("ij,ij->i", rel, vectors) / lengths_sq
        t = np.clip(t, 0.0, 1.0)

        projections = starts + t[:, None] * vectors
        delta = point - projections
        d2 = np.einsum("ij,ij->i", delta, delta)

        distances[i] = math.sqrt(float(np.min(d2)))

    return distances


def path_length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def calculate_metrics(actual: np.ndarray, ideal: np.ndarray) -> dict:
    errors = nearest_distances_to_polyline(actual, ideal)

    return {
        "points": len(actual),
        "length": path_length(actual),
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95)),
        "max": float(np.max(errors)),
    }


# ============================================================
# POMOCNICZE
# ============================================================

def plot_subset(points: np.ndarray) -> np.ndarray:
    if len(points) <= MAX_PLOT_POINTS:
        return points

    indices = np.linspace(0, len(points) - 1, MAX_PLOT_POINTS).astype(int)
    return points[indices]


def write_summary(results: list, ideal: np.ndarray, ideal_file: Path):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ideal_length = path_length(ideal)

    lines = []
    lines.append("POROWNANIE TRAJEKTORII SAMOCHODU")
    lines.append("=" * 112)
    lines.append(f"Trajektoria referencyjna: {ideal_file.name}")
    lines.append(f"Liczba punktow referencyjnych: {len(ideal)}")
    lines.append(f"Dlugosc trajektorii referencyjnej: {ideal_length:.4f} m")
    lines.append("")
    lines.append(
        f"{'Predkosc':>9} | {'Liczba pkt':>10} | {'Dlugosc [m]':>12} | "
        f"{'MAE [m]':>9} | {'RMSE [m]':>9} | {'Mediana [m]':>12} | "
        f"{'P95 [m]':>9} | {'Max [m]':>9}"
    )
    lines.append("-" * 112)

    for item in results:
        m = item["metrics"]
        lines.append(
            f"{item['speed']:>8.2f} | {m['points']:>10d} | {m['length']:>12.3f} | "
            f"{m['mae']:>9.4f} | {m['rmse']:>9.4f} | {m['median']:>12.4f} | "
            f"{m['p95']:>9.4f} | {m['max']:>9.4f}"
        )

    lines.append("")
    lines.append("Pliki uwzglednione w zestawieniu:")
    for item in results:
        lines.append(f"  {item['file'].name}")

    lines.append("")
    lines.append("Metoda: dla kazdego punktu trajektorii rzeczywistej obliczono")
    lines.append("najmniejsza odleglosc od dowolnego odcinka trajektorii referencyjnej.")
    lines.append("MAE, RMSE, mediana, P95 i maksimum dotycza tych odleglosci.")
    lines.append("Uwaga: jest to blad geometryczny (cross-track), bez uwzgledniania")
    lines.append("kolejnosci przejechania punktow na trasie.")

    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ============================================================
# WYKRESY
# ============================================================

def show_trajectory_plots(ideal: np.ndarray, results: list):
    ideal_plot = plot_subset(ideal)

    for start in range(0, len(results), TRAJECTORIES_PER_PLOT):
        group = results[start:start + TRAJECTORIES_PER_PLOT]

        plt.figure(figsize=(11, 8))
        plt.plot(
            ideal_plot[:, 0],
            ideal_plot[:, 1],
            color="black",
            linewidth=2.5,
            label="Trajektoria referencyjna",
            zorder=5,
        )

        for item in group:
            points = plot_subset(item["points"])
            plt.plot(
                points[:, 0],
                points[:, 1],
                linewidth=1.3,
                label=f"Rzeczywista, v = {item['speed']:.2f} m/s",
            )

        first_speed = group[0]["speed"]
        last_speed = group[-1]["speed"]

        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title(
            f"Porownanie trajektorii - predkosci "
            f"{first_speed:.2f} do {last_speed:.2f} m/s"
        )
        plt.axis("equal")
        plt.grid(True, alpha=0.35)
        plt.legend()
        plt.tight_layout()

    # Pokaz wszystkie utworzone wykresy. Nie zapisujemy ich do plikow.
    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    if not IDEAL_CSV.is_file():
        raise FileNotFoundError(
            f"Brak trajektorii referencyjnej: {IDEAL_CSV}\n"
            "Umiesc track_trajectory.csv w katalogu config."
        )

    actual_files = list(CONFIG_DIR.glob(ACTUAL_PATTERN))

    if not actual_files:
        raise FileNotFoundError(
            f"Nie znaleziono plikow pasujacych do wzorca {ACTUAL_PATTERN}\n"
            f"Katalog: {CONFIG_DIR}"
        )

    ideal = load_xy(IDEAL_CSV, IDEAL_X_COLUMN, IDEAL_Y_COLUMN)

    # Odczyt predkosci i sortowanie numeryczne, np. 0.22, 0.25, ..., 1.30.
    actual_files_with_speed = [
        (get_speed_from_filename(path), path) for path in actual_files
    ]
    actual_files_with_speed.sort(key=lambda item: item[0])

    results = []

    print(f"\nZnaleziono {len(actual_files_with_speed)} trajektorii rzeczywistych.")
    print(f"Trajektoria referencyjna: {IDEAL_CSV.name}")

    for speed, path in actual_files_with_speed:
        actual = load_xy(path, ACTUAL_X_COLUMN, ACTUAL_Y_COLUMN)
        metrics = calculate_metrics(actual, ideal)

        results.append({
            "speed": speed,
            "file": path,
            "points": actual,
            "metrics": metrics,
        })

        print(
            f"v={speed:.2f} m/s | {path.name} | "
            f"RMSE={metrics['rmse']:.4f} m | MAE={metrics['mae']:.4f} m"
        )

    write_summary(results, ideal, IDEAL_CSV)

    print("\n=== PODSUMOWANIE ===")
    print(f"Liczba porownanych trajektorii: {len(results)}")
    print(f"Tabela tekstowa: {SUMMARY_TXT}")
    print("Wykresy zostana wyswietlone; nie beda zapisywane.")

    show_trajectory_plots(ideal, results)


if __name__ == "__main__":
    main()
