#!/usr/bin/env python3


import csv
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# KONFIGURACJA - UZUPELNIJ DWIE PELNE SCIEZKI DO PLIKOW CSV
# ============================================================

CONFIG_DIR = Path("/home/developer/ros2_ws/src/roboracer_sim/config")

# Zamien koncowe "plik.csv" na nazwy odpowiednich plikow.
IDEAL_CSV = Path("/home/developer/ros2_ws/src/roboracer_sim/config/plik.csv")
ACTUAL_CSV = Path("/home/developer/ros2_ws/src/roboracer_sim/config/plik.csv")

OUTPUT_DIR = CONFIG_DIR / "trajectory_comparison"
X_COLUMN = "x_m"
Y_COLUMN = "y_m"
MAX_PLOT_POINTS = 12000


def load_xy(path: Path, description: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku {description}: {path}")

    points = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or X_COLUMN not in reader.fieldnames or Y_COLUMN not in reader.fieldnames:
            raise ValueError(
                f"{path} musi miec kolumny '{X_COLUMN}' i '{Y_COLUMN}'. "
                f"Znalezione: {reader.fieldnames}"
            )
        for row in reader:
            try:
                x, y = float(row[X_COLUMN]), float(row[Y_COLUMN])
            except (TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                points.append((x, y))

    data = np.asarray(points, dtype=float)
    if len(data) < 2:
        raise ValueError(f"{path}: potrzeba co najmniej dwoch poprawnych punktow XY.")

    # Usuniecie kolejnych identycznych punktow.
    keep = np.ones(len(data), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(data, axis=0), axis=1) > 1e-10
    data = data[keep]
    if len(data) < 2:
        raise ValueError(f"{path}: za malo roznych punktow XY.")
    return data


def cumulative_distance(points: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def nearest_on_polyline(query: np.ndarray, reference: np.ndarray):
    """Najblizszy punkt na dowolnym odcinku trajektorii referencyjnej."""
    starts = reference[:-1]
    vectors = reference[1:] - reference[:-1]
    lengths_sq = np.einsum("ij,ij->i", vectors, vectors)

    projections = np.empty_like(query)
    distances = np.empty(len(query))
    segment_ids = np.empty(len(query), dtype=int)
    signed = np.empty(len(query))

    for i, point in enumerate(query):
        rel = point - starts
        t = np.einsum("ij,ij->i", rel, vectors) / lengths_sq
        t = np.clip(t, 0.0, 1.0)
        candidates = starts + t[:, None] * vectors
        delta = point - candidates
        d2 = np.einsum("ij,ij->i", delta, delta)
        j = int(np.argmin(d2))

        projections[i] = candidates[j]
        distances[i] = math.sqrt(float(d2[j]))
        segment_ids[i] = j

        # Dodatni blad oznacza polozenie po lewej stronie kierunku odcinka idealnego.
        cross = vectors[j, 0] * delta[j, 1] - vectors[j, 1] * delta[j, 0]
        signed[i] = math.copysign(distances[i], cross) if distances[i] > 0 else 0.0

    return projections, distances, segment_ids, signed


def plot_subset(points: np.ndarray) -> np.ndarray:
    if len(points) <= MAX_PLOT_POINTS:
        return points
    return points[np.linspace(0, len(points) - 1, MAX_PLOT_POINTS).astype(int)]


def save_rows(path: Path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    if IDEAL_CSV.resolve() == ACTUAL_CSV.resolve():
        raise ValueError(
            "IDEAL_CSV i ACTUAL_CSV wskazuja ten sam plik. "
            "Wpisz dwie rozne nazwy plikow w konfiguracji programu."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ideal = load_xy(IDEAL_CSV, "trajektorii idealnej")
    actual = load_xy(ACTUAL_CSV, "trajektorii rzeczywistej")

    ideal_s = cumulative_distance(ideal)
    actual_s = cumulative_distance(actual)
    projections, errors, segment_ids, signed = nearest_on_polyline(actual, ideal)

    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    median = float(np.median(errors))
    p95 = float(np.percentile(errors, 95))
    maximum = float(np.max(errors))
    mean_signed = float(np.mean(signed))

    # Szczegolowy wynik dla kazdej probki rzeczywistej trajektorii.
    detail = [
        [
            i, f"{actual_s[i]:.6f}",
            f"{actual[i, 0]:.6f}", f"{actual[i, 1]:.6f}",
            f"{projections[i, 0]:.6f}", f"{projections[i, 1]:.6f}",
            int(segment_ids[i]), f"{errors[i]:.6f}", f"{signed[i]:.6f}",
        ]
        for i in range(len(actual))
    ]
    save_rows(
        OUTPUT_DIR / "errors_by_point.csv",
        ["sample", "actual_distance_m", "actual_x_m", "actual_y_m",
         "nearest_ideal_x_m", "nearest_ideal_y_m", "ideal_segment",
         "cross_track_error_m", "signed_cross_track_error_m"],
        detail,
    )

    summary = [
        ("ideal_point_count", len(ideal)),
        ("actual_point_count", len(actual)),
        ("ideal_path_length_m", f"{ideal_s[-1]:.6f}"),
        ("actual_path_length_m", f"{actual_s[-1]:.6f}"),
        ("MAE_m", f"{mae:.6f}"),
        ("RMSE_m", f"{rmse:.6f}"),
        ("median_error_m", f"{median:.6f}"),
        ("95th_percentile_error_m", f"{p95:.6f}"),
        ("maximum_error_m", f"{maximum:.6f}"),
        ("mean_signed_error_m", f"{mean_signed:.6f}"),
    ]
    save_rows(OUTPUT_DIR / "summary.csv", ["metric", "value"], summary)

    # Wykres 1: przebieg obu trajektorii w ukladzie XY.
    plt.figure(figsize=(10, 8))
    plt.plot(*plot_subset(ideal).T, label="Trajektoria idealna", linewidth=2)
    plt.plot(*plot_subset(actual).T, label="Trajektoria rzeczywista", linewidth=1.2)
    plt.scatter(ideal[0, 0], ideal[0, 1], marker="o", s=55, label="Start idealny")
    plt.scatter(actual[0, 0], actual[0, 1], marker="x", s=65, label="Start rzeczywisty")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.title("Porownanie trajektorii samochodu")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trajectory_comparison.png", dpi=180)
    plt.close()

    # Wykres 2: odleglosc od idealnej trasy w funkcji drogi rzeczywistej.
    plt.figure(figsize=(12, 5))
    plt.plot(actual_s, errors, linewidth=1, label="Blad poprzeczny")
    plt.axhline(mae, linestyle="--", label=f"MAE = {mae:.3f} m")
    plt.axhline(rmse, linestyle=":", label=f"RMSE = {rmse:.3f} m")
    plt.xlabel("Droga przejechana przez samochod [m]")
    plt.ylabel("Odleglosc od trajektorii idealnej [m]")
    plt.title("Blad poprzeczny w funkcji drogi")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cross_track_error.png", dpi=180)
    plt.close()

    # Wykres 3: histogram bledow.
    plt.figure(figsize=(9, 5))
    plt.hist(errors, bins=40, edgecolor="black")
    plt.xlabel("Blad poprzeczny [m]")
    plt.ylabel("Liczba probek")
    plt.title("Rozklad bledu poprzecznego")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "error_histogram.png", dpi=180)
    plt.close()

    print("\n=== POROWNANIE TRAJEKTORII ===")
    print(f"Liczba punktow idealnych:       {len(ideal)}")
    print(f"Liczba punktow rzeczywistych:   {len(actual)}")
    print(f"Dlugosc idealnej trasy:         {ideal_s[-1]:.3f} m")
    print(f"Dlugosc rzeczywistej trasy:     {actual_s[-1]:.3f} m")
    print(f"MAE (sredni blad bezwzgledny):  {mae:.4f} m")
    print(f"RMSE:                           {rmse:.4f} m")
    print(f"Mediana bledu:                  {median:.4f} m")
    print(f"95. percentyl bledu:            {p95:.4f} m")
    print(f"Maksymalny blad:                {maximum:.4f} m")
    print(f"Sredni blad ze znakiem:         {mean_signed:.4f} m")
    print(f"\nWyniki zapisano w: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
