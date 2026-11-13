"""Fit k-means on class skill ratings, save an elbow plot, and label k=3 clusters."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

SKILL_COLUMNS = [
    "skill_programming",
    "skill_data_stats",
    "skill_frontend_design",
    "skill_business_pitch",
]
DATA_PATH = Path(__file__).resolve().parent / "class_dataset.csv"
PLOT_PATH = Path(__file__).resolve().parent / "elbow.png"
K_VALUES = range(1, 7)
CHOSEN_K = 3


def main() -> None:
    students = pd.read_csv(DATA_PATH)
    skills = students[SKILL_COLUMNS]
    n_students = len(students)
    print(f"Number of students: {n_students}")

    inertias = []
    for k in K_VALUES:
        model = KMeans(n_clusters=k, n_init=10, random_state=0)
        model.fit(skills)
        inertias.append(model.inertia_)
        print(f"k={k} inertia={model.inertia_:.4f}")

    fig, ax = plt.subplots()
    ax.plot(list(K_VALUES), inertias, marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.set_title("K-means elbow (unscaled skill ratings)")
    ax.set_xticks(list(K_VALUES))
    fig.tight_layout()
    fig.savefig(PLOT_PATH)
    plt.close(fig)

    model = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=0)
    students["cluster"] = model.fit_predict(students[SKILL_COLUMNS])
    counts = students["cluster"].value_counts().sort_index()
    for label, centroid in enumerate(model.cluster_centers_):
        averages = ", ".join(f"{value:.2f}" for value in centroid)
        print(f"Cluster {label}: n={counts[label]} centroid=({averages})")
    students.to_csv(DATA_PATH, index=False)


if __name__ == "__main__":
    main()
