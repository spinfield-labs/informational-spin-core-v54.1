# icoer_visualizer.py
import matplotlib.pyplot as plt

# Pesos padrão da versão 5.2/5.3
FACTOR_WEIGHTS = {
    "S": 0.21,
    "L": 0.20,
    "E": 0.17,
    "C": 0.25,
    "M": 0.10,
    "A": 0.07
}

def plot_icoer_factors(scores: dict, title="I_TGU Factor Contribution"):
    factors = ["S", "L", "E", "C", "M", "A"]
    values = [scores.get(f, 0.0) for f in factors]
    weights = [FACTOR_WEIGHTS[f] for f in factors]
    contributions = [v * w for v, w in zip(values, weights)]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(factors, contributions, color="mediumslateblue")
    plt.title(title)
    plt.ylabel("Weighted Contribution")
    plt.xlabel("Factors")
    plt.ylim(0, max(contributions) + 0.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, val in zip(bars, contributions):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.show()
