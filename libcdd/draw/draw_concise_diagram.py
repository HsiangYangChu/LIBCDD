import matplotlib.pyplot as plt
import numpy as np

def draw_concise_diagram(x):
    _, ax = plt.subplots(1, 1, figsize=(8, 6))
    t = range(x.size)
    ax.plot(t, x, "b-", lw=0.5)
    ax.set_xlim(-.01 * x.size, x.size * 1.01 - 1)
    ax.set_xlabel("index", fontsize=14)
    ax.set_ylabel("value", fontsize=14)
    ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
    yrange = ymax - ymin if ymax > ymin else 1
    ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
    plt.tight_layout()
    plt.show()