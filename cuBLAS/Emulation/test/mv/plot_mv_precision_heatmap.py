"""
plot_mv_precision_heatmap.py

Reads mv_precision_gflops.csv produced by mv_precision_benchmark and draws a
3×2 heatmap grid — one panel per precision — showing GFLOP/s as a function of
matrix dimensions m (rows) and k (columns) for the MV operation y = A*x (n=1).

Usage:
    python3 plot_mv_precision_heatmap.py [csv_file]
    (csv_file defaults to mv_precision_gflops.csv in the current directory)

Output:
    mv_precision_heatmap.png
"""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import LogFormatter


def load_csv(path):
    """Return {precision: {(m, k): gflops}}."""
    data = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            prec = row["precision"]
            m, k = int(row["m"]), int(row["k"])
            data.setdefault(prec, {})[m, k] = float(row["gflops"])
    return data


# Panel order: TC row first, then CUDA-core row
PREC_ORDER = ["INT8_TC", "FP16_TC", "FP32_TC",
              "FP16_CC", "FP32_CC", "FP64_CC"]

PREC_TITLE = {
    "INT8_TC": "INT8 Tensor Core\n[GemmEx, COMPUTE_32I]",
    "FP16_TC": "FP16 Tensor Core\n[Hgemm, n=1]",
    "FP32_TC": "FP32 Tensor Core (TF32)\n[GemmEx, COMPUTE_32F_FAST_TF32]",
    "FP16_CC": "FP16 CUDA cores\n[GemmEx, COMPUTE_16F_PEDANTIC]",
    "FP32_CC": "FP32 CUDA cores\n[cublasSgemv, PEDANTIC_MATH]",
    "FP64_CC": "FP64 CUDA cores\n[cublasDgemv]",
}

CMAP = {
    "INT8_TC": "YlOrRd",
    "FP16_TC": "YlGnBu",
    "FP32_TC": "PuBuGn",
    "FP16_CC": "YlOrBr",
    "FP32_CC": "BuPu",
    "FP64_CC": "BuGn",
}


def fmt_val(v):
    if v <= 0:   return "—"
    if v >= 100: return f"{v:.0f}"
    if v >= 10:  return f"{v:.1f}"
    if v >= 1:   return f"{v:.2f}"
    return f"{v:.3f}"


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "mv_precision_gflops.csv"
    data = load_csv(csv_path)

    all_m = sorted({mk[0] for v in data.values() for mk in v})
    all_k = sorted({mk[1] for v in data.values() for mk in v})
    NM, NK = len(all_m), len(all_k)
    m_labels = [str(v) for v in all_m]
    k_labels = [str(v) for v in all_k]

    mats = {}
    for prec in PREC_ORDER:
        mat = np.zeros((NM, NK))
        for i, m in enumerate(all_m):
            for j, k in enumerate(all_k):
                mat[i, j] = data.get(prec, {}).get((m, k), 0.0)
        mats[prec] = mat

    # 2 rows × 3 columns: top row = TC paths, bottom row = CUDA-core paths
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        "Matrix-Vector Multiply  y = A·x  (n=1)  —  GFLOP/s\n"
        "Top row: Tensor Core paths  |  Bottom row: CUDA core paths  |  "
        "Rows: m (output)   Cols: k (input)   Log colour scale",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for ax, prec in zip(axes.flat, PREC_ORDER):
        mat  = mats[prec]
        cmap = CMAP[prec]

        pos = mat[mat > 0]
        safe = np.where(mat > 0, mat, (pos.min() * 0.1) if pos.size else 1e-3)
        norm = mcolors.LogNorm(vmin=safe.min(), vmax=safe.max())

        im = ax.imshow(safe, cmap=cmap, norm=norm, aspect="auto")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                          format=LogFormatter(base=10, labelOnlyBase=False))
        cb.set_label("GFLOP/s", fontsize=9)

        for i in range(NM):
            for j in range(NK):
                v    = mat[i, j]
                text = fmt_val(v)
                rgba = plt.get_cmap(cmap)(norm(safe[i, j]))
                lum  = rgba[0]*0.299 + rgba[1]*0.587 + rgba[2]*0.114
                fc   = "white" if lum < 0.5 else "black"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=7, color=fc, fontweight="bold")

        ax.set_title(PREC_TITLE[prec], fontsize=10, fontweight="bold", pad=7)
        ax.set_xlabel("k  (input vector length)", fontsize=9)
        ax.set_ylabel("m  (matrix rows)", fontsize=9)
        ax.set_xticks(range(NK)); ax.set_xticklabels(k_labels, fontsize=8, rotation=30)
        ax.set_yticks(range(NM)); ax.set_yticklabels(m_labels, fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = "mv_precision_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
