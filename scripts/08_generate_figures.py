"""Phase 3: 報告圖表生成腳本。

產生 5 張圖表供期末報告使用，存到 experiments/figures/。
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

# 中文字型設定
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

OUT_DIR = project_root / "experiments" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED = project_root / "data" / "processed"
RAW_TAIEX = project_root / "data" / "raw" / "taiex"

TEST_START = "2025-01-01"


def load_data():
    """載入所有需要的資料。"""
    ai = pd.read_csv(PROCESSED / "attention_index.csv", encoding="utf-8-sig")
    bi = pd.read_csv(PROCESSED / "bullish_index.csv", encoding="utf-8-sig")
    pi = pd.read_csv(PROCESSED / "propagation_index.csv", encoding="utf-8-sig")
    taiex = pd.read_csv(RAW_TAIEX / "taiex_daily.csv", encoding="utf-8-sig")
    gs = pd.read_csv(PROCESSED / "grid_search_results.csv")

    for df in [ai, bi, pi, taiex]:
        df["date"] = pd.to_datetime(df["date"])

    return ai, bi, pi, taiex, gs


def fig2_index_timeseries(ai, bi, pi, taiex):
    """圖2: AI/BI/PI 三指標時序圖 + TAIEX 收盤價。"""
    # 合併到共同日期
    merged = ai[["date", "ai_raw"]].merge(
        bi[["date", "bi_raw"]], on="date", how="inner"
    ).merge(
        pi[["date", "pi_raw"]], on="date", how="inner"
    ).merge(
        taiex[["date", "close"]], on="date", how="inner"
    ).sort_values("date")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    test_date = pd.Timestamp(TEST_START)

    # TAIEX
    ax = axes[0]
    ax.plot(merged["date"], merged["close"], color="#333333", linewidth=0.8)
    ax.axvline(test_date, color="red", linestyle="--", alpha=0.7, label="測試期起點")
    ax.set_ylabel("TAIEX 收盤價")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("TAIEX 加權指數與三項情緒指標時序圖", fontsize=13, fontweight="bold")

    # AI
    ax = axes[1]
    ax.plot(merged["date"], merged["ai_raw"], color="#2196F3", linewidth=0.6)
    ax.axvline(test_date, color="red", linestyle="--", alpha=0.7)
    ax.set_ylabel("AI (關注度)")
    ax.fill_between(merged["date"], 0, merged["ai_raw"], alpha=0.15, color="#2196F3")

    # BI
    ax = axes[2]
    ax.plot(merged["date"], merged["bi_raw"], color="#4CAF50", linewidth=0.6)
    ax.axvline(test_date, color="red", linestyle="--", alpha=0.7)
    ax.set_ylabel("BI (看漲指數)")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)

    # PI
    ax = axes[3]
    ax.plot(merged["date"], merged["pi_raw"], color="#FF9800", linewidth=0.6)
    ax.axvline(test_date, color="red", linestyle="--", alpha=0.7)
    ax.set_ylabel("PI (傳播指數)")
    ax.set_xlabel("日期")

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # 標注訓練/測試期
    for ax in axes:
        ylim = ax.get_ylim()
        ax.text(pd.Timestamp("2023-01-01"), ylim[1] * 0.92, "訓練期", fontsize=8, color="gray")
        ax.text(pd.Timestamp("2025-06-01"), ylim[1] * 0.92, "測試期", fontsize=8, color="red", alpha=0.7)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig2_index_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig2_index_timeseries.png")


def fig3_sentiment_space(ai, bi, pi, taiex):
    """圖3: 3D 情緒空間散佈圖（測試期樣本）。"""
    merged = ai[["date", "ai_raw"]].merge(
        bi[["date", "bi_raw"]], on="date"
    ).merge(
        pi[["date", "pi_raw"]], on="date"
    ).merge(
        taiex[["date", "trend_label"]], on="date"
    ).sort_values("date")

    # Z-score（訓練期統計）
    train = merged[merged["date"] < TEST_START]
    for col in ["ai_raw", "bi_raw", "pi_raw"]:
        m, s = train[col].mean(), train[col].std()
        merged[col + "_z"] = (merged[col] - m) / s

    test = merged[merged["date"] >= TEST_START].copy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 漲跌分色
    up = test[test["trend_label"] == 1]
    dn = test[test["trend_label"] == -1]

    ax.scatter(up["ai_raw_z"], up["bi_raw_z"], up["pi_raw_z"],
               c="#E53935", marker="^", s=25, alpha=0.6, label="上漲日")
    ax.scatter(dn["ai_raw_z"], dn["bi_raw_z"], dn["pi_raw_z"],
               c="#1E88E5", marker="v", s=25, alpha=0.6, label="下跌日")

    # 畫一個示意橢球
    center = [test["ai_raw_z"].mean(), test["bi_raw_z"].mean(), test["pi_raw_z"].mean()]
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    rx = test["ai_raw_z"].std() * 1.2
    ry = test["bi_raw_z"].std() * 1.2
    rz = test["pi_raw_z"].std() * 1.2
    x = center[0] + rx * np.outer(np.cos(u), np.sin(v))
    y = center[1] + ry * np.outer(np.sin(u), np.sin(v))
    z = center[2] + rz * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.08, color="gray")
    ax.plot_wireframe(x, y, z, alpha=0.1, color="gray", linewidth=0.3)

    ax.set_xlabel("AI (關注度)", fontsize=10)
    ax.set_ylabel("BI (看漲指數)", fontsize=10)
    ax.set_zlabel("PI (傳播指數)", fontsize=10)
    ax.set_title("測試期情緒空間分佈與 HDS 橢球示意", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.view_init(elev=20, azim=135)

    fig.savefig(OUT_DIR / "fig3_sentiment_space.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig3_sentiment_space.png")


def fig4_grid_search_heatmap(gs):
    """圖4: Grid Search 熱力圖（alpha × theta，分面=window×version）。"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    windows = [20, 40, 60]
    versions = ["baseline", "llm_bi"]
    version_labels = {"baseline": "Baseline", "llm_bi": "LLM-BI"}

    vmin, vmax = 0.3, 0.85

    for i, version in enumerate(versions):
        for j, w in enumerate(windows):
            ax = axes[i][j]
            subset = gs[(gs["version"] == version) & (gs["window"] == w)]

            pivot = subset.pivot_table(
                index="alpha", columns="theta", values="accuracy"
            )

            sns.heatmap(
                pivot, ax=ax, annot=True, fmt=".1%",
                cmap="RdYlGn", vmin=vmin, vmax=vmax,
                cbar=j == 2,
                linewidths=0.5,
            )
            ax.set_title(f"{version_labels[version]} — w={w} ({w//20}M)", fontsize=10)
            ax.set_xlabel("θ (機率門檻)" if i == 1 else "")
            ax.set_ylabel("α (樣本比例)" if j == 0 else "")

    fig.suptitle("Grid Search 準確率熱力圖（α × θ × 視窗 × 版本）", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig4_grid_search_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig4_grid_search_heatmap.png")


def fig5_coverage_accuracy_scatter(gs):
    """圖5: Coverage vs. Accuracy 散佈圖。"""
    fig, ax = plt.subplots(figsize=(9, 6))

    markers = {20: "o", 40: "s", 60: "D"}
    colors = {"baseline": "#78909C", "llm_bi": "#E53935"}

    for version in ["baseline", "llm_bi"]:
        for w in [20, 40, 60]:
            subset = gs[(gs["version"] == version) & (gs["window"] == w)]
            label = f"{'BL' if version == 'baseline' else 'LLM'} w={w}"
            ax.scatter(
                subset["coverage"], subset["accuracy"],
                marker=markers[w],
                c=colors[version],
                s=60, alpha=0.6,
                edgecolors="white", linewidth=0.5,
                label=label,
            )

    # 標注最佳配置
    best_llm = gs[(gs["version"] == "llm_bi") & (gs["coverage"] > 0.05)].nlargest(1, "accuracy").iloc[0]
    best_practical = gs[(gs["version"] == "llm_bi") & (gs["coverage"] > 0.25)].nlargest(1, "accuracy").iloc[0]

    ax.annotate(
        f"最佳準確率\n{best_llm['accuracy']:.1%}",
        xy=(best_llm["coverage"], best_llm["accuracy"]),
        xytext=(best_llm["coverage"] + 0.08, best_llm["accuracy"] - 0.03),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="red", lw=1),
        color="red",
    )
    ax.annotate(
        f"最佳實用\n{best_practical['accuracy']:.1%}/{best_practical['coverage']:.0%}",
        xy=(best_practical["coverage"], best_practical["accuracy"]),
        xytext=(best_practical["coverage"] + 0.05, best_practical["accuracy"] + 0.05),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="red", lw=1),
        color="red",
    )

    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="隨機基準 (50%)")
    ax.set_xlabel("覆蓋率 (Coverage)", fontsize=11)
    ax.set_ylabel("準確率 (Accuracy)", fontsize=11)
    ax.set_title("覆蓋率 vs. 準確率 — Baseline vs. LLM-BI（96 組參數）", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    ax.set_xlim(-0.02, 0.45)
    ax.set_ylim(0.25, 0.90)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig5_coverage_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig5_coverage_accuracy.png")


def main():
    print("=" * 50)
    print("  報告圖表生成")
    print("=" * 50)

    ai, bi, pi, taiex, gs = load_data()

    print("\n[圖2] 三指標時序圖...")
    fig2_index_timeseries(ai, bi, pi, taiex)

    print("[圖3] 3D 情緒空間散佈圖...")
    fig3_sentiment_space(ai, bi, pi, taiex)

    print("[圖4] Grid Search 熱力圖...")
    fig4_grid_search_heatmap(gs)

    print("[圖5] Coverage vs. Accuracy 散佈圖...")
    fig5_coverage_accuracy_scatter(gs)

    print(f"\n所有圖表已儲存至 {OUT_DIR}/")
    print("=" * 50)


if __name__ == "__main__":
    main()