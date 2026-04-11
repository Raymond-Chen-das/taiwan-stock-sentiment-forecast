# 台灣股市情緒空間趨勢預測模型

> **課程專題** — 東吳大學資料科學碩專「時空分析方法應用」

將社群媒體情緒空間模型（Liu et al., 2025, *Knowledge-Based Systems*）應用於台灣加權股價指數 (TAIEX) 趨勢預測，並提出以大型語言模型 (LLM) 增強看漲指數的方案。

## 研究摘要

本研究將三項情緒指標——關注度指數 (AI)、看漲指數 (BI)、傳播指數 (PI)——映射至三維情緒空間，透過粒子群最佳化 (PSO) 搜尋情緒驅動子空間 (SDS) 以預測隔日價格趨勢。在複製過程中發現並修正了 Z-score 資料洩漏與預測目標時間對齊兩個方法論問題，並提出 LLM-BI 增強方案對 PTT 文章進行語義分類），在 96 組參數 Grid Search 中系統性優於 Baseline。

**主要結果：**

| 配置 | 版本 | 覆蓋率 | 準確率 |
|------|------|--------|--------|
| 最佳準確率 (w=40, α=0.25, θ=0.75) | LLM-BI | 6.2% | **81.2%** |
| 同上 | Baseline | 6.1% | 68.8% |
| 最佳實用 (w=20, α=0.20, θ=0.65) | LLM-BI | **30.4%** | **62.0%** |
| 同上 | Baseline | 27.2% | 59.2% |

## 專案結構

```
taiwan-sentiment-forecast/
├── config/
│   └── settings.yaml            # 模型參數與資料來源設定
├── src/
│   ├── collectors/              # 資料收集（TAIEX、Google Trends、PTT）
│   ├── processors/              # 指標建構（AI、BI、PI）與時間對齊
│   ├── models/                  # 情緒空間模型、PSO 搜尋、SDS 偵測、評估器
│   ├── visualization/           # Plotly 視覺化
│   └── utils/                   # 設定載入、交易日曆、日誌
├── scripts/
│   ├── 00_validate_apis.py      # Step 0: API 連線驗證
│   ├── 01_collect_all.py        # Step 1: 批次資料收集
│   ├── 02_build_indices.py      # Step 2: 建構 AI/BI/PI 指標
│   ├── 03_quick_correlation.py  # Step 3: 指標相關性檢驗
│   ├── 04_go_nogo_report.py     # Step 4: Go/No-Go 決策報告
│   ├── 05_run_model.py          # Step 5: Baseline SDS 模型
│   ├── 06_llm_bi_model.py       # Step 6: LLM-BI 增強模型
│   ├── 07_grid_search.py        # Step 7: Alpha/Theta Grid Search
│   └── 08_generate_figures.py   # Step 8: 報告圖表生成
├── experiments/
│   ├── final_report.md          # 期末報告（繁體中文）
│   ├── figures/                 # 報告圖表
│   ├── phase2_baseline_report.md
│   └── go_nogo_report.txt
├── tests/                       # 單元測試
├── requirements.txt
└── README.md
```

## 方法論

### 資料來源

| 指標 | 來源 | 衡量面向 |
|------|------|----------|
| 關注度指數 (AI) | Google Trends | 投資者搜尋行為熱度 |
| 看漲指數 (BI) | PTT Stock 板 | 散戶情緒方向（推噓比 / LLM 分類） |
| 傳播指數 (PI) | PTT Stock 板 | 社群互動強度（PCA 降維） |

- **訓練期**：2021/01/04 – 2024/12/31（970 個交易日）
- **測試期**：2025/01/02 – 2026/01/30（262 個交易日）

### 核心方法

1. **情緒空間建構**：將 AI/BI/PI 經 Z-score 標準化後映射為 3D 空間中的點
2. **HDS 分割**：PSO 搜尋最優橢球半徑，將空間分為內/外子空間
3. **SDS 偵測**：四個子模型（AI×BI / AI×PI / BI×PI / AI×BI×PI）獨立偵測
4. **多重過濾**：Support、Gini 指數、K 近鄰一致性篩選
5. **LLM 增強**：Claude Opus 4.6 對 PTT 文章語義分類，替代測試期推噓比 BI

### 修正的方法論問題

- **Z-score 資料洩漏**：標準化改為僅使用訓練期統計量
- **預測目標時間對齊**：加入 `shift(-1)` 確保用 t 天情緒預測 t+1 天趨勢

## 快速開始

### 環境建置

```bash
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

### 執行流程

```bash
# 資料收集與前處理
python scripts/00_validate_apis.py
python scripts/01_collect_all.py --source all
python scripts/02_build_indices.py

# 模型執行
python scripts/05_run_model.py          # Baseline
python scripts/06_llm_bi_model.py       # LLM-BI（需先準備 llm_sentiment.json）
python scripts/07_grid_search.py        # 參數搜尋（96 組，約 30 分鐘）

# 報告圖表
python scripts/08_generate_figures.py
```

## 參考文獻

Liu, W., Chen, Y., Yang, X., & Wang, J. (2025). Sentiment-driven subspace detection for stock trend prediction via particle swarm optimization. *Knowledge-Based Systems*, 309, 112739.

## 授權

本專案為課堂學術研究用途。
