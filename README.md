# OpenAI Privacy Filter Demo

本專案是 [openai/privacy-filter](https://github.com/openai/privacy-filter)（套件名稱 `opf`）的展示腳本，
用於評估這顆 PII 偵測／遮罩模型的：

- **載入記憶體用量**（process RSS 增量；GPU 模式另量 CUDA allocated）
- **載入耗時**
- **推論耗時**（區分 cold / warm，warm 取多次平均）

並提供：

- `demo.py` — CLI 模式，跑 benchmark 並輸出量測報告
- `app.py` — Streamlit Web UI 模式，互動式輸入文字、查看遮罩結果與偵測到的 PII 區段

推論一律使用模型內建的 **constrained Viterbi decoder**（`decode_mode="viterbi"`）；
若需對照，可切換成 `argmax`（純逐 token argmax）。

---

## 環境需求

| 項目 | 需求 |
|---|---|
| Python | ≥ 3.10 |
| 磁碟空間 | 約 3 GB（模型權重首次下載） |
| 記憶體 | CPU 模式建議 ≥ 12 GB RAM；GPU 模式建議 ≥ 8 GB VRAM |
| GPU（可選） | NVIDIA + CUDA-enabled PyTorch |

### 模型權重路徑

權重存放位置由 `opf` 套件本身決定（不是 demo 程式控制的），優先順序為：

1. CLI 參數 `--checkpoint-dir`（demo 啟動時會 `export OPF_CHECKPOINT=<path>` 給 `opf` 用）
2. 環境變數 `OPF_CHECKPOINT`
3. 預設 `~/.opf/privacy_filter/`

若該目錄不存在或為空，`opf.OPF(...)` 會在初始化時透過 `huggingface_hub` 從
[`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) 自動下載權重（約 3 GB）。
demo 啟動時會印出實際使用的路徑與「found / will auto-download」狀態，方便確認。

---

## 安裝

```bash
git clone https://github.com/KuoZui/openai_privacy_filter_demo.git
cd openai_privacy_filter_demo

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`requirements.txt` 會：

- 從 GitHub 安裝 `opf`（連帶安裝 `torch` / `safetensors` / `huggingface_hub` / `tiktoken` / `numpy`）
- 安裝 `psutil`（量測 RSS）
- 安裝 `streamlit`（Web UI）

---

## 使用方式

### 1. CLI Benchmark 模式（`demo.py`）

跑內建 6 段範例文字（英文 3 + 中文 3），輸出量測報告：

```bash
python demo.py
```

常用選項：

```bash
# 切換到 GPU
python demo.py --device cuda

# 改用 argmax 解碼做對照（預設是 viterbi）
python demo.py --decode-mode argmax

# 自訂單一文字
python demo.py --text "Hi, I'm Alice and my email is alice@example.com."

# 改變 warm 階段的取樣數
python demo.py --warmup-runs 1 --measure-runs 10

# JSON 輸出（方便寫入 log 或用 jq 處理）
python demo.py --json | jq .
```

完整參數：

| 參數 | 預設 | 說明 |
|---|---|---|
| `--device` | `cpu` | `cpu` / `cuda` |
| `--decode-mode` | `viterbi` | `viterbi` / `argmax` |
| `--warmup-runs` | `1` | cold-run 數量（warm 統計從第幾筆開始） |
| `--measure-runs` | `5` | warm-run 數量（取 avg/min/max） |
| `--text` | （無） | 自訂單一文字，提供時會略過內建範例 |
| `--json` | （關閉） | 只輸出 JSON、不印 human report |
| `--checkpoint-dir` | `$OPF_CHECKPOINT` 或 `~/.opf/privacy_filter` | 模型權重資料夾；若不存在會由 `opf` 自動下載 |

#### CLI 輸出範例

```
====================================================================
OpenAI Privacy Filter — Benchmark Report
====================================================================
  device       : cpu
  decode_mode  : viterbi
  load_time    : 18.452 s

Memory
  RSS before   : 412.3 MB
  RSS after    : 6,851.7 MB
  RSS delta    : 6,439.4 MB   <-- model footprint on host

Inference
  cold (1st)   : 4.213 s
  warm avg     : 1.872 s   (min 1.701 s, max 2.014 s, n=5)

Redactions
--------------------------------------------------------------------
[cold #0] 4.213 s
  input    : Hi, I'm Alice Chen. You can reach me at alice.chen@example.com or +1 (415) 555-0142.
  redacted : Hi, I'm [PERSON]. You can reach me at [EMAIL] or [PHONE].
  spans    : 3 found
             - {'type': 'PERSON', 'start': 8, 'end': 18, 'text': 'Alice Chen'}
             - {'type': 'EMAIL', 'start': 41, 'end': 63, 'text': 'alice.chen@example.com'}
             - {'type': 'PHONE', 'start': 67, 'end': 84, 'text': '+1 (415) 555-0142'}
...
```

> 上方數字為示意，實際結果會依機器規格、模型版本、輸入長度而異。

---

### 2. Web UI 模式（`app.py`，Streamlit）

啟動：

```bash
streamlit run app.py
```

預設網址：<http://localhost:8501>

切換 GPU（`--` 之後的參數會傳給 `app.py` 內部的 argparse）：

```bash
streamlit run app.py -- --device cuda
```

#### 介面說明

```
┌── Sidebar ───────────┐  ┌── Main ────────────────────────────────────┐
│ Device: cpu          │  │ [Load time] [RSS delta] [CUDA] [Decode]    │
│ Decode mode: ▼ viterbi│  │ ────────────────────────────────────────  │
│   - viterbi          │  │ [載入英文範例] [載入中文範例]                │
│   - argmax           │  │                                            │
│ [重新載入模型]        │  │ ┌──────────────────────────────────────┐ │
│                      │  │ │ 輸入文字（textarea）                   │ │
└──────────────────────┘  │ └──────────────────────────────────────┘ │
                          │ [Redact]                                   │
                          │                                            │
                          │ ┌─ Redacted text ──┐ ┌─ Detected spans ─┐│
                          │ │ Hi, I'm [PERSON].│ │ type | start | … ││
                          │ │ ...              │ │ EMAIL | 41 | 63  ││
                          │ └──────────────────┘ └──────────────────┘│
                          └────────────────────────────────────────────┘
```

- **頂部指標**：模型載入時間、RSS 增量、CUDA 增量（若啟用 GPU）、目前 decode mode
- **側邊欄 → Decode mode**：可即時切換 `viterbi` / `argmax`，下一次 Redact 生效
- **側邊欄 → 重新載入模型**：清除 `@st.cache_resource` 快取後重新載入（適用切換 device 後）
- **Redact 結果**：左側為遮罩後文字、右側為偵測到的 spans 表格
- 模型只在 session 第一次載入，後續切換 decode mode 或重新整理頁面**不會**重載

---

## Docker 本地執行

模型權重會在 `docker build` 階段直接從 Hugging Face 下載進 image，**不需要先在本機 stage 任何檔案**。

```bash
# 建議先帶 HF_TOKEN 作為 build ARG（避免匿名速率限制，下載快非常多）
docker build --build-arg HF_TOKEN=hf_xxxxxxxxxxxx -t opf-demo .

# 或不設 token（會慢，但仍可運作）
docker build -t opf-demo .

# 啟動
docker run --rm -p 8501:8501 opf-demo
# 開瀏覽器：http://localhost:8501
```

> 首次 build 約 10–20 分鐘：torch 安裝（~2 GB）+ 模型下載（~3 GB）。Build 完 image 約 5 GB；後續只改 `app.py` / `demo.py` 重 build 會命中 layer cache，幾秒內完成。
>
> ⚠️ 用 `--build-arg` 傳 `HF_TOKEN`，token 會存在 image build history（不在最終 image 內，但能透過 `docker history` 看到）。對 demo 用途可接受；要更嚴謹請改用 BuildKit secret（本 Dockerfile 為相容 Railway 的 builder 沒採用）。

---

## 部署到 Railway

1. 在 [Railway](https://railway.app) 建立新 service，選「Deploy from GitHub」並連到本 repo（`main` branch）
2. Railway 會自動偵測 `Dockerfile` 並開始 build；build 階段會把 ~3 GB 模型權重下載進 image
3. **建議設定 HF_TOKEN build arg** 以避免匿名速率限制：
   - Service → Variables → 新增 `HF_TOKEN`，值為 [Hugging Face read token](https://huggingface.co/settings/tokens)
   - Railway 會自動把 service variables 當成 build args 傳給 Dockerfile（`ARG HF_TOKEN`）
   - 沒設也能 build，只是會比較慢
   - ⚠️ Railway 的 builder 不支援 BuildKit `--mount=type=secret`，所以本 Dockerfile 用 `ARG` 接 token；token 會留在 image build history 裡
4. **Build 時間警告**：首次 build 約 10–20 分鐘；之後若只改程式碼會命中 layer cache、重 build 較快
5. **記憶體警告 ⚠️**：1.5B 模型 RSS 約 6–7 GB
   - Railway **Hobby plan**（8 GB）會卡得很緊，可能 OOM
   - 建議升至 **Pro plan**，或改用 Volume 持久化權重 + 不 bake 進 image 的策略（本 demo 沒採用）
6. Service 啟動後 Railway 會給一個 public URL，瀏覽器打開即看到 Streamlit UI

### 環境變數對照表

| 變數 | 階段 | 用途 |
|---|---|---|
| `HF_TOKEN` | **build only** | 透過 `--build-arg HF_TOKEN=...` 傳入（Railway service variable 會自動帶入）；提速 + 避免 rate limit。runtime 不再需要 |
| `PORT` | runtime | Railway 自動注入；Streamlit 會綁這個 port |
| `OPF_CHECKPOINT` | runtime（選用） | 預設 `/root/.opf/privacy_filter`（image 中已含權重）；除非要改路徑否則不用設 |

---

## 量測方式說明

| 量測項目 | 方法 |
|---|---|
| Process RSS | `psutil.Process().memory_info().rss` |
| CUDA allocated | `torch.cuda.memory_allocated()` |
| CUDA reserved（cache pool） | `torch.cuda.memory_reserved()` |
| Load time | `time.perf_counter()` 量 `OPF(...) → get_runtime()` 整段 |
| Cold inference | 模型載入後第一次 `redact()` 呼叫 |
| Warm inference | 第 `warmup-runs+1` 次起的 `redact()` 呼叫，取平均／最小／最大 |

> **為何 cold 與 warm 要分開？**  
> 第一次推論常會包含 PyTorch kernel JIT 編譯、CUDA context 初始化、Python import lazy load 等一次性成本，
> 不能代表穩態效能。把 cold 拆出來看才不會誤判。

---

## Decode mode 說明

`opf` 模型輸出 33 種 token-level 標籤（1 個 background + 8 個 PII 類別 × BIOES 4 種邊界角色）。
從 logits 取出最終 PII span 有兩種策略：

- **`viterbi`（預設）**  
  使用 constrained linear-chain Viterbi decoder，搭配「線性鏈轉移評分」來限制不合法的 BIOES 邊界轉移
  （例如禁止 `O → I-EMAIL`、強制 `B-EMAIL → I-EMAIL` 或 `B-EMAIL → E-EMAIL`），
  最佳化的是「整段路徑」的總分，因此 span 邊界一致性較好。

- **`argmax`**  
  逐 token 獨立 argmax，不考慮上下標籤關聯。實作簡單、計算稍快，但容易出現邊界破碎（例如 BIO 段中間斷掉）。

通常建議使用 `viterbi`；`argmax` 在這個 demo 中保留作為 A/B 對照用。

---

## 已知限制

- **CPU 推論慢**：1.5B 參數的 transformer encoder，CPU 單句推論秒級以上很正常。要做大量處理請用 GPU。
- **記憶體佔用大**：CPU 上 RSS 增量約 6 GB+；GPU 上 VRAM 增量約 3 GB+。
- **語言偏向**：模型主要在英文 PII 資料訓練，中文輸入的偵測效果可能不如英文（特別是中文姓名、地址）。
- **長文 context**：模型支援 128K context，但實際吞吐受裝置記憶體限制，超長輸入請自行分段。
- **Streamlit 切換 device**：在 Web UI 切換 device 必須點側邊欄「重新載入模型」按鈕，因模型用 `@st.cache_resource` 快取住。
- **首次執行需下載模型**：約 3 GB，請保留磁碟空間並注意網路。

---

## 檔案結構

```
openai_privacy_filter_demo/
├── README.md           # 本檔案
├── requirements.txt    # opf + psutil + streamlit
├── demo.py             # CLI benchmark
├── app.py              # Streamlit Web UI
├── Dockerfile          # 部署用容器（Python 3.12-slim，build 時抽模型）
├── .dockerignore       # 排除 .venv 等避免 build context 過大
└── .gitignore          # 排除 .venv / model_cache / 本機 secrets
```

---

## 參考資料

- 模型／套件原始碼：<https://github.com/openai/privacy-filter>
- 模型權重（Hugging Face）：<https://huggingface.co/openai/privacy-filter>
- License：Apache-2.0（依照 `opf` 上游）
