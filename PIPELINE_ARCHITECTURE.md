# Pipeline Architecture: Fine-Tuning Qwen3:4B cho Sentiment Analyst

> **Phiên bản**: v2.0  
> **Cập nhật**: 2026-06-20  
> **Mục tiêu**: Fine-tune model Qwen3:4B local để thay thế quick_think_llm cho Sentiment Analyst agent

---

## 1. Tổng Quan Kiến Trúc

### 1.1 Sơ Đồ Pipeline Tổng Thể

```
╔══════════════════════════════════════════════════════════════════════════╗
║                          PHASE 1-4: LOCAL MACHINE                      ║
║                                                                        ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                    01_collect_data.py                            │    ║
║  │                                                                 │    ║
║  │  ┌──────────┐  ┌──────────────┐  ┌──────────────┐              │    ║
║  │  │ YFinance  │  │  StockTwits   │  │   Reddit     │              │    ║
║  │  │  News API │  │  Public API   │  │  Public API  │              │    ║
║  │  └─────┬─────┘  └──────┬───────┘  └──────┬───────┘              │    ║
║  │        │               │                 │                      │    ║
║  │        └───────────────┼─────────────────┘                      │    ║
║  │                        ▼                                        │    ║
║  │            ┌────────────────────┐                               │    ║
║  │            │ raw_examples.jsonl │  (~1080 examples)             │    ║
║  │            │ + SQLite database  │  (structured storage)         │    ║
║  │            └─────────┬──────────┘                               │    ║
║  └──────────────────────┼──────────────────────────────────────────┘    ║
║                         ▼                                              ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                  01b_generate_golden.py                         │    ║
║  │                                                                 │    ║
║  │    raw_examples.jsonl                                           │    ║
║  │         │                                                       │    ║
║  │         ▼                                                       │    ║
║  │    ┌──────────────────────────────────────┐                     │    ║
║  │    │  gpt-oss-120b @ FPT Cloud            │                     │    ║
║  │    │  mkp-api.fptcloud.com/v1             │                     │    ║
║  │    │  MoE 117B params (5.1B active)       │                     │    ║
║  │    └──────────────┬───────────────────────┘                     │    ║
║  │                   ▼                                             │    ║
║  │    ┌──────────────────────────────┐                             │    ║
║  │    │  golden_examples.jsonl       │                             │    ║
║  │    │  + SQLite (golden_responses) │                             │    ║
║  │    └──────────────┬───────────────┘                             │    ║
║  └───────────────────┼─────────────────────────────────────────────┘    ║
║                      ▼                                                 ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                  02_prepare_dataset.py                           │    ║
║  │                                                                 │    ║
║  │    golden_examples.jsonl                                        │    ║
║  │         │                                                       │    ║
║  │         ├──→ Format ChatML (Qwen3 native)                      │    ║
║  │         ├──→ Validate (5 sections, table, direction)            │    ║
║  │         ├──→ Shuffle (seed=42)                                  │    ║
║  │         │                                                       │    ║
║  │         ▼                                                       │    ║
║  │    ┌────────┐  ┌────────┐  ┌────────┐                          │    ║
║  │    │ train  │  │  val   │  │  test  │                          │    ║
║  │    │ 80%    │  │  10%   │  │  10%   │                          │    ║
║  │    │ ~864   │  │  ~108  │  │  ~108  │                          │    ║
║  │    └────────┘  └────────┘  └────────┘                          │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
║                         │                                              ║
║                   Upload to Google Drive                               ║
╚═════════════════════════╪══════════════════════════════════════════════╝
                          ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                       PHASE 5: GOOGLE COLAB PRO                        ║
║                                                                        ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                     03_train.ipynb                               │    ║
║  │                                                                 │    ║
║  │    Google Drive                                                 │    ║
║  │      │                                                          │    ║
║  │      ├── data/train.jsonl                                       │    ║
║  │      ├── data/val.jsonl                                         │    ║
║  │      │                                                          │    ║
║  │      ▼                                                          │    ║
║  │    ┌──────────────────────────────────────────┐                 │    ║
║  │    │  Unsloth + TRL SFTTrainer                │                 │    ║
║  │    │                                          │                 │    ║
║  │    │  Base: Qwen/Qwen3-4B (HuggingFace)      │                 │    ║
║  │    │  Method: QLoRA 4-bit                     │                 │    ║
║  │    │  LoRA: r=16, alpha=32                    │                 │    ║
║  │    │  GPU: T4/L4/A100 (auto-detect)           │                 │    ║
║  │    │  Epochs: 3                               │                 │    ║
║  │    └──────────────┬───────────────────────────┘                 │    ║
║  │                   │                                             │    ║
║  │                   ▼                                             │    ║
║  │    ┌──────────────────────────────────────────┐                 │    ║
║  │    │  Output (saved to Google Drive):          │                 │    ║
║  │    │  ├── models/lora_adapter/                 │                 │    ║
║  │    │  ├── models/merged/                       │                 │    ║
║  │    │  └── models/gguf/unsloth.Q4_K_M.gguf     │                 │    ║
║  │    └──────────────┬───────────────────────────┘                 │    ║
║  └───────────────────┼─────────────────────────────────────────────┘    ║
║                      │                                                 ║
║                Download GGUF from Drive                                ║
╚══════════════════════╪═════════════════════════════════════════════════╝
                       ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                     PHASE 6-7: LOCAL MACHINE                           ║
║                                                                        ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                    04_evaluate.py                                │    ║
║  │                                                                 │    ║
║  │    test.jsonl                                                   │    ║
║  │      │                                                          │    ║
║  │      ├──→ Baseline (qwen3:4b gốc)  ──→ 108 responses           │    ║
║  │      └──→ Fine-tuned (sentiment-ft) ──→ 108 responses           │    ║
║  │                    │                                            │    ║
║  │                    ▼                                            │    ║
║  │    ┌──────────────────────────────────────────┐                 │    ║
║  │    │  5 Evaluation Metrics:                    │                 │    ║
║  │    │  1. Structure Score (auto)                │                 │    ║
║  │    │  2. ROUGE-1/2/L (auto)                    │                 │    ║
║  │    │  3. Sentiment Accuracy (auto)             │                 │    ║
║  │    │  4. GPT-as-Judge via FPT Cloud (semi)     │                 │    ║
║  │    │  5. End-to-End Pipeline (manual)           │                 │    ║
║  │    └──────────────────────────────────────────┘                 │    ║
║  │                    │                                            │    ║
║  │              Go / No-Go                                         │    ║
║  │    Structure ≥ 0.80 | Accuracy ≥ 70%                            │    ║
║  │    ROUGE-1 ≥ 0.35  | Judge ≥ 3.0/5                             │    ║
║  └──────────────────────┬──────────────────────────────────────────┘    ║
║                         │ (nếu ĐẠT)                                   ║
║                         ▼                                              ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                     Deploy & Integrate                          │    ║
║  │                                                                 │    ║
║  │    GGUF ──→ ollama create sentiment-analyst-ft                  │    ║
║  │                                                                 │    ║
║  │    .env: TRADINGAGENTS_SENTIMENT_LLM=sentiment-analyst-ft       │    ║
║  │                                                                 │    ║
║  │    ┌──────────────────────────────────────────────────┐         │    ║
║  │    │  CryptoAgents Pipeline (modified)                 │         │    ║
║  │    │                                                   │         │    ║
║  │    │  Market Analyst ──→ qwen3:4b (unchanged)          │         │    ║
║  │    │  Sentiment Analyst ──→ sentiment-analyst-ft ★ NEW │         │    ║
║  │    │  News Analyst ──→ qwen3:4b (unchanged)            │         │    ║
║  │    │  Fundamentals ──→ qwen3:4b (unchanged)            │         │    ║
║  │    │  Quantitative ──→ qwen3:4b (unchanged)            │         │    ║
║  │    └──────────────────────────────────────────────────┘         │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 2. Thành Phần Chi Tiết

### 2.1 Data Sources (Input)

| # | Nguồn | API | Dữ liệu thu được | Rate Limit |
|---|---|---|---|---|
| 1 | **Yahoo Finance News** | `yfinance` Python lib | Tiêu đề + summary + source + link | ~Unlimited (thông qua yf library) |
| 2 | **StockTwits** | `api.stocktwits.com` (public, no key) | Messages + Bullish/Bearish labels + user + timestamp | ~30 req/min |
| 3 | **Reddit** | `reddit.com/r/{sub}/search.json` (public) | Posts + scores + comments + body excerpts | ~10 req/min/IP |

### 2.2 Teacher Model (Golden Response Generator)

| Thuộc tính | Giá trị |
|---|---|
| **Model** | `gpt-oss-120b` (OpenAI open-weight MoE) |
| **Host** | FPT Cloud Infrastructure |
| **Endpoint** | `https://mkp-api.fptcloud.com/v1/chat/completions` |
| **Auth** | Bearer token (env: `FCI_API_KEY`) |
| **Params** | `temperature=0.3`, `max_tokens=3000`, `top_p=0.9` |
| **Architecture** | MoE 117B total params, 5.1B active per forward pass |
| **License** | Apache 2.0 |

### 2.3 Student Model (Fine-Tuning Target)

| Thuộc tính | Giá trị |
|---|---|
| **Base Model** | `Qwen/Qwen3-4B` (HuggingFace) |
| **Method** | QLoRA (4-bit quantization + LoRA adapters) |
| **LoRA Config** | r=16, alpha=32, dropout=0.05 |
| **Target Modules** | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| **Training** | Unsloth + TRL SFTTrainer |
| **Platform** | Google Colab Pro (T4/L4/A100) |
| **Output Format** | GGUF (q4_k_m) → Ollama |

### 2.4 Evaluation Pipeline

```
Test Set (108 examples)
    │
    ├──→ qwen3:4b (baseline) ──→ 108 responses ──┐
    │                                              │
    └──→ sentiment-analyst-ft ──→ 108 responses ──┤
                                                   ▼
                                          ┌─────────────────┐
                                          │ Compare Metrics  │
                                          │                  │
                                          │ 1. Structure     │
                                          │ 2. ROUGE         │
                                          │ 3. Accuracy      │
                                          │ 4. GPT-Judge     │
                                          │ 5. End-to-End    │
                                          └─────────────────┘
```

---

## 3. Cấu Trúc Thư Mục

```
CryptoAgents/
└── finetune/                                    # ← Thư mục chứa TẤT CẢ files fine-tuning
    ├── PIPELINE_ARCHITECTURE.md                 # Tài liệu kiến trúc (file này)
    │
    ├── scripts/                                 # Tất cả scripts thực thi
    │   ├── 01_collect_data.py                   # Thu thập raw data (local)
    │   ├── 01b_generate_golden.py               # Sinh golden responses (FPT Cloud API)
    │   ├── 02_prepare_dataset.py                # Format + split dataset
    │   ├── 03_train.ipynb                       # Training notebook (Colab Pro)
    │   ├── 04_evaluate.py                       # Đánh giá baseline vs fine-tuned
    │   └── 05_export_ollama.py                  # Export GGUF + deploy Ollama
    │
    ├── config/                                  # Cấu hình
    │   ├── tickers.py                           # Danh sách 60 tickers (30 crypto + 30 stock)
    │   ├── Modelfile                            # Ollama Modelfile
    │   └── requirements.txt                     # Dependencies cho local scripts
    │
    ├── data/                                    # Dữ liệu
    │   ├── raw/                                 # Raw data từ APIs
    │   │   └── raw_examples.jsonl
    │   ├── golden/                              # Golden responses từ gpt-oss-120b
    │   │   └── golden_examples.jsonl
    │   ├── db/                                  # SQLite database (structured storage)
    │   │   └── finetune_data.db                 # ← Lưu trữ dạng SQL table
    │   ├── train.jsonl                          # Training set (80%)
    │   ├── val.jsonl                            # Validation set (10%)
    │   └── test.jsonl                           # Test set (10%)
    │
    ├── models/                                  # Model outputs
    │   ├── lora_adapter/                        # LoRA weights sau training
    │   ├── merged/                              # Full model merged
    │   └── gguf/                                # GGUF cho Ollama
    │
    └── eval_results/                            # Kết quả đánh giá
        ├── baseline_qwen3_4b.json
        └── finetuned_results.json
```

---

## 4. SQL Schema — Lưu Trữ Dữ Liệu Có Cấu Trúc

Tất cả dữ liệu thu thập được sẽ lưu song song vào **SQLite database** (`finetune/data/db/finetune_data.db`) bên cạnh JSONL files. Đây là schema chi tiết:

### 4.1 Bảng `raw_examples` — Dữ liệu thu thập từ APIs

```sql
CREATE TABLE IF NOT EXISTS raw_examples (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT NOT NULL,               -- VD: "BTC-USD", "NVDA"
    trade_date      TEXT NOT NULL,               -- VD: "2026-06-15" (YYYY-MM-DD)
    asset_type      TEXT NOT NULL DEFAULT 'stock', -- "crypto" hoặc "stock"
    
    -- Dữ liệu từ 3 nguồn
    news_block      TEXT,                        -- Raw text từ Yahoo Finance News
    stocktwits_block TEXT,                       -- Raw text từ StockTwits API
    reddit_block    TEXT,                        -- Raw text từ Reddit API
    
    -- System message đã build (giống hệt agent thật)
    system_message  TEXT NOT NULL,               -- Full system prompt
    
    -- Metadata
    news_article_count    INTEGER DEFAULT 0,     -- Số bài báo trong news_block
    stocktwits_msg_count  INTEGER DEFAULT 0,     -- Số messages StockTwits
    stocktwits_bullish    INTEGER DEFAULT 0,     -- Số messages Bullish
    stocktwits_bearish    INTEGER DEFAULT 0,     -- Số messages Bearish
    reddit_post_count     INTEGER DEFAULT 0,     -- Số posts Reddit
    
    -- Tracking
    collected_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(ticker, trade_date)                   -- Mỗi ticker+date chỉ 1 record
);

-- Index cho query nhanh
CREATE INDEX idx_raw_ticker ON raw_examples(ticker);
CREATE INDEX idx_raw_date ON raw_examples(trade_date);
CREATE INDEX idx_raw_asset_type ON raw_examples(asset_type);
```

### 4.2 Bảng `golden_responses` — Golden responses từ gpt-oss-120b

```sql
CREATE TABLE IF NOT EXISTS golden_responses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_example_id  INTEGER NOT NULL,            -- FK → raw_examples.id
    ticker          TEXT NOT NULL,
    trade_date      TEXT NOT NULL,
    
    -- Golden response
    golden_response TEXT NOT NULL,                -- Full response từ gpt-oss-120b
    
    -- Phân tích tự động từ golden response
    sentiment_direction TEXT,                    -- "Bullish" / "Bearish" / "Neutral" / "Mixed"
    has_source_breakdown BOOLEAN DEFAULT 0,      -- Có section breakdown không
    has_divergence_analysis BOOLEAN DEFAULT 0,   -- Có phân tích divergence không
    has_catalysts_risks BOOLEAN DEFAULT 0,       -- Có catalysts/risks không
    has_markdown_table BOOLEAN DEFAULT 0,        -- Có markdown table không
    structure_score REAL DEFAULT 0,              -- 0.0 → 1.0 (đủ mấy sections)
    
    -- API metadata
    model_used      TEXT DEFAULT 'gpt-oss-120b', -- Model đã sinh response
    api_endpoint    TEXT DEFAULT 'mkp-api.fptcloud.com',
    tokens_input    INTEGER DEFAULT 0,           -- Input tokens
    tokens_output   INTEGER DEFAULT 0,           -- Output tokens
    latency_ms      INTEGER DEFAULT 0,           -- Thời gian response (ms)
    
    -- Tracking
    generated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (raw_example_id) REFERENCES raw_examples(id)
);

CREATE INDEX idx_golden_ticker ON golden_responses(ticker);
CREATE INDEX idx_golden_direction ON golden_responses(sentiment_direction);
```

### 4.3 Bảng `dataset_splits` — Phân chia Train/Val/Test

```sql
CREATE TABLE IF NOT EXISTS dataset_splits (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    golden_id       INTEGER NOT NULL,            -- FK → golden_responses.id
    ticker          TEXT NOT NULL,
    trade_date      TEXT NOT NULL,
    split           TEXT NOT NULL,               -- "train" / "val" / "test"
    
    -- ChatML formatted message (full prompt + response)
    system_content  TEXT NOT NULL,               -- System message
    user_content    TEXT NOT NULL DEFAULT 'Continue',
    assistant_content TEXT NOT NULL,             -- Golden response
    
    -- Token counts
    total_tokens    INTEGER DEFAULT 0,           -- Tổng tokens (approx)
    
    FOREIGN KEY (golden_id) REFERENCES golden_responses(id)
);

CREATE INDEX idx_split ON dataset_splits(split);
```

### 4.4 Bảng `eval_results` — Kết quả đánh giá

```sql
CREATE TABLE IF NOT EXISTS eval_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT NOT NULL,               -- "qwen3:4b" hoặc "sentiment-analyst-ft"
    test_example_id INTEGER NOT NULL,            -- FK → dataset_splits.id (split="test")
    ticker          TEXT NOT NULL,
    trade_date      TEXT NOT NULL,
    
    -- Model response
    model_response  TEXT NOT NULL,               -- Response từ model đang eval
    
    -- Metric scores
    structure_score REAL,                        -- 0.0 → 1.0
    rouge1_f        REAL,                        -- ROUGE-1 F1
    rouge2_f        REAL,                        -- ROUGE-2 F1
    rougeL_f        REAL,                        -- ROUGE-L F1
    sentiment_direction_pred TEXT,               -- Predicted direction
    sentiment_direction_ref  TEXT,               -- Reference direction
    sentiment_match BOOLEAN,                     -- pred == ref?
    
    -- GPT-as-Judge scores (nullable — chỉ chạy trên sample)
    judge_accuracy      REAL,                    -- 1-5 scale
    judge_evidence      REAL,
    judge_structure     REAL,
    judge_actionability REAL,
    judge_nuance        REAL,
    judge_average       REAL,
    
    -- Tracking
    evaluated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_eval_model ON eval_results(model_name);
```

### 4.5 Queries Hữu Ích

```sql
-- Tổng quan dataset
SELECT asset_type, COUNT(*) as count,
       AVG(stocktwits_msg_count) as avg_twits,
       AVG(reddit_post_count) as avg_reddit
FROM raw_examples GROUP BY asset_type;

-- Phân bố sentiment direction trong golden responses
SELECT sentiment_direction, COUNT(*) as count,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM golden_responses), 1) as pct
FROM golden_responses GROUP BY sentiment_direction;

-- So sánh baseline vs fine-tuned
SELECT model_name,
       ROUND(AVG(structure_score), 3) as avg_structure,
       ROUND(AVG(rouge1_f), 3) as avg_rouge1,
       ROUND(AVG(CASE WHEN sentiment_match THEN 1.0 ELSE 0.0 END), 3) as accuracy,
       ROUND(AVG(judge_average), 2) as avg_judge
FROM eval_results GROUP BY model_name;

-- Tickers có sentiment direction biased
SELECT ticker, sentiment_direction, COUNT(*) as count
FROM golden_responses
GROUP BY ticker, sentiment_direction
ORDER BY ticker, count DESC;
```

---

## 5. Danh Sách 60 Tickers (30 Crypto + 30 Stock)

### 5.1 Crypto Tickers (30)

| # | Ticker | Tên | Market Cap Tier |
|---|---|---|---|
| 1 | BTC-USD | Bitcoin | Large Cap |
| 2 | ETH-USD | Ethereum | Large Cap |
| 3 | BNB-USD | Binance Coin | Large Cap |
| 4 | SOL-USD | Solana | Large Cap |
| 5 | XRP-USD | Ripple | Large Cap |
| 6 | ADA-USD | Cardano | Large Cap |
| 7 | DOGE-USD | Dogecoin | Large Cap |
| 8 | AVAX-USD | Avalanche | Mid Cap |
| 9 | DOT-USD | Polkadot | Mid Cap |
| 10 | LINK-USD | Chainlink | Mid Cap |
| 11 | MATIC-USD | Polygon | Mid Cap |
| 12 | UNI-USD | Uniswap | Mid Cap |
| 13 | ATOM-USD | Cosmos | Mid Cap |
| 14 | LTC-USD | Litecoin | Mid Cap |
| 15 | NEAR-USD | NEAR Protocol | Mid Cap |
| 16 | ARB-USD | Arbitrum | Mid Cap |
| 17 | OP-USD | Optimism | Mid Cap |
| 18 | APT-USD | Aptos | Mid Cap |
| 19 | SUI-USD | Sui | Mid Cap |
| 20 | FIL-USD | Filecoin | Mid Cap |
| 21 | HBAR-USD | Hedera | Mid Cap |
| 22 | ICP-USD | Internet Computer | Mid Cap |
| 23 | RENDER-USD | Render | Small Cap |
| 24 | INJ-USD | Injective | Small Cap |
| 25 | TIA-USD | Celestia | Small Cap |
| 26 | SEI-USD | Sei | Small Cap |
| 27 | FET-USD | Fetch.ai | Small Cap (AI) |
| 28 | PEPE-USD | Pepe | Small Cap (Meme) |
| 29 | WIF-USD | dogwifhat | Small Cap (Meme) |
| 30 | BONK-USD | Bonk | Small Cap (Meme) |

### 5.2 Stock Tickers (30)

| # | Ticker | Tên | Sector |
|---|---|---|---|
| 1 | NVDA | NVIDIA | Semiconductors |
| 2 | AAPL | Apple | Tech Hardware |
| 3 | MSFT | Microsoft | Software |
| 4 | GOOGL | Alphabet | Internet Services |
| 5 | AMZN | Amazon | E-Commerce/Cloud |
| 6 | TSLA | Tesla | EVs/Energy |
| 7 | META | Meta Platforms | Social Media |
| 8 | AMD | AMD | Semiconductors |
| 9 | INTC | Intel | Semiconductors |
| 10 | NFLX | Netflix | Streaming |
| 11 | CRM | Salesforce | Enterprise Software |
| 12 | ORCL | Oracle | Enterprise Software |
| 13 | SHOP | Shopify | E-Commerce |
| 14 | SQ | Block (Square) | Fintech |
| 15 | COIN | Coinbase | Crypto Exchange |
| 16 | MARA | Marathon Digital | Crypto Mining |
| 17 | RIOT | Riot Platforms | Crypto Mining |
| 18 | MSTR | MicroStrategy | Bitcoin Treasury |
| 19 | PLTR | Palantir | AI/Data Analytics |
| 20 | SNOW | Snowflake | Cloud Data |
| 21 | UBER | Uber | Ride-sharing |
| 22 | ABNB | Airbnb | Travel/Hospitality |
| 23 | SOFI | SoFi Technologies | Fintech |
| 24 | ARM | Arm Holdings | Semiconductors |
| 25 | SMCI | Super Micro | AI Infrastructure |
| 26 | CRWD | CrowdStrike | Cybersecurity |
| 27 | NET | Cloudflare | Cloud/CDN |
| 28 | DKNG | DraftKings | Online Gambling |
| 29 | RBLX | Roblox | Gaming/Metaverse |
| 30 | PATH | UiPath | AI/Automation |

### 5.3 Tại sao chọn các tickers này?

- **Đa dạng market cap**: Large/Mid/Small cap → model không bias vào chỉ large caps
- **Đa dạng sector**: Tech, Finance, Crypto, Gaming, Energy → generalize tốt
- **Đa dạng sentiment patterns**: Meme coins (PEPE, WIF, BONK) có sentiment cực đoan; Blue chips (AAPL, MSFT) có sentiment ổn định
- **StockTwits/Reddit coverage**: Tất cả 60 tickers đều có discussion volume đáng kể trên social media
- **Crypto-related stocks**: COIN, MARA, RIOT, MSTR → cầu nối giữa crypto và stock sentiment

**Ước tính dataset**: 60 tickers × 18 tuần (Jan-Jun 2026) = **~1080 examples**

---

## 6. Phân Tích Ngôn Ngữ: Tiếng Việt vs Tiếng Anh

### 6.1 Hiện Trạng Trong Hệ Thống

Hệ thống CryptoAgents có sẵn cơ chế multi-language qua `output_language` config:

```python
# tradingagents/agents/utils/agent_utils.py (dòng 27-40)
def get_language_instruction() -> str:
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""  # Không thêm instruction → mặc định English
    return f" Write your entire response in {lang}."
```

Instruction này được append vào **cuối system prompt** của sentiment_analyst.py (dòng 162):
```python
{get_language_instruction()}  # → " Write your entire response in Vietnamese."
```

### 6.2 So Sánh Ảnh Hưởng Đến Chất Lượng Fine-Tuning

| Tiêu chí | Tiếng Anh | Tiếng Việt |
|---|---|---|
| **Training data quality** | ✅ Tốt nhất — gpt-oss-120b được train chủ yếu bằng English | 🟡 Tốt — Qwen3 hỗ trợ Vietnamese nhưng golden responses có thể kém hơn một chút |
| **Input data language** | ✅ Match — News/StockTwits/Reddit đều bằng English | ⚠️ Mismatch — Input (EN) → Output (VI) = cross-lingual, khó hơn cho model nhỏ |
| **Qwen3 Vietnamese capability** | N/A | ✅ Tốt — Qwen3 có Vietnamese trong pre-training data |
| **Tokenizer efficiency** | ✅ Tối ưu — ít tokens per word | 🟡 Kém hơn ~20-30% — tiếng Việt cần nhiều tokens hơn per word |
| **Downstream compatibility** | ✅ Bull/Bear Researcher đọc report EN tự nhiên | ⚠️ Researchers nhận report tiếng Việt nhưng system prompt của chúng là EN → confusion tiềm ẩn |
| **Evaluation metrics** | ✅ ROUGE/GPT-Judge hoạt động tốt | 🟡 ROUGE kém chính xác hơn cho Vietnamese (word boundary khác) |
| **Token budget** | ✅ ~1000-2000 tokens/response | ⚠️ ~1200-2600 tokens/response (+20-30%) |

### 6.3 Phân Tích Rủi Ro Tiếng Việt

**Rủi ro 1: Cross-lingual task tăng độ khó fine-tuning**

```
Input (English):
  "Bullish: 18 (60%) · Bearish: 7 (23%)..."
  "[2026-06-18 · @user1 · Bullish] BTC looking strong..."

Output yêu cầu (Vietnamese):
  "Hướng sentiment tổng thể: Tăng giá (Bullish)
   Tỷ lệ tích cực trên StockTwits là 60%, cho thấy..."
```

Model 4B phải vừa **hiểu English input** vừa **sinh Vietnamese output** → đây là cross-lingual summarization, khó hơn đáng kể so với monolingual (EN→EN).

**Rủi ro 2: Downstream agents bị confused**

```
Sentiment Report (Vietnamese):
  "Hướng tổng thể: Tăng giá với độ tin cậy cao..."
  
Bull Researcher prompt (English):
  "Based on the sentiment report, argue for buying..."
  → Researcher phải đọc hiểu tiếng Việt → có thể hiểu sai
```

**Rủi ro 3: Golden responses tiếng Việt kém hơn**

`gpt-oss-120b` sinh Vietnamese responses với chất lượng thấp hơn English ~10-15% (dựa trên benchmarks), đặc biệt ở:
- Thuật ngữ tài chính (có thể dùng sai hoặc không nhất quán)
- Formatting markdown table (có thể lỗi)

### 6.4 Khuyến Nghị

> **KHUYẾN NGHỊ: Giữ tiếng Anh cho fine-tuning**

**Lý do:**
1. **Input = English** (News, StockTwits, Reddit đều EN) → output EN = monolingual task, dễ hơn cho model 4B
2. **Downstream agents** đều hoạt động bằng English → sentiment report EN tích hợp tự nhiên
3. **Golden responses** chất lượng cao nhất khi EN→EN
4. **Token efficiency** tiết kiệm ~20-30% budget

**Nếu vẫn muốn tiếng Việt**: Có thể thực hiện được nhưng cần:
- Tăng training data lên ~1500+ examples (bù cho cross-lingual difficulty)
- Tăng LoRA rank lên r=32 (tăng model capacity)
- Kiểm tra golden responses tiếng Việt kỹ hơn (manual review 10-15%)
- Chấp nhận ROUGE scores thấp hơn ~15-20% so với EN baseline
- Có thể cần tăng `max_seq_length` lên 5120 (tiếng Việt tốn tokens hơn)

---

## 7. Timeline Thực Thi

| Phase | Công việc | Thời gian | Chạy ở đâu |
|---|---|---|---|
| 1 | Chuẩn bị cấu trúc + config | ~15 phút | Local |
| 2 | Thu thập raw data (60 tickers × 18 tuần) | ~4-5 giờ | Local |
| 3 | Sinh golden responses (gpt-oss-120b) | ~1-2 giờ | Local → FPT Cloud |
| 4 | Format + split dataset | ~15 phút | Local |
| — | Upload data lên Google Drive | ~10 phút | Local |
| 5 | Training QLoRA trên Colab | ~2-4 giờ | Colab Pro |
| — | Download GGUF từ Drive | ~10 phút | Local |
| 6 | Evaluation (baseline vs fine-tuned) | ~1 giờ | Local |
| 7 | Deploy Ollama + integrate | ~15 phút | Local |
| **Tổng** | | **~10-13 giờ** | |

> **Gợi ý**: Chạy Phase 2-3 ban đêm (automated), Phase 5 (Colab) vào sáng hôm sau.

---

## 8. Quy Trình Đánh Giá Chi Tiết: Baseline vs Fine-Tuned

### 8.1 Tổng Quan Flow Đánh Giá

```
                    ┌──────────────────────┐
                    │  BƯỚC 1: BASELINE    │
                    │  Đánh giá qwen3:4b   │
                    │  (TRƯỚC fine-tuning)  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  BƯỚC 2: TRAINING    │
                    │  Fine-tune QLoRA     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  BƯỚC 3: FINE-TUNED  │
                    │  Đánh giá model mới  │
                    │  (SAU fine-tuning)    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  BƯỚC 4: SO SÁNH     │
                    │  Baseline vs FT      │
                    └──────────┬───────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐   ┌──────────────┐
        │  ĐẠT ✅  │   │ CẢI THIỆN│   │ THẤT BẠI ❌  │
        │          │   │ KHÔNG    │   │              │
        │ Deploy   │   │ ĐÁNG KỂ │   │ FT ≤ Base    │
        │ Ollama   │   │ 🟡      │   │              │
        └──────────┘   └────┬─────┘   └──────┬───────┘
                            │                │
                            └────────┬───────┘
                                     ▼
                          ┌──────────────────┐
                          │  BƯỚC 5:         │
                          │  CHẨN ĐOÁN       │
                          │  ROOT CAUSE      │
                          └────────┬─────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │  BƯỚC 6:         │
                          │  CẢI THIỆN       │
                          │  & THỬ LẠI       │
                          └──────────────────┘
```

### 8.2 BƯỚC 1: Đánh Giá Baseline (TRƯỚC Fine-Tuning)

**Mục đích**: Thiết lập mốc so sánh — model gốc `qwen3:4b` hoạt động tốt đến đâu khi làm Sentiment Analyst.

**Chạy trên tập nào**: `test.jsonl` (~108 examples) — CÙNG tập test sẽ dùng cho fine-tuned model.

**Cách chạy**:
```python
# Chạy qwen3:4b gốc (chưa fine-tune) qua Ollama
for example in test_set:
    prompt = example["messages"][0]["content"]  # system message
    response = ollama.generate(model="qwen3:4b", prompt=prompt)
    # Lưu response + tính metrics
```

**Metrics đánh giá (5 metrics)**:

| # | Metric | Đo cái gì | Cách tính | Tập test |
|---|---|---|---|---|
| M1 | **Structure Score** | Output có đủ 5 sections không | Kiểm tra 5 keywords/patterns, score = (số sections đạt / 5) | test.jsonl (108 ex) |
| M2 | **ROUGE-1/2/L F1** | Output giống golden response đến đâu | So sánh n-gram overlap giữa response vs golden | test.jsonl (108 ex) |
| M3 | **Sentiment Direction Accuracy** | Phân loại đúng Bullish/Bearish/Neutral/Mixed | Extract direction từ cả response và golden, so sánh | test.jsonl (108 ex) |
| M4 | **GPT-as-Judge Score** | Chất lượng tổng thể (5 tiêu chí) | gpt-oss-120b chấm điểm 1-5 cho: Accuracy, Evidence, Structure, Actionability, Nuance | 20 random samples từ test |
| M5 | **Inference Latency** | Tốc độ response | Đo thời gian từ send prompt → receive full response | test.jsonl (108 ex) |

**Output**: `eval_results/baseline_qwen3_4b.json`

```json
{
    "model": "qwen3:4b",
    "test_set_size": 108,
    "metrics": {
        "structure_score_avg": 0.49,
        "rouge1_avg": 0.28,
        "rouge2_avg": 0.10,
        "rougeL_avg": 0.23,
        "sentiment_accuracy": 0.50,
        "gpt_judge_avg": 2.3,
        "latency_avg_ms": 4500
    },
    "per_example_results": [...]
}
```

### 8.3 BƯỚC 3: Đánh Giá Fine-Tuned (SAU Fine-Tuning)

**Chạy trên CÙNG tập test**: `test.jsonl` (~108 examples) — đảm bảo so sánh apple-to-apple.

**Cách chạy**:
```python
# Chạy model fine-tuned qua Ollama
for example in test_set:
    prompt = example["messages"][0]["content"]
    response = ollama.generate(model="sentiment-analyst-ft", prompt=prompt)
    # Lưu response + tính CÙNG metrics
```

**Output**: `eval_results/finetuned_results.json` (cùng format với baseline)

### 8.4 BƯỚC 4: So Sánh & Phân Loại Kết Quả

**Bảng so sánh sẽ được in ra console và lưu vào SQLite**:

```
╔══════════════════════════════════════════════════════════════════╗
║              EVALUATION COMPARISON REPORT                        ║
╠══════════════════╦════════════╦════════════╦═══════╦═════════════╣
║ Metric           ║ Baseline   ║ Fine-Tuned ║ Δ     ║ Verdict     ║
╠══════════════════╬════════════╬════════════╬═══════╬═════════════╣
║ Structure Score  ║ 0.49       ║ ???        ║ ???   ║ ???         ║
║ ROUGE-1 F1       ║ 0.28       ║ ???        ║ ???   ║ ???         ║
║ ROUGE-2 F1       ║ 0.10       ║ ???        ║ ???   ║ ???         ║
║ ROUGE-L F1       ║ 0.23       ║ ???        ║ ???   ║ ???         ║
║ Sent. Accuracy   ║ 0.50       ║ ???        ║ ???   ║ ???         ║
║ GPT-Judge Avg    ║ 2.3        ║ ???        ║ ???   ║ ???         ║
║ Latency (ms)     ║ 4500       ║ ???        ║ ???   ║ ???         ║
╚══════════════════╩════════════╩════════════╩═══════╩═════════════╝
```

**3 loại kết quả có thể xảy ra**:

| Loại | Điều kiện | Hành động |
|---|---|---|
| ✅ **ĐẠT** | Tất cả 4 Go/No-Go criteria đạt ngưỡng VÀ cải thiện ≥15% trung bình | → Deploy lên Ollama |
| 🟡 **CẢI THIỆN KHÔNG ĐÁNG KỂ** | Đạt Go/No-Go NHƯNG cải thiện <15% trung bình | → Chẩn đoán & cải thiện (Bước 5) |
| ❌ **THẤT BẠI** | Fine-tuned ≤ Baseline ở BẤT KỲ metric nào HOẶC không đạt Go/No-Go | → Chẩn đoán & cải thiện (Bước 5) |

**Go/No-Go Criteria (ngưỡng tối thiểu cho fine-tuned model)**:

| # | Metric | Ngưỡng | Tại sao |
|---|---|---|---|
| G1 | Structure Score | ≥ 0.80 | Output phải có ≥ 4/5 sections — đây là lý do chính fine-tune |
| G2 | Sentiment Direction Accuracy | ≥ 70% | Phải phân loại đúng hướng cơ bản |
| G3 | ROUGE-1 F1 | ≥ 0.35 | Phải có overlap hợp lý với golden reference |
| G4 | GPT-Judge Average | ≥ 3.0/5 | Phải đạt "acceptable" theo GPT đánh giá |

**Ngưỡng "cải thiện đáng kể" (so với baseline)**:

| Metric | Cải thiện tối thiểu | Mong đợi |
|---|---|---|
| Structure Score | ≥ +0.25 (≥+50%) | +0.45 (+92%) |
| ROUGE-1 F1 | ≥ +0.10 (≥+35%) | +0.22 (+79%) |
| Sentiment Accuracy | ≥ +0.15 (≥+30%) | +0.30 (+60%) |
| GPT-Judge | ≥ +0.5 (≥+20%) | +1.5 (+65%) |

### 8.5 BƯỚC 5: Chẩn Đoán Root Cause (Khi Kết Quả Không Đạt)

**Cây quyết định chẩn đoán — theo từng triệu chứng**:

```
Kết quả không đạt
│
├─── [A] Fine-tuned THẤP HƠN baseline?
│    │
│    ├── [A1] Structure Score giảm?
│    │   → NGUYÊN NHÂN: Catastrophic forgetting — model quên cách format
│    │   → XEM: Training data có đúng ChatML format không?
│    │   → XEM: Epochs quá nhiều? (overfit rồi quên instruction following)
│    │
│    ├── [A2] Sentiment Accuracy giảm?
│    │   → NGUYÊN NHÂN: Training data bị bias sentiment
│    │   → XEM: SELECT sentiment_direction, COUNT(*) FROM golden_responses GROUP BY sentiment_direction
│    │   → NẾU một hướng chiếm >60%: data imbalanced
│    │
│    ├── [A3] ROUGE giảm?
│    │   → NGUYÊN NHÂN: Model sinh output quá khác so với golden style
│    │   → XEM: So sánh length distribution: baseline vs fine-tuned vs golden
│    │   → NẾU fine-tuned ngắn hơn golden nhiều: max_tokens bị cắt
│    │
│    └── [A4] GPT-Judge giảm?
│        → NGUYÊN NHÂN: Output degrade tổng thể
│        → XEM: Đọc 5 examples xấu nhất (lowest judge score) thủ công
│        → Thường do: repetition, hallucination, hoặc gibberish
│
├─── [B] Fine-tuned CAO HƠN baseline nhưng CẢI THIỆN < 15%?
│    │
│    ├── [B1] Structure Score tăng ít (<+0.15)?
│    │   → NGUYÊN NHÂN: LoRA rank quá thấp — không đủ capacity để học format
│    │   → THỬ: Tăng r=32 (gấp đôi)
│    │
│    ├── [B2] ROUGE tăng ít (<+0.05)?
│    │   → NGUYÊN NHÂN: Golden responses quá diverse — mỗi response khác style
│    │   → THỬ: Giảm temperature golden generation → 0.1
│    │   → THỬ: Thêm explicit format template vào prompt golden
│    │
│    ├── [B3] Accuracy tăng ít (<+0.10)?
│    │   → NGUYÊN NHÂN: Task vốn ambiguous — nhiều examples có thể là Bullish lẫn Mixed
│    │   → THỬ: Merge Bullish/Bearish thành 2 class thay vì 4
│    │
│    └── [B4] Tất cả metrics tăng ít đồng đều?
│        → NGUYÊN NHÂN: Dataset quá nhỏ hoặc thiếu diversity
│        → THỬ: Tăng gấp đôi training data (thêm tickers + time periods)
│        → THỬ: Tăng epochs 3→5
│
└─── [C] Một số metrics đạt, một số không?
     │
     ├── [C1] Structure ĐẠT nhưng Accuracy KHÔNG ĐẠT?
     │   → Model học được FORMAT nhưng không học được REASONING
     │   → THỬ: Tăng LoRA rank (thêm capacity cho reasoning)
     │   → THỬ: Thêm "chain of thought" vào golden responses
     │
     ├── [C2] Accuracy ĐẠT nhưng Structure KHÔNG ĐẠT?
     │   → Model học được sentiment nhưng không theo đúng format
     │   → THỬ: Filter golden responses thiếu sections ra khỏi dataset
     │   → THỬ: Thêm format enforcement vào system prompt
     │
     └── [C3] ROUGE ĐẠT nhưng GPT-Judge KHÔNG ĐẠT?
         → Model copy style nhưng thiếu depth/nuance
         → THỬ: Chọn golden responses chất lượng cao hơn (filter bằng GPT-Judge trước)
         → THỬ: Thêm negative examples (xấu) + positive examples (tốt) để model phân biệt
```

### 8.6 BƯỚC 6: Playbook Cải Thiện (Theo Thứ Tự Ưu Tiên)

Khi kết quả không đạt, thực hiện các bước sau **THEO THỨ TỰ** (từ ít effort → nhiều effort):

#### Level 1: Quick Fixes (30 phút - 1 giờ)

| # | Hành động | Khi nào áp dụng | Chi phí |
|---|---|---|---|
| 1.1 | **Kiểm tra data format** — chạy lại `validate_dataset()` | Mọi trường hợp fail | 0 |
| 1.2 | **Filter golden responses kém** — loại bỏ examples thiếu sections/table | Structure Score thấp | 0 |
| 1.3 | **Kiểm tra tokenizer** — đảm bảo `apply_chat_template()` đúng | Output bị lỗi ký tự | 0 |

```python
# Quick check: phân bố chất lượng golden responses
import sqlite3
conn = sqlite3.connect("finetune/data/db/finetune_data.db")
cursor = conn.execute("""
    SELECT structure_score, COUNT(*) as count 
    FROM golden_responses 
    GROUP BY ROUND(structure_score, 1) 
    ORDER BY structure_score
""")
for row in cursor:
    print(f"  Score {row[0]:.1f}: {row[1]} examples")
# Nếu nhiều examples có score < 0.8 → filter chúng ra
```

#### Level 2: Hyperparameter Tuning (2-4 giờ, chạy lại Colab)

| # | Hành động | Khi nào áp dụng | Thay đổi |
|---|---|---|---|
| 2.1 | **Tăng LoRA rank** r=16 → r=32 | Cải thiện không đáng kể | Tăng capacity 2x |
| 2.2 | **Giảm learning rate** 2e-4 → 1e-4 | Loss oscillating | Training ổn định hơn |
| 2.3 | **Tăng epochs** 3 → 5 | Val loss chưa plateau | Thêm thời gian converge |
| 2.4 | **Giảm epochs** 3 → 2 | Overfitting (val loss tăng) | Tránh overfit |
| 2.5 | **Tăng dropout** 0.05 → 0.1 | Overfitting | Regularization |

```python
# Chạy lại training với config mới trên Colab
CONFIG_V2 = {
    "lora_r": 32,           # Tăng từ 16
    "lora_alpha": 64,       # Giữ 2×r
    "learning_rate": 1e-4,  # Giảm từ 2e-4
    "num_train_epochs": 5,  # Tăng từ 3
}
```

#### Level 3: Data Improvement (4-8 giờ)

| # | Hành động | Khi nào áp dụng | Chi phí |
|---|---|---|---|
| 3.1 | **Tăng training data** thêm 20 tickers × 18 tuần | Dataset quá nhỏ | ~$15-20 API |
| 3.2 | **Cải thiện golden quality** — giảm temperature → 0.1 | Golden responses inconsistent | ~$20-40 API |
| 3.3 | **Cân bằng sentiment distribution** | Accuracy thấp do bias | Resample |
| 3.4 | **Thêm hard examples** — examples mà baseline sai | Model không học edge cases | Manual curation |

```sql
-- Kiểm tra sentiment balance
SELECT sentiment_direction, 
       COUNT(*) as count,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM golden_responses), 1) as pct
FROM golden_responses 
GROUP BY sentiment_direction;

-- Kỳ vọng: mỗi direction ~20-35%. Nếu một direction > 50% → imbalanced
-- Fix: undersample majority class hoặc oversample minority class
```

#### Level 4: Architecture Changes (1-2 ngày, chạy lại full pipeline)

| # | Hành động | Khi nào áp dụng | Impact |
|---|---|---|---|
| 4.1 | **Thử model base lớn hơn**: Qwen3:8B | 4B không đủ capacity | VRAM tăng, cần A100 |
| 4.2 | **Full fine-tuning** thay vì LoRA | LoRA không đủ | Cần GPU lớn hơn |
| 4.3 | **Thêm DPO/RLHF stage** sau SFT | Quality plateau sau SFT | Phức tạp hơn nhiều |

#### Level 5: Fundamental Rethink (nếu tất cả trên thất bại)

| # | Hành động | Chi tiết |
|---|---|---|
| 5.1 | **Kiểm tra lại assumption** | Sentiment task có thực sự phù hợp cho model 4B local? |
| 5.2 | **Đơn giản hóa task** | Chỉ yêu cầu output sentiment direction + confidence, bỏ report dài |
| 5.3 | **Hybrid approach** | Dùng model fine-tuned cho classification + model gốc cho report generation |

### 8.7 Bảng Tổng Hợp: Metric → Nguyên Nhân → Giải Pháp

| Triệu chứng | Top 3 Nguyên Nhân | Giải pháp (theo thứ tự thử) |
|---|---|---|
| **Structure Score < 0.80** | 1. Golden data thiếu sections 2. LoRA rank quá thấp 3. Epochs chưa đủ | 1. Filter golden → 2. r=32 → 3. epochs=5 |
| **Accuracy < 70%** | 1. Data imbalanced (sentiment bias) 2. Cross-lingual confusion 3. Task ambiguity | 1. Balance data → 2. Giữ EN output → 3. Merge classes |
| **ROUGE-1 < 0.35** | 1. Output length mismatch 2. Golden responses quá diverse 3. Model hallucinate | 1. Check length → 2. Lower temp golden → 3. More data |
| **GPT-Judge < 3.0** | 1. Output thiếu evidence 2. Repetition 3. Irrelevant content | 1. Check golden quality → 2. Tăng dropout → 3. Manual review |
| **FT worse than baseline** | 1. Data format lỗi 2. Catastrophic forgetting 3. Overfit | 1. Validate format → 2. Giảm epochs → 3. Giảm LR |

### 8.8 Kỳ Vọng Kết Quả (Dựa Trên Literature)

Dựa trên các paper về QLoRA fine-tuning cho domain-specific tasks:

| Metric | Baseline (4B gốc) | FT Kỳ vọng (thận trọng) | FT Kỳ vọng (lạc quan) | Ceiling (teacher 120B) |
|---|---|---|---|---|
| Structure Score | ~0.49 | 0.82-0.88 | 0.90-0.96 | ~0.98 |
| ROUGE-1 F1 | ~0.28 | 0.38-0.44 | 0.45-0.55 | ~0.65 |
| ROUGE-2 F1 | ~0.10 | 0.17-0.22 | 0.22-0.30 | ~0.40 |
| Sent. Accuracy | ~50% | 68-75% | 78-88% | ~92% |
| GPT-Judge Avg | ~2.3/5 | 3.2-3.6 | 3.6-4.2 | ~4.5 |

> **Ceiling** là giới hạn trên lý thuyết — fine-tuned model KHÔNG THỂ vượt teacher model (gpt-oss-120b). Mục tiêu thực tế là đạt ~70-80% hiệu năng teacher.

### 8.9 Checklist Đánh Giá Hoàn Chỉnh

```
□ TRƯỚC TRAINING
  □ Chạy baseline evaluation trên test.jsonl
  □ Lưu kết quả vào eval_results/baseline_qwen3_4b.json
  □ Lưu vào SQLite eval_results table
  □ Ghi nhận baseline latency

□ SAU TRAINING  
  □ Deploy model fine-tuned lên Ollama
  □ Chạy fine-tuned evaluation trên CÙNG test.jsonl
  □ Lưu kết quả vào eval_results/finetuned_results.json
  □ Lưu vào SQLite eval_results table

□ SO SÁNH
  □ In bảng so sánh Baseline vs Fine-Tuned
  □ Kiểm tra 4 Go/No-Go criteria
  □ Tính % cải thiện cho mỗi metric
  □ Phân loại kết quả: ĐẠT / CẢI THIỆN KHÔNG ĐÁNG KỂ / THẤT BẠI

□ NẾU KHÔNG ĐẠT
  □ Chạy diagnostic queries trên SQLite
  □ Xác định root cause theo cây quyết định (8.5)
  □ Thực hiện Level 1 fixes trước
  □ Nếu vẫn không đạt → Level 2 (retrain)
  □ Ghi log mọi thay đổi và kết quả
```
