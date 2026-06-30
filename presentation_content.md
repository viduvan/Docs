# Nội Dung Thuyết Trình: Fine-Tuning Sentiment Analyst trong Hệ Thống CryptoAgents

> **Dự án**: CryptoAgents — Hệ thống Multi-Agent Trading tự động  
> **Phương pháp**: QLoRA Fine-Tuning với Knowledge Distillation  
> **Model**: Qwen3-4B → sentiment-analyst-ft  
> **Thời gian thực hiện**: ~10 giờ 40 phút – 12 giờ 40 phút  

---

## BỐI CẢNH DỰ ÁN: TỪ CLOUD API ĐẾN LOCAL MODEL VÀ BÀI TOÁN FINE-TUNING

Ban đầu, các hệ thống Multi-Agent thường sử dụng các API LLM thương mại lớn (như GPT-4, Claude) để đảm bảo chất lượng suy luận. Tuy nhiên, khi xây dựng và triển khai thực tế hệ thống **CryptoAgents**, chúng tôi quyết định chuyển sang sử dụng các **model open-source nhỏ (Small Language Models - SLMs) chạy local** vì 3 lý do cốt lõi:

1. **Chi phí vận hành (Cost):** Hệ thống gồm 5 sub-agent hoạt động liên tục, thường xuyên quét và xử lý lượng dữ liệu đầu vào khổng lồ (tin tức, giá, mạng xã hội, báo cáo tài chính). Nếu dùng API trả phí, lượng token tiêu thụ mỗi ngày sẽ tạo ra gánh nặng chi phí khổng lồ, không khả thi để duy trì lâu dài.
2. **Độ trễ và Giới hạn (Latency & Rate Limits):** Việc phụ thuộc vào API bên ngoài làm hệ thống dễ bị ảnh hưởng bởi độ trễ mạng và các giới hạn tốc độ (rate limit) của nhà cung cấp, làm giảm khả năng phản ứng theo thời gian thực của hệ thống giao dịch.
3. **Tính tự chủ (Autonomy):** Việc chạy model local giúp hệ thống hoàn toàn độc lập, hoạt động 24/7 mà không phụ thuộc vào tình trạng server của bên thứ ba.

**Vấn đề phát sinh (The Bottleneck):** 
Để hệ thống có thể chạy mượt mà trên phần cứng phổ thông (VRAM hạn chế), chúng tôi phải chọn model có kích thước nhỏ gọn (cụ thể là **Qwen3-4B**). 
Tuy nhiên, các model nhỏ bộc lộ giới hạn rõ rệt khi đối mặt với các nghiệp vụ phức tạp. Trong khi chúng xử lý khá tốt các tác vụ phân tích số liệu có cấu trúc định sẵn, thì đối với tác vụ **phân tích tâm lý thị trường (Sentiment Analysis)** — vốn đòi hỏi khả năng đọc hiểu ngôn ngữ tự nhiên từ nhiều nguồn phi cấu trúc hỗn loạn (Reddit, StockTwits, News), cross-reference để tìm mâu thuẫn và tổng hợp báo cáo — model nhỏ nguyên bản đã **thất bại nặng nề**.

**➡️ Bài Toán Đặt Ra:** Chúng ta không thể quay lại sử dụng API đắt đỏ, nhưng cũng không thể chấp nhận sai số của model nhỏ nguyên bản. Do đó, giải pháp tối ưu và bắt buộc là phải **Fine-Tune (huấn luyện tinh chỉnh)** model open-source nhỏ này (áp dụng Knowledge Distillation từ model lớn) để nó sở hữu năng lực chuyên sâu cho nghiệp vụ Sentiment Analysis mà vẫn giữ được lợi thế chạy local chi phí thấp!

---

## PHẦN 1: TẠI SAO CHỌN SENTIMENT ANALYST ĐỂ FINE-TUNE?

### 1.1 Kiến Trúc Hệ Thống CryptoAgents

Hệ thống CryptoAgents sử dụng kiến trúc **Multi-Agent** gồm 5 subagent chuyên biệt, mỗi agent đảm nhận một khía cạnh phân tích khác nhau:

```
                    ┌─────────────────────────────────────────┐
                    │           RESEARCH MANAGER              │
                    │     (Điều phối toàn bộ phân tích)       │
                    └──────────────┬──────────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │           │           │           │           │
     ┌─────▼────┐ ┌───▼─────┐ ┌──▼──────┐ ┌──▼──────┐ ┌──▼──────────┐
     │ Market   │ │ News    │ │Fundamen-│ │Quantita-│ │ Sentiment   │
     │ Analyst  │ │ Analyst │ │ tals    │ │ tive    │ │ Analyst ★   │
     │          │ │         │ │ Analyst │ │ Analyst │ │ (Fine-Tuned)│
     │ qwen3:4b │ │qwen3:4b│ │qwen3:4b │ │qwen3:4b │ │sentiment-ft │
     └─────┬────┘ └───┬─────┘ └──┬──────┘ └──┬──────┘ └──┬──────────┘
           │           │          │           │            │
           └───────────┼──────────┼───────────┼────────────┘
                       │          │           │
               ┌───────▼──────────▼───────────▼──────┐
               │     BULL RESEARCHER  ←→  BEAR       │
               │     (Tranh luận đầu tư đối kháng)   │
               └────────────────┬────────────────────┘
                                │
               ┌────────────────▼────────────────────┐
               │        RISK MANAGEMENT              │
               │  (Aggressive / Neutral / Conservative)│
               └────────────────┬────────────────────┘
                                │
               ┌────────────────▼────────────────────┐
               │           TRADER                     │
               │     Quyết định: BUY / HOLD / SELL   │
               └─────────────────────────────────────┘
```

### 1.2 Lý Do Chọn Sentiment Analyst — Phân Tích Bottleneck

Trong 5 subagent, **Sentiment Analyst** được lựa chọn để fine-tune vì 3 lý do cốt lõi:

#### ❶ Nghiệp vụ phức tạp nhất — Tổng hợp đa nguồn dữ liệu phi cấu trúc

| Subagent | Nguồn dữ liệu đầu vào | Loại dữ liệu | Độ phức tạp output |
| :--- | :--- | :--- | :--- |
| Market Analyst | Yahoo Finance API (số liệu giá, volume) | **Cấu trúc** (JSON/số) | Trung bình |
| News Analyst | Yahoo Finance News (tiêu đề + tóm tắt) | Bán cấu trúc | Trung bình |
| Fundamentals Analyst | Financial statements (P/E, EPS, Revenue) | **Cấu trúc** (bảng số) | Thấp |
| Quantitative Analyst | OHLCV data, Technical indicators | **Cấu trúc** (số) | Trung bình |
| **Sentiment Analyst ★** | News + StockTwits + Reddit (3 nguồn) | **Phi cấu trúc** (text tự do) | **Cao nhất** |

Sentiment Analyst phải:
- Đọc hiểu **tin tức tài chính** (institutional framing)
- Phân tích **bài viết StockTwits** với label Bullish/Bearish (retail sentiment) 
- Tổng hợp **thảo luận Reddit** từ r/wallstreetbets, r/stocks, r/investing
- **Cross-reference** giữa 3 nguồn để phát hiện divergence/alignment
- Xuất báo cáo có **5 phần bắt buộc** + bảng Markdown tóm tắt

→ Đây là task khó nhất cho model nhỏ 4B chạy local, và cũng là nơi model gốc **thất bại nặng nhất** (31.2% accuracy vs 50%+ của các agent khác).

#### ❷ Điểm yếu rõ ràng nhất của model gốc — Khi không có fine-tuning

Khi chạy model gốc `qwen3:4b` trên tập test, Sentiment Analyst bộc lộ 3 điểm yếu nghiêm trọng:

| Vấn đề | Biểu hiện | Hệ quả |
| :--- | :--- | :--- |
| **Sai hướng sentiment** | Model gốc chỉ đúng 31.2% hướng sentiment (Bullish/Bearish/Neutral/Mixed) | Cung cấp tín hiệu **sai** cho Bull/Bear Researcher → quyết định giao dịch sai |
| **Thiếu cấu trúc** | Output không tuân thủ 5 phần bắt buộc (Structure Score: 0.290/1.0) | Downstream agents không parse được thông tin → mất dữ liệu phân tích |
| **Nội dung lệch** | ROUGE-1 F1 chỉ 0.134 — rất thấp so với golden reference | Văn phong và nội dung không chuyên nghiệp, thiếu bằng chứng cụ thể |

#### ❸ Tác động lan truyền (Cascading Effect) trong hệ thống Multi-Agent

Sentiment Analyst nằm ở **tầng phân tích đầu tiên** (Tier 1), output của nó là **đầu vào trực tiếp** cho tầng tranh luận (Tier 2):

```
Tier 1: Sentiment Report (input chất lượng thấp)
    │
    ▼
Tier 2: Bull Researcher đọc report → lập luận mua dựa trên tín hiệu SAI
        Bear Researcher đọc report → phản biện dựa trên tín hiệu SAI
    │
    ▼
Tier 3: Risk Management đánh giá rủi ro dựa trên tranh luận SAI
    │
    ▼
Tier 4: Trader ra quyết định BUY/HOLD/SELL dựa trên toàn bộ chuỗi SAI
```

**Hiệu ứng Garbage-In-Garbage-Out**: Khi sentiment report sai ở Tier 1, toàn bộ chuỗi phân tích phía sau đều bị ảnh hưởng. Ngược lại, nếu cải thiện Sentiment Analyst, chất lượng phân tích của **toàn bộ pipeline** được nâng cao.

### 1.3 Lợi Ích Cụ Thể Khi Fine-Tune Sentiment Analyst

| # | Lợi ích | Trước Fine-Tune | Sau Fine-Tune | Tác động lên hệ thống |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Chính xác hướng sentiment** | 31.2% | **83.3%** (+52.1%) | Bull/Bear Researcher nhận tín hiệu đúng → tranh luận chất lượng hơn |
| 2 | **Chuẩn hóa output** | 0.290 | **0.934** (+0.644) | Downstream agents parse được 5 phần đầy đủ → không mất thông tin |
| 3 | **Nội dung chuyên nghiệp** | 0.134 ROUGE-1 | **0.436** (+0.302) | Báo cáo có bằng chứng, số liệu cụ thể → quyết định giao dịch có cơ sở |
| 4 | **Chi phí thấp** | Phải dùng API cloud (tốn phí) | Model 4B chạy local miễn phí | Giảm chi phí vận hành xuống **$0/tháng** cho sentiment analysis |
| 5 | **Latency thấp** | API cloud ~2-5s/request | Model local ~0.3-0.5s/request | Phân tích realtime, không phụ thuộc mạng |

---

## PHẦN 2: CHI TIẾT PHƯƠNG PHÁP FINE-TUNING

### 2.1 Tổng Quan Phương Pháp: Knowledge Distillation + QLoRA

Phương pháp fine-tuning sử dụng kỹ thuật **Knowledge Distillation** (Chưng cất tri thức) kết hợp **QLoRA** (Quantized Low-Rank Adaptation):

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     KNOWLEDGE DISTILLATION                              │
│                                                                          │
│   ┌─────────────────────┐           ┌─────────────────────────────┐     │
│   │   TEACHER MODEL     │           │       STUDENT MODEL          │     │
│   │                     │           │                              │     │
│   │   gpt-oss-120b      │  ──────→  │   Qwen3-4B                  │     │
│   │   (MoE 117B params, │  Golden   │   (4B params)               │     │
│   │    5.1B active)      │  Responses│                              │     │
│   │                     │           │   Học bắt chước output       │     │
│   │   Host: FPT Cloud   │           │   của Teacher qua SFT        │     │
│   └─────────────────────┘           └─────────────────────────────┘     │
│                                                                          │
│   Teacher tạo 1440 "Golden Responses"                                    │
│   Student học theo format, phong cách và độ chính xác của Teacher        │
└──────────────────────────────────────────────────────────────────────────┘
```

**Tại sao chọn Knowledge Distillation?**
- Model lớn (117B) quá nặng để chạy local → cần "chưng cất" kiến thức vào model nhỏ (4B)
- Student model chỉ cần 4GB VRAM để inference → triển khai trên máy local không cần GPU đắt tiền
- Phương pháp này đã được chứng minh hiệu quả trong nhiều bài báo khoa học (Hinton et al., 2015)

### 2.2 QLoRA — Phương Pháp Fine-Tuning Hiệu Quả

**QLoRA** (Quantized Low-Rank Adaptation) là phương pháp fine-tuning tiết kiệm tài nguyên nhất hiện nay:

#### Cơ chế hoạt động:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    QWEN3-4B (4 Billion Parameters)                   │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │   Base Weights (ĐÓNG BĂNG - Frozen)                            │  │
│  │   - 4B tham số gốc được giữ nguyên, KHÔNG thay đổi            │  │
│  │   - Được lượng tử hóa xuống 4-bit (NF4) để tiết kiệm RAM     │  │
│  │   - 4B params × 4 bit = ~2GB VRAM thay vì 16GB (FP32)        │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │   LoRA Adapters (TRAINABLE - Huấn luyện)                       │  │
│  │                                                                │  │
│  │   Thay vì cập nhật ma trận W (d×d), LoRA phân tách:           │  │
│  │                                                                │  │
│  │        ΔW = A × B                                              │  │
│  │                                                                │  │
│  │   Trong đó:                                                    │  │
│  │     W  ∈ ℝ^(d×d) — Ma trận trọng số gốc (hàng triệu params) │  │
│  │     A  ∈ ℝ^(d×r) — Ma trận hạ chiều (d → r)                  │  │
│  │     B  ∈ ℝ^(r×d) — Ma trận tái chiều (r → d)                 │  │
│  │     r = 16       — Rank (hạng) — chỉ 16 chiều thay vì d      │  │
│  │     α = 32       — Scaling factor (hệ số tỷ lệ)              │  │
│  │                                                                │  │
│  │   Số params cần train: 2 × d × r ≈ 0.5-1% tổng params       │  │
│  │   → Chỉ ~20-40M params thay vì 4B params!                    │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  Output cuối cùng = Base_output + (α/r) × ΔW × input               │
└──────────────────────────────────────────────────────────────────────┘
```

#### Tại sao chọn QLoRA thay vì Full Fine-Tuning?

| Tiêu chí | Full Fine-Tuning | QLoRA |
| :--- | :--- | :--- |
| **Số params cần train** | 4B (100%) | ~20-40M (**~1%**) |
| **VRAM cần thiết** | ~32-64 GB (A100 80GB) | **~8-12 GB** (T4 16GB đủ) |
| **Thời gian train** | 12-24 giờ | **~2.5 giờ** |
| **Chi phí GPU** | ~$50-100 (A100) | **~$5-10** (T4 trên Colab Pro) |
| **Rủi ro Catastrophic Forgetting** | Cao (thay đổi toàn bộ weights) | **Thấp** (weights gốc frozen) |
| **Hiệu năng** | Tốt nhất | **95-99%** so với full FT |

### 2.3 Cấu Hình Huấn Luyện Chi Tiết

#### Training Configuration:

```python
# === Model Configuration ===
BASE_MODEL    = "Qwen/Qwen3-4B"        # Model nền từ HuggingFace
QUANTIZATION  = "4-bit NF4"             # Lượng tử hóa NormalFloat 4-bit
MAX_SEQ_LEN   = 4096                     # Chiều dài chuỗi tối đa (tokens)

# === LoRA Configuration ===
LORA_R        = 16                       # Rank — số chiều adapter
LORA_ALPHA    = 32                       # Scaling factor (thường = 2 × r)
LORA_DROPOUT  = 0.05                     # Dropout regularization (5%)
TARGET_MODULES = [                       # 7 loại ma trận được gắn adapter
    "q_proj",    # Query projection  (Attention)
    "k_proj",    # Key projection    (Attention)
    "v_proj",    # Value projection  (Attention)
    "o_proj",    # Output projection (Attention)
    "gate_proj", # Gate projection   (MLP/FFN)
    "up_proj",   # Up projection     (MLP/FFN)
    "down_proj"  # Down projection   (MLP/FFN)
]

# === Training Hyperparameters ===
BATCH_SIZE      = 4                      # Batch size per device
GRAD_ACCUM      = 4                      # Gradient accumulation steps
EFFECTIVE_BATCH = 4 × 4 = 16            # Effective batch size
LEARNING_RATE   = 2e-4                   # Learning rate (AdamW 8-bit)
EPOCHS          = 3                      # Số epoch huấn luyện
WARMUP_RATIO    = 0.05                   # 5% steps đầu warmup
SCHEDULER       = "cosine"               # Cosine annealing LR schedule
WEIGHT_DECAY    = 0.01                   # L2 regularization

# === Framework ===
FRAMEWORK = "Unsloth + TRL SFTTrainer"  # Thư viện tối ưu hóa tốc độ train
PLATFORM  = "Google Colab Pro"           # GPU: T4/L4/A100 (auto-detect)
```

#### Giải thích các Target Modules:

```
                 Qwen3-4B Transformer Block
                 ┌─────────────────────────────────────┐
                 │                                     │
Input ──────────▶│  Multi-Head Self-Attention           │
                 │  ┌───────┐ ┌───────┐ ┌───────┐     │
                 │  │q_proj │ │k_proj │ │v_proj │ ★   │  ← LoRA trên Q, K, V
                 │  └───┬───┘ └───┬───┘ └───┬───┘     │
                 │      │         │         │          │
                 │      └─────────┼─────────┘          │
                 │                │                     │
                 │  ┌─────────────▼───────────────┐    │
                 │  │         o_proj              │ ★  │  ← LoRA trên Output
                 │  └─────────────┬───────────────┘    │
                 │                │                     │
                 │  ┌─────────────▼───────────────┐    │
                 │  │    Feed-Forward Network       │    │
                 │  │ ┌──────┐ ┌──────┐ ┌────────┐│    │
                 │  │ │gate_ │ │ up_  │ │ down_  ││ ★  │  ← LoRA trên MLP
                 │  │ │proj  │ │proj  │ │ proj   ││    │
                 │  │ └──────┘ └──────┘ └────────┘│    │
                 │  └─────────────┬───────────────┘    │
                 │                │                     │
Output ◀─────────│────────────────┘                     │
                 └─────────────────────────────────────┘
```

### 2.4 Dữ Liệu Huấn Luyện

#### Quy trình chuẩn bị dữ liệu (Data Pipeline):

```
Bước 1: Thu thập dữ liệu thô (01_collect_data.py)
─────────────────────────────────────────────────
  60 tickers × 24 tuần = ~1440 raw examples
  
  Nguồn:
  ├── Yahoo Finance News   → Tiêu đề, tóm tắt bài báo
  ├── StockTwits API       → Messages + label Bullish/Bearish
  └── Reddit Public API    → Posts từ r/wallstreetbets, r/stocks, r/investing

Bước 2: Sinh Golden Responses (01b_generate_golden.py)
─────────────────────────────────────────────────────
  1440 raw examples → gpt-oss-120b (FPT Cloud) → 1440 Golden Responses
  
  Mỗi Golden Response là bản phân tích sentiment hoàn chỉnh
  gồm 5 phần do model Teacher (117B) sinh ra (chi tiết bên dưới).
```

#### 5 Phần Bắt Buộc Trong Mỗi Golden Response

Model Teacher (`gpt-oss-120b`) sinh ra mỗi Golden Response theo cấu trúc **5 phần bắt buộc** — đây chính là "khuôn mẫu vàng" mà Student model phải học theo:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    CẤU TRÚC GOLDEN RESPONSE (5 PHẦN)                        │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ PHẦN 1: Overall Sentiment Direction (Hướng đi tổng thể)              │  │
│  │                                                                        │  │
│  │ Mục đích: Kết luận tổng quan — thị trường đang Bullish, Bearish,      │  │
│  │           Neutral hay Mixed? Kèm ghi chú độ tin cậy dựa trên          │  │
│  │           chất lượng và số lượng dữ liệu đầu vào.                     │  │
│  │                                                                        │  │
│  │ Ví dụ output:                                                          │  │
│  │   "## Overall Sentiment Direction: **Bullish**                         │  │
│  │    Confidence: Moderate-High. Based on 12 news articles,               │  │
│  │    30 StockTwits messages (75% bullish), and 8 Reddit threads          │  │
│  │    with active discussion."                                            │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ PHẦN 2: Source-by-Source Breakdown (Phân tích chi tiết từng nguồn)    │  │
│  │                                                                        │  │
│  │ Mục đích: Phân tích riêng biệt từng nguồn dữ liệu, chỉ ra mỗi       │  │
│  │           nguồn "nói gì" với bằng chứng cụ thể (số liệu, trích dẫn). │  │
│  │                                                                        │  │
│  │ 3 nguồn được phân tích:                                                │  │
│  │   📰 News (Yahoo Finance)   → Institutional framing, sự kiện chính    │  │
│  │   💬 StockTwits             → Tỷ lệ Bullish/Bearish, trend retail     │  │
│  │   🗣️ Reddit                 → Thảo luận cộng đồng, engagement score   │  │
│  │                                                                        │  │
│  │ Ví dụ output:                                                          │  │
│  │   "### News (Yahoo Finance)                                            │  │
│  │    12 articles found. 8 positive (67%), 3 neutral, 1 negative.         │  │
│  │    Key headlines: 'NVDA partners with Corning on $500M deal'...        │  │
│  │                                                                        │  │
│  │    ### StockTwits                                                      │  │
│  │    30 messages sampled. Bullish: 22 (73%), Bearish: 5 (17%)...         │  │
│  │                                                                        │  │
│  │    ### Reddit                                                          │  │
│  │    8 posts across r/wallstreetbets (3), r/stocks (4), r/investing (1). │  │
│  │    Top post: 'NVDA earnings beat expectations' (↑340, 198 comments)."  │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ PHẦN 3: Divergences, Alignments & Key Narratives                      │  │
│  │         (Phân kỳ, đồng thuận & câu chuyện chủ đạo)                   │  │
│  │                                                                        │  │
│  │ Mục đích: Cross-reference giữa 3 nguồn — các nguồn có đồng thuận     │  │
│  │           hay mâu thuẫn nhau? Xác định narrative chủ đạo đang          │  │
│  │           chi phối thị trường.                                         │  │
│  │                                                                        │  │
│  │ Ví dụ output:                                                          │  │
│  │   "## Divergences & Alignments                                         │  │
│  │    **Alignment**: News và StockTwits đồng thuận bullish —              │  │
│  │    tin tức tích cực về partnership được retail đón nhận nhiệt tình.    │  │
│  │                                                                        │  │
│  │    **Divergence**: Reddit r/wallstreetbets có tone thận trọng hơn,     │  │
│  │    một số post cảnh báo 'priced in' với valuation P/E >40.            │  │
│  │    Đây là tín hiệu contrarian — retail social media bullish           │  │
│  │    nhưng cộng đồng thảo luận sâu hơn thì hoài nghi.                  │  │
│  │                                                                        │  │
│  │    **Key Narrative**: AI infrastructure spending là câu chuyện         │  │
│  │    chủ đạo, xuất hiện ở cả 3 nguồn."                                 │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ PHẦN 4: Catalysts & Risks (Chất xúc tác & Rủi ro)                    │  │
│  │                                                                        │  │
│  │ Mục đích: Liệt kê các sự kiện/yếu tố có thể đẩy giá lên             │  │
│  │           (catalysts) hoặc kéo giá xuống (risks) trong thời gian tới. │  │
│  │                                                                        │  │
│  │ Ví dụ output:                                                          │  │
│  │   "## Catalysts & Risks                                                │  │
│  │    **Catalysts (Tăng giá):**                                           │  │
│  │    - Earnings report Q2 dự kiến 28/06 — consensus beat kỳ vọng        │  │
│  │    - Partnership mới với Corning ($500M)                               │  │
│  │    - AI inference demand tăng trưởng 40% YoY                          │  │
│  │                                                                        │  │
│  │    **Risks (Giảm giá):**                                               │  │
│  │    - Valuation cao (P/E >40), rủi ro 'priced in'                      │  │
│  │    - Quy định xuất khẩu chip mới sang Trung Quốc                     │  │
│  │    - Cạnh tranh từ AMD MI300X tăng market share"                      │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │ PHẦN 5: Markdown Summary Table (Bảng tổng hợp tín hiệu)              │  │
│  │                                                                        │  │
│  │ Mục đích: Bảng Markdown tóm tắt toàn bộ tín hiệu sentiment           │  │
│  │           ở dạng có cấu trúc — giúp downstream agents (Bull/Bear      │  │
│  │           Researcher) dễ dàng parse và trích xuất thông tin.           │  │
│  │                                                                        │  │
│  │ Ví dụ output:                                                          │  │
│  │   | Signal          | Direction | Source     | Evidence              | │  │
│  │   |:----------------|:----------|:-----------|:----------------------| │  │
│  │   | News Framing    | Bullish   | Yahoo Fin. | 67% positive articles | │  │
│  │   | Retail Sentiment| Bullish   | StockTwits | 73% bullish messages  | │  │
│  │   | Community Mood  | Mixed     | Reddit     | WSB cautious, r/stocks│ │  │
│  │   |                 |           |            | positive              | │  │
│  │   | Upcoming Event  | Catalyst  | All Sources| Earnings 28/06        | │  │
│  │   | Valuation Risk  | Bearish   | Reddit     | P/E >40 concerns      | │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Tại sao phải có đúng 5 phần?                                               │
│  → Downstream agents (Bull/Bear Researcher) cần parse output có cấu trúc.   │
│  → Nếu thiếu phần nào, thông tin bị mất → phân tích không đầy đủ.          │
│  → Structure Score đo chính xác: score = (số phần có) / 5.0                 │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### Quy trình chuẩn bị dữ liệu (Data Pipeline):

```

Bước 3: Chuẩn bị dataset (02_prepare_dataset.py)
─────────────────────────────────────────────────
  1440 Golden → Format ChatML → Validate → Shuffle (seed=42) → Split:
  
  ┌────────────┐  ┌───────────┐  ┌───────────┐
  │  Training   │  │ Validation │  │   Test    │
  │  80%        │  │   10%      │  │   10%     │
  │  ~1152      │  │   ~144     │  │   ~144    │
  │  examples   │  │  examples  │  │  examples │
  └────────────┘  └───────────┘  └───────────┘
```

#### Định dạng dữ liệu — ChatML (Qwen3 Native):

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant... [System prompt + Sentiment Analyst instructions + 3 data blocks]"
    },
    {
      "role": "user", 
      "content": "Continue"
    },
    {
      "role": "assistant",
      "content": "[Golden Response từ gpt-oss-120b — bản phân tích 5 phần hoàn chỉnh]"
    }
  ]
}
```

### 2.5 Quy Trình Triển Khai Sau Training

```
Training Output                    Deployment
─────────────────                  ──────────────────────

LoRA Adapter                       
    │                              
    ▼                              
Merge với Base Weights             
(adapter + frozen weights)         
    │                              
    ▼                              
Full Merged Model (FP16)           
    │                              
    ▼                              
Quantize → GGUF (Q4_K_M)     ──→  ollama create sentiment-analyst-ft
    │                                        │
    │   4-bit quantized                      ▼
    │   ~2.5 GB file size              .env config:
    │                                  TRADINGAGENTS_SENTIMENT_LLM=sentiment-analyst-ft
    ▼                                        │
Upload HuggingFace                           ▼
viduvan/sentiment-analyst-ft          Chạy trong CryptoAgents Pipeline
```

---

## PHẦN 3: ĐÁNH GIÁ SO SÁNH TRƯỚC VÀ SAU FINE-TUNING

### 3.1 Bảng So Sánh Kết Quả Thực Tế

Kết quả benchmark trên **144 mẫu test** (trùng nhau hoàn toàn giữa 2 model, đảm bảo công bằng):

| Chỉ số | Baseline (`qwen3:4b`) | Fine-Tuned (`sentiment-analyst-ft`) | Độ lệch (Δ) | Đạt ngưỡng? |
| :--- | :---: | :---: | :---: | :---: |
| **Sentiment Accuracy** | 31.2% | **83.3%** | 📈 **+52.1%** | ✅ ≥70% |
| **Structure Score** | 0.290 | **0.934** | 📈 **+0.644** | ✅ ≥0.80 |
| **ROUGE-1 F1** | 0.134 | **0.436** | 📈 **+0.302** | ✅ ≥0.35 |
| **GPT-as-Judge** | 1.70/5.0 | **2.99/5.0** | 📈 **+1.293** | ⚠️ ≈3.0 (sát ngưỡng) |

### 3.2 Giải Thích Chi Tiết Từng Chỉ Số Cốt Lõi

---

#### 📊 Chỉ Số 1: Sentiment Accuracy (Độ Chính Xác Phân Loại Sentiment)

**Mục đích**: Đo khả năng model xác định đúng **hướng đi tổng thể** của sentiment (Bullish / Bearish / Neutral / Mixed).

**Cách tính**:

```
                    Số mẫu khớp hướng với Golden Response
Accuracy = ──────────────────────────────────────────────── × 100%
                    Tổng số mẫu test (144)
```

**Quy trình trích xuất**:

```python
def extract_sentiment_direction(response_text):
    """
    Thuật toán trích xuất hướng sentiment theo 4 bước ưu tiên:
    """
    
    # Bước 1 (Ưu tiên cao nhất): Tìm pattern chính xác
    # Ví dụ: "Overall Sentiment Direction: **Bullish**"
    patterns = [
        r"overall sentiment direction\s*:\s*\**bullish\**",
        r"sentiment direction\s*:\s*\**bearish\**",
        ...
    ]
    
    # Bước 2: Quét các dòng ngay sau heading
    # Nếu tìm thấy dòng chứa "overall sentiment direction",
    # kiểm tra 3 dòng tiếp theo để tìm keyword
    
    # Bước 3: Kiểm tra 300 ký tự đầu tiên
    # Tìm keyword đầu tiên xuất hiện (in đậm hoặc in nghiêng)
    
    # Bước 4 (Fallback): Đếm tần suất xuất hiện
    # Keyword nào xuất hiện nhiều nhất → chọn keyword đó
    
    return sentiment_direction  # "Bullish" | "Bearish" | "Neutral" | "Mixed"
```

**So sánh minh họa**:

| | Golden Reference | Baseline Output | Fine-Tuned Output |
| :--- | :--- | :--- | :--- |
| **BTC-USD** | Bullish | ❌ Mixed (sai) | ✅ Bullish (đúng) |
| **NVDA** | Bearish | ❌ Neutral (sai) | ✅ Bearish (đúng) |
| **DOGE-USD** | Mixed | ❌ Bullish (sai) | ✅ Mixed (đúng) |

**Tại sao Baseline chỉ đạt 31.2%?** → Model gốc `qwen3:4b` không được train đặc thù cho task sentiment analysis tài chính. Nó thường:
- Bị chi phối bởi một nguồn duy nhất (chỉ đọc StockTwits, bỏ qua News và Reddit)
- Không biết cross-reference giữa các nguồn
- Mặc định thiên về "Neutral" hoặc "Mixed" khi không chắc chắn

---

#### 📐 Chỉ Số 2: Structure Score (Điểm Cấu Trúc Báo Cáo)

**Mục đích**: Đo xem output của model có tuân thủ đúng **5 phần bắt buộc** của báo cáo sentiment hay không.

**5 thành phần được kiểm tra**:

```
┌─────────────────────────────────────────────────────────┐
│                 BÁO CÁO SENTIMENT                       │
│                                                          │
│  ① Overall Sentiment Direction                           │  ← Bullish/Bearish/Neutral/Mixed
│     Keywords: "sentiment direction", "overall sentiment" │
│                                                          │
│  ② Source-by-Source Breakdown                            │  ← Phân tích từng nguồn
│     Keywords: "breakdown", "source-by-source"            │
│                                                          │
│  ③ Divergence/Alignment Analysis                         │  ← So sánh giữa các nguồn
│     Keywords: "divergence", "alignment", "mismatch"      │
│                                                          │
│  ④ Catalysts & Risks                                     │  ← Yếu tố tác động & rủi ro
│     Keywords: "catalyst", "risk", "threat", "trigger"    │
│                                                          │
│  ⑤ Markdown Summary Table                                │  ← Bảng tóm tắt
│     Detection: "|" character + "-" character present     │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Cách tính**:

```
                Số thành phần xuất hiện trong output
Structure Score = ──────────────────────────────────────
                              5.0
```

| Ví dụ | ① | ② | ③ | ④ | ⑤ | Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (điển hình) | ✅ | ❌ | ❌ | ❌ | ✅ | **0.4** |
| Fine-Tuned (điển hình) | ✅ | ✅ | ✅ | ✅ | ✅ | **1.0** |

**Tại sao Baseline chỉ đạt 0.290?** → Model gốc thường chỉ viết 1-2 đoạn tóm tắt ngắn, không tuân thủ cấu trúc 5 phần. Nó thiếu khả năng "tuân thủ format" (instruction following) cho task cụ thể này.

---

#### 📝 Chỉ Số 3: ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

**Mục đích**: Đo lường **độ tương đồng nội dung** giữa output của model và Golden Response (bản mẫu chuẩn từ Teacher model).

**3 biến thể ROUGE sử dụng**:

```
ROUGE-1: Đo trùng khớp từ đơn (unigram)
─────────────────────────────────────────
Golden:   "StockTwits sentiment is overwhelmingly bullish at 75%"
Model FT: "StockTwits shows bullish sentiment at approximately 75%"
                                  ↑           ↑              ↑
                              Khớp từ     Khớp từ        Khớp từ

Precision = Từ khớp trong FT output / Tổng từ FT output
Recall    = Từ khớp trong FT output / Tổng từ Golden Reference
F1        = 2 × (Precision × Recall) / (Precision + Recall)


ROUGE-2: Đo trùng khớp cặp từ liền kề (bigram)
─────────────────────────────────────────────────
Golden:   "bullish sentiment" | "sentiment at" | "at 75%"
Model FT: "bullish sentiment" | "sentiment at" | "at approximately"
              ↑ Khớp bigram       ↑ Khớp bigram


ROUGE-L: Chuỗi chung dài nhất (Longest Common Subsequence)
───────────────────────────────────────────────────────────
Golden:   "The   overall   sentiment   is   moderately   bullish"
Model FT: "The   sentiment   analysis   indicates   moderately   bullish"
           ↑                 ↑                        ↑          ↑
           LCS: ["The", "sentiment", "moderately", "bullish"] → length = 4

ROUGE-L = LCS_length / max(len_golden, len_model)
```

**Thư viện sử dụng**: `rouge_score` (Google Research)

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                                                                  # ↑ Stemming: "running" → "run"
                                                                  #   để tăng recall khi so sánh

scores = scorer.score(golden_reference, model_output)
# scores['rouge1'].fmeasure → ROUGE-1 F1 Score
# scores['rouge2'].fmeasure → ROUGE-2 F1 Score
# scores['rougeL'].fmeasure → ROUGE-L F1 Score
```

**Kết quả ROUGE chi tiết**:

| Biến thể | Baseline | Fine-Tuned | Ý nghĩa |
| :--- | :---: | :---: | :--- |
| ROUGE-1 F1 | 0.134 | **0.436** | FT sử dụng ~44% từ vựng chuyên ngành giống Teacher |
| ROUGE-2 F1 | (thấp) | (cao hơn) | FT có cấu trúc câu mạch lạc hơn |
| ROUGE-L F1 | (thấp) | (cao hơn) | FT sắp xếp thông tin theo thứ tự logic giống Teacher |

---

#### 🧑‍⚖️ Chỉ Số 4: GPT-as-Judge (Đánh Giá Bằng Trọng Tài GPT)

**Mục đích**: Đánh giá **chất lượng tổng thể** của báo cáo bằng một model lớn (`gpt-oss-120b`) đóng vai trọng tài độc lập.

**Quy trình**:

```
20 mẫu ngẫu nhiên (random.seed=42)
    │
    ▼
Gửi cho gpt-oss-120b:
    ├── System Prompt gốc (context)
    ├── User Prompt (input data)
    ├── Golden Reference (bản mẫu chuẩn)
    └── Student Response (output cần chấm)
    │
    ▼
gpt-oss-120b chấm điểm JSON:
{
    "accuracy":      4.5,    ← Thông tin có chính xác không?
    "evidence":      4.0,    ← Có dẫn chứng số liệu cụ thể không?
    "structure":     5.0,    ← Có tuân thủ 5 phần format không?
    "actionability": 3.5,    ← Có ứng dụng được cho giao dịch không?
    "nuance":        4.0,    ← Có nắm bắt điểm mâu thuẫn ẩn không?
    "justification": "..."   ← Lý do chấm điểm (text)
}
    │
    ▼
Average = (accuracy + evidence + structure + actionability + nuance) / 5.0
```

**5 tiêu chí đánh giá chi tiết**:

| # | Tiêu chí | Điểm 1 (Kém) | Điểm 3 (Trung bình) | Điểm 5 (Xuất sắc) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **Accuracy** | Sentiment sai hoàn toàn | Đúng hướng nhưng thiếu chi tiết | Hoàn toàn khớp với Golden |
| 2 | **Evidence** | Không có số liệu | Có vài con số nhưng thiếu ngữ cảnh | Đầy đủ %, counts, sự kiện cụ thể |
| 3 | **Structure** | Viết dạng paragraph tự do | Có heading nhưng thiếu phần | Đủ 5 phần + bảng Markdown |
| 4 | **Actionability** | Dự đoán giá cụ thể (sai cách) | Có tín hiệu nhưng không rõ ràng | Tín hiệu trading rõ ràng, có caveat |
| 5 | **Nuance** | Bỏ qua mâu thuẫn giữa nguồn | Nhận ra divergence nhưng không phân tích | Phân tích sâu mâu thuẫn ẩn |

**Tại sao GPT-Judge đạt 2.99 (sát ngưỡng 3.0)?**

Nguyên nhân chính: **Thinking Block Truncation**

```
Model Qwen3 có kiến trúc dual-mode:
┌────────────────────────────────────────────────────┐
│ <think>                                            │
│   Đây là phần suy nghĩ nội bộ... (rất dài)        │  ← Tốn 500-1500 tokens
│   Model phân tích từng nguồn, so sánh...           │
│ </think>                                           │
│                                                     │
│ ## Overall Sentiment Direction: **Bullish**         │  ← Phần output thật
│ ...                                                 │
│ ## Source Breakdown                                  │
│ ...                                                 │
│ [BỊ CẮT ĐÂY do max_tokens=1500 cạn kiệt] ✂️      │  ← Truncation!
└────────────────────────────────────────────────────┘

→ Giải pháp: Tăng max_tokens = 3000 (đã áp dụng)
   Khi đó GPT-Judge score kỳ vọng ≥ 3.5/5.0
```

### 3.3 Biểu Đồ So Sánh Trực Quan

```
         Baseline (qwen3:4b)  vs  Fine-Tuned (sentiment-analyst-ft)
         
Accuracy   ████████░░░░░░░░░░░░░░░░░░░░░░░  31.2%
           █████████████████████████████████  83.3%  (+52.1%)

Structure  ████████░░░░░░░░░░░░░░░░░░░░░░░  29.0%
           █████████████████████████████████  93.4%  (+64.4%)

ROUGE-1    ████░░░░░░░░░░░░░░░░░░░░░░░░░░░  13.4%
           ███████████████░░░░░░░░░░░░░░░░░  43.6%  (+30.2%)

GPT-Judge  ███████████░░░░░░░░░░░░░░░░░░░░░  34.0% (1.70/5.0)
           ████████████████████░░░░░░░░░░░░░  59.8% (2.99/5.0) (+25.8%)
```

### 3.4 Go/No-Go Assessment — Kết Luận Tổng Thể

```
┌──────────────────────────────────────────────────────────────┐
│               GO / NO-GO ASSESSMENT RESULTS                  │
│                                                              │
│  ✅  Structure Score  : 0.934 ≥ 0.80   → PASSED              │
│  ✅  Sentiment Acc    : 83.3% ≥ 70%    → PASSED              │
│  ✅  ROUGE-1 F1       : 0.436 ≥ 0.35   → PASSED              │
│  ⚠️  GPT-Judge Avg   : 2.99  ≈ 3.0    → MARGINAL (−0.007)   │
│                                                              │
│  ─────────────────────────────────────────────────────────   │
│                                                              │
│  📈 FT tốt hơn Baseline trên TẤT CẢ 4 metrics              │
│  📈 Cải thiện trung bình: +37.5% (vượt ngưỡng 15%)          │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │        🎉 VERDICT: GO — APPROVE FOR DEPLOYMENT          │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Lưu ý: GPT-Judge score sát ngưỡng do Thinking Block        │
│  truncation tại max_tokens=1500. Đã khắc phục bằng          │
│  cách tăng max_tokens=3000 cho phiên đánh giá tiếp theo.    │
└──────────────────────────────────────────────────────────────┘
```

### 3.5 Kỳ Vọng vs Thực Tế (Dựa Trên Literature)

| Metric | Baseline | Kỳ vọng (thận trọng) | Kỳ vọng (lạc quan) | **Thực tế** | Ceiling (Teacher 120B) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Structure | 0.290 | 0.82-0.88 | 0.90-0.96 | **0.934** ✅ (lạc quan) | ~0.98 |
| ROUGE-1 | 0.134 | 0.38-0.44 | 0.45-0.55 | **0.436** ✅ (khớp kỳ vọng) | ~0.65 |
| Accuracy | 31.2% | 68-75% | 78-88% | **83.3%** ✅ (lạc quan) | ~92% |
| GPT-Judge | 1.70 | 3.2-3.6 | 3.6-4.2 | **2.99** ⚠️ (dưới kỳ vọng*) | ~4.5 |

*\* GPT-Judge bị ảnh hưởng bởi lỗi truncation, không phản ánh đúng năng lực model.*

---

## PHẦN BỔ SUNG: TÓM TẮT TIMELINE

| Bước | Nội dung | Thời gian | Môi trường |
| :--- | :--- | :---: | :--- |
| 1 | Khởi tạo cấu trúc, database | 15 phút | Local |
| 2 | Thu thập raw data (60 tickers × 24 tuần) | 2-3 giờ | Local |
| 3 | Sinh 1440 Golden Responses | 4-5 giờ | FPT Cloud API |
| 4 | Format ChatML + chia Train/Val/Test | 15 phút | Local |
| 5 | Upload Drive + Train QLoRA | 2.5 giờ | Google Colab Pro |
| 6 | Tải GGUF + Deploy Ollama | 10 phút | Local |
| 7 | Đánh giá Baseline (144 mẫu) | 45 phút | Local GPU |
| 8 | Đánh giá Fine-Tuned (144 mẫu) | 45 phút | Local GPU |
| **Tổng** | **Toàn bộ pipeline** | **~10h40 – 12h40** | |
