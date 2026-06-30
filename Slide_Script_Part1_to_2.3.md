# KỊCH BẢN THUYẾT TRÌNH (TỪ ĐẦU ĐẾN HẾT PHẦN 2.3)

Tài liệu này cung cấp cấu trúc các Slide thuyết trình và lời thoại (kịch bản nói) tương ứng cho phần thuyết trình của bạn. Phần 2 đã được bổ sung chi tiết để giảng giải kỹ thuật chuyên sâu.

---

## Slide 1: Tiêu Đề
**Nội dung trên Slide:**
- **Tên Đề Tài:** Tối ưu hóa Sentiment Analyst Agent trong Hệ thống Giao dịch Crypto Multi-Agent.
- **Giải pháp kỹ thuật:** Knowledge Distillation & QLoRA Fine-tuning trên mô hình Qwen3-4B.
- **Người trình bày:** [Tên của bạn]

**Kịch bản nói (Script):**
> "Chào thầy và các bạn. Hôm nay em xin trình bày về việc tối ưu hóa một thành phần cốt lõi trong hệ thống giao dịch tự động CryptoAgents, đó là Sentiment Analyst Agent. Trọng tâm của bài thuyết trình sẽ tập trung vào việc áp dụng kỹ thuật Knowledge Distillation và QLoRA để huấn luyện mô hình ngôn ngữ nhỏ Qwen3-4B, giúp nó đạt hiệu năng tương đương các mô hình lớn nhưng với chi phí vận hành bằng không."

---

## Slide 2: Bối Cảnh Dự Án - Sự Chuyển Dịch Từ API Sang Local SLM
**Nội dung trên Slide:**
- **Vấn đề của API (GPT-4/Claude):** Chi phí cao ($2K-$5K/tháng), độ trễ lớn (2-5s/request), rủi ro bảo mật dữ liệu & giới hạn rate-limit.
- **Xu hướng:** Dùng Small Language Models (SLM) chạy local hoàn toàn miễn phí.
- **Thách thức cốt lõi:** SLM chạy local (dưới 8 Tỷ tham số) bị thiếu khả năng suy luận logic sâu sắc đối với dữ liệu tài chính phức tạp.

**Kịch bản nói (Script):**
> "Trước khi đi vào kỹ thuật, em muốn nói về bối cảnh của dự án. Ban đầu, các hệ thống giao dịch Agentic thường phụ thuộc vào API của OpenAI hay Anthropic. Tuy nhiên, việc phải gọi API liên tục cho hàng chục mã cổ phiếu mỗi ngày đẩy chi phí lên hàng ngàn đô la mỗi tháng, kèm theo độ trễ mạng lớn. Vì vậy, nhóm em quyết định chuyển dịch toàn bộ hệ thống về chạy trên các mô hình ngôn ngữ nhỏ (SLM) ngay tại Local. Dù giải quyết được bài toán chi phí, nhưng SLM nguyên bản lại gặp vấn đề lớn về độ chính xác khi phân tích các bài báo tài chính chuyên sâu. Đó là lý do chúng ta bắt buộc phải Fine-tuning."

---

## Slide 3: Kiến Trúc Hệ Thống CryptoAgents & Bài Toán Cổ Chai
**Nội dung trên Slide:**
- **Sơ đồ kiến trúc:** Research Manager → 4 Analysts (Market, News, Funda, Sentiment) → Bull/Bear → Risk → Trader.
- **Sentiment Analyst (The Bottleneck):** Nằm ở tầng đầu vào (Tier 1).
- **Hệ quả:** Garbage-In-Garbage-Out (Phân tích sai → Tranh luận sai → Giao dịch sai).

**Kịch bản nói (Script):**
> "Nhìn vào sơ đồ kiến trúc, hệ thống của chúng em là một dây chuyền Multi-Agent. Research Manager điều phối 4 chuyên gia phân tích tầng Tier 1. Dữ liệu từ các chuyên gia này sẽ được đưa xuống tầng Tranh luận, qua Quản trị rủi ro và ra quyết định. 
> Tại sao lại chọn Fine-tune Sentiment Analyst? Vì Agent này là một điểm thắt cổ chai (bottleneck). Dữ liệu mạng xã hội chứa rất nhiều nhiễu và từ lóng ('to the moon', 'priced in'). Nếu Agent này phân tích sai (đầu vào rác - Garbage In), thì các Agent bên dưới sẽ ra quyết định hoàn toàn sai (Garbage Out). Cải thiện được mắt xích này đồng nghĩa với việc nâng cấp độ chính xác của toàn hệ thống."

---

## Slide 4: Tại Sao Lại Chọn Qwen3-4B Làm Foundation Model?
**Nội dung trên Slide:**
- **Kích thước lý tưởng:** 4 Tỷ tham số (Chỉ chiếm ~2GB RAM khi lượng tử hóa, chạy mượt trên mọi GPU).
- **Ngữ cảnh siêu rộng:** 32,000 tokens (Nạp được cùng lúc 20-30 bài báo).
- **Kiến trúc tối ưu:** Hỗ trợ Function Calling và xử lý định dạng cấu trúc (JSON/Markdown) cực tốt.

**Kịch bản nói (Script):**
> "Để làm mô hình nền tảng, em đã chọn Qwen3 bản 4 Tỷ tham số. Mô hình này là điểm cân bằng hoàn hảo: nó đủ nhỏ để load vào bộ nhớ card đồ họa bình thường như RTX 3060/4060, nhưng lại có cửa sổ ngữ cảnh lên tới 32.000 tokens, cho phép đọc rất nhiều tin tức cùng lúc. Quan trọng hơn, dòng Qwen nổi tiếng với khả năng tuân thủ định dạng nghiêm ngặt, rất phù hợp làm Agent tự động sinh báo cáo chuẩn form."

---

## Slide 5: Phương Pháp 1 - Knowledge Distillation (Chưng Cất Tri Thức)
**Nội dung trên Slide:**
- **Mô hình Thầy (Teacher):** `gpt-oss-120b` (117 Tỷ tham số).
- **Mô hình Trò (Student):** `Qwen3-4B` (4 Tỷ tham số).
- **Phương pháp thực thi:** Black-box KD thông qua Supervised Fine-Tuning (SFT).
- Trò học theo bộ "Golden Responses" do Thầy sinh ra.

**Kịch bản nói (Script):**
> "Đi vào phần phương pháp kỹ thuật. Kỹ thuật đầu tiên là Knowledge Distillation hay Chưng cất tri thức. Thay vì ép mô hình nhỏ tự học từ số không, em dùng mô hình gpt-oss-120b với 117 tỷ tham số đóng vai trò làm 'Thầy'. 
> Do giao tiếp qua API (Black-box), Thầy sẽ phân tích dữ liệu thô và sinh ra 1440 bài mẫu chuẩn mực, gọi là Golden Responses. Sau đó, mô hình Qwen3-4B đóng vai 'Trò' sẽ dùng phương pháp Supervised Fine-Tuning để học cách bắt chước chính xác phong cách, cấu trúc và logic phân tích tài chính của Thầy."

---

## Slide 6: Hiệu Quả Kinh Tế (ROI) Của Knowledge Distillation
**Nội dung trên Slide:**
- **Chi phí Chưng cất (1 lần duy nhất):** 1440 mẫu × ~1,500 tokens = ~2.16M tokens → **Chỉ tốn ~$2**.
- **Chi phí Vận hành (Nếu dùng API truyền thống):** 60 tickers × 365 ngày = Hàng chục triệu tokens → **Tốn ~$30 - $50/tháng**.
- **Kết quả:** Trả phí $2 một lần, sở hữu mô hình chạy offline tốc độ cao (~0.3s) mãi mãi.

**Kịch bản nói (Script):**
> "Kỹ thuật chưng cất này mang lại tỷ suất hoàn vốn (ROI) cực cao. Để tạo 1440 mẫu dữ liệu huấn luyện, em chỉ tốn khoảng hơn 2 triệu tokens API, tương đương với việc trả 2 đô-la một lần duy nhất. 
> Thử làm bài toán so sánh: Nếu không chưng cất mà gọi API hàng ngày cho hệ thống giao dịch, ta sẽ tốn hàng chục đô mỗi tháng, chưa kể bị giới hạn API. Bỏ ra 2 đô-la ban đầu, hệ thống thu về một mô hình độc quyền, chạy offline bảo mật và thời gian phản hồi chỉ dưới nửa giây."

---

## Slide 7: Phương Pháp 2 - QLoRA (Quantized Low-Rank Adaptation)
**Nội dung trên Slide:**
- **Full Fine-Tuning vs QLoRA:** 
  - Full FT: Cập nhật 100% tham số (4B) → Cần ~32GB VRAM.
  - QLoRA: Lượng tử hóa mô hình gốc về 4-bit (chỉ đọc) + Huấn luyện adapter LoRA (16-bit).
- **Phép màu của LoRA:** Tách ma trận $W$ thành $A \times B$.
  - Cập nhật vỏn vẹn **~0.78% tham số**.
  - Tiêu thụ VRAM giảm từ 32GB xuống **~5GB VRAM**.

**Kịch bản nói (Script):**
> "Để quá trình dạy 'Trò' diễn ra ngay trên phần cứng phổ thông, em dùng kỹ thuật QLoRA. Nếu huấn luyện toàn bộ 4 tỷ tham số, bộ nhớ card đồ họa sẽ bị thổi phồng lên 32GB do phải lưu trữ gradients và optimizer.
> QLoRA giải quyết bằng hai bước: Thứ nhất, lượng tử hóa (Quantize) mô hình gốc xuống 4-bit và đóng băng nó, thu gọn bộ nhớ từ 8GB xuống 2GB. Thứ hai, dùng LoRA. Thay vì cập nhật ma trận khổng lồ 4096x4096, LoRA chèn thêm hai ma trận nhỏ A và B song song. Nhờ toán học hạ chiều, chúng ta chỉ cần cập nhật vỏn vẹn chưa tới 1% tổng số tham số của mô hình. Nhờ vậy, bộ nhớ huấn luyện giảm xuống chỉ còn khoảng 5GB VRAM, chạy dư sức trên Google Colab hay card máy tính ở nhà."

---

## Slide 8: Khám Phá 7 Ma Trận LoRA (All-Linear) Trong Transformer
**Nội dung trên Slide:**
- Sơ đồ **Qwen3-4B Transformer Block**.
- **Nhóm Attention (`q_proj, k_proj, v_proj, o_proj`)**: Hỗ trợ tập trung, tìm kiếm từ khóa tài chính liên quan.
- **Nhóm FFN (`gate_proj, up_proj, down_proj`)**: Lưu trữ tri thức mới, lọc nhiễu mạng xã hội (qua hàm kích hoạt SiLU/SwiGLU).
- **Trực quan:** Kiến thức gốc (Bách khoa toàn thư đã in) + LoRA Adapter (Tờ giấy Note dán thêm).

**Kịch bản nói (Script):**
> "Slide cuối cùng là cách em cấu hình LoRA bên trong mạng Nơ-ron. Thông thường, để tiết kiệm, người ta chỉ gắn LoRA vào ma trận Query (Q) và Value (V). Tuy nhiên, để phân tích tài chính sâu, em quyết định gắn LoRA vào toàn bộ 7 ma trận tuyến tính cốt lõi (all-linear).
> Bốn ma trận nhóm Attention giúp mô hình nhạy bén hơn trong việc tìm kiếm sự liên kết giữa các từ khóa (như từ 'lãi suất' và 'giá coin'). Ba ma trận nhóm FFN đóng vai trò như bộ não lưu trữ suy luận, giúp lọc bỏ thông tin nhiễu.
> Để dễ hình dung: Mô hình gốc như một cuốn bách khoa toàn thư in đóng bìa cứng, ta không thể tẩy xóa. Việc gắn LoRA tương tự như việc ta dùng một tờ giấy nhớ (sticky note), viết những nhận định tài chính mới lên đó và dán vào trang sách. Khi đọc, hệ thống sẽ kết hợp cả kiến thức nền tảng trong sách và cập nhật mới từ giấy nhớ để đưa ra kết luận sắc bén nhất."
