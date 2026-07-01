# KỊCH BẢN Q&A (GIẢNG VIÊN PHẢN BIỆN)
Tài liệu này tổng hợp các câu hỏi hóc búa giảng viên có thể hỏi từ phần bối cảnh đến hết phần 2.3 và hướng dẫn cách trả lời chuyên sâu.

---

### Câu hỏi 1: Tại sao hệ thống có nhiều Agent mà em chỉ chọn Fine-tune mỗi Sentiment Analyst?
**Cách trả lời (Tự tin & Logic):**
> Thưa thầy, trong kiến trúc hệ thống, Sentiment Analyst đóng vai trò cung cấp dữ liệu nền (Tier 1). Phân tích cảm xúc văn bản tài chính rất khó vì nó chứa nhiều từ lóng mạng xã hội và tín hiệu ngầm (như 'priced in'). Nếu bước này sai, toàn bộ lập luận của Bull/Bear Agent và kết luận của Trader ở các tầng dưới đều là vô nghĩa (hiệu ứng Garbage-In-Garbage-Out). Em tập trung giải quyết cái 'Nút thắt cổ chai' (Bottleneck) này để mang lại tác động lớn nhất cho toàn hệ thống, thay vì phân tán tài nguyên đi Fine-tune các Agent khác.

---

### Câu hỏi 2: Em nói là áp dụng 'Knowledge Distillation' (Chưng cất tri thức), hàm mất mát cụ thể ở đây là gì và công thức như thế nào?
**Cách trả lời (Hiểu bản chất Toán học):**
> Dạ, phương pháp em dùng cụ thể là **Black-box KD (Sequence-Level Knowledge Distillation)** thông qua Supervised Fine-Tuning. Khác với White-box KD truyền thống là tính khoảng cách KL-Divergence trên soft-logits của Teacher, ở đây do Teacher là GPT qua API, em chỉ lấy được Hard-labels (văn bản text đầu ra / Golden Responses).
> Do đó, hàm mất mát (Loss Function) được sử dụng chính là **Causal Language Modeling Loss (Cross-Entropy)**. Công thức là:
> 
> $$
> \mathcal{L}_{SFT} = - \sum_{i=1}^{N} \log P_{\theta}(y_i | x, y_{<i})
> $$
> 
> Trong đó:
> - $x$ là context đầu vào (bài báo tài chính).
> - $y_i$ là token thứ $i$ trong văn bản Golden Response của Teacher.
> - $\theta$ là bộ trọng số của Student (Qwen3-4B).
> 
> Hàm này tính toán log xác suất để ép Student model (Qwen3-4B) phải dự đoán chính xác từng token tiếp theo giống y hệt như những gì Teacher đã phân tích.
> 
> **Ví dụ tính toán cụ thể:**
> Giả sử Teacher phân tích xong và đưa ra kết luận (Golden Response) gồm 2 tokens là: $y_1=$ `"Buy"`, $y_2=$ `"NVDA"`.
> - **Bước 1 (Dự đoán token 1):** Dựa vào tin tức tài chính đầu vào ($x$), Student model dự đoán xác suất từ tiếp theo là `"Buy"` đạt $0.8$ ($80\%$).
>   $\rightarrow$ **Loss 1** $= -\log(0.8) \approx 0.223$
> - **Bước 2 (Dự đoán token 2):** Dựa vào tin tức ($x$) và từ đã biết (`"Buy"`), Student dự đoán từ tiếp theo là `"NVDA"` với xác suất chỉ đạt $0.5$ ($50\%$).
>   $\rightarrow$ **Loss 2** $= -\log(0.5) \approx 0.693$
> - **Tổng Loss toàn chuỗi:** $\mathcal{L}_{SFT} = 0.223 + 0.693 = \mathbf{0.916}$
> 
> Mục tiêu của quá trình huấn luyện là sử dụng đạo hàm để tinh chỉnh trọng số (weights) sao cho tổng Loss này giảm **càng gần 0 càng tốt**. Khi Loss tiến về 0, đồng nghĩa với việc xác suất dự đoán đúng các từ `"Buy NVDA"` của Student sẽ tiến dần tới $1.0$ ($100\%$), tức là Student đã suy nghĩ y hệt như Teacher.

---

### Câu hỏi 2b: Cụ thể quá trình dùng đạo hàm để tinh chỉnh trọng số (Gradient Descent) hoạt động như thế nào sau khi tính được Loss?
**Cách trả lời (Tối ưu hóa - Optimization):**
> Dạ thưa thầy, cốt lõi của việc cập nhật trọng số nằm ở công thức đạo hàm (Gradient) của hàm Cross-Entropy kết hợp với Softmax. Công thức đạo hàm theo điểm số đầu ra (logits) cực kỳ tối giản:
> **$\text{Gradient} = \text{Xác suất Dự đoán} - \text{Thực tế}$**
> 
> Dựa vào ví dụ trên, giả sử từ điển chỉ có 2 từ `"Buy"` và `"Sell"`. Mô hình đang dự đoán $P(\text{"Buy"}) = 0.8$ và $P(\text{"Sell"}) = 0.2$. Từ đúng là `"Buy"`.
> - Với từ đúng `"Buy"` (Thực tế = 1): Gradient $= 0.8 - 1 = \mathbf{-0.2}$
> - Với từ sai `"Sell"` (Thực tế = 0): Gradient $= 0.2 - 0 = \mathbf{+0.2}$
> 
> **Ý nghĩa của Gradient:** 
> Gradient **âm** ($-0.2$) báo hiệu cho mạng nơ-ron phải **TĂNG** điểm số của từ `"Buy"`. Gradient **dương** ($+0.2$) báo hiệu mạng nơ-ron phải **GIẢM** điểm số của từ `"Sell"`.
> 
> Sau đó, thuật toán Lan truyền ngược (Backpropagation) đưa Gradient này vào công thức cập nhật trọng số:
> 
> $$
> W_{mới} = W_{cũ} - \eta \times \text{Gradient}
> $$
> 
> *(Với $\eta$ là Learning Rate)*
> 
> Do Gradient của từ `"Buy"` là số âm, trừ đi số âm thành cộng, nên trọng số $W$ chi phối từ `"Buy"` sẽ được **tăng lên**. Ở lần dự đoán tiếp theo, xác suất của `"Buy"` sẽ nhích từ $0.8$ lên $0.85$, tiến tới mức lý tưởng là $1.0$.

---

### Câu hỏi 3: QLoRA giảm tham số như thế nào? Tại sao lại chọn tham số Rank $r=16$?
**Cách trả lời (Toán học Ma trận):**
> Dạ, LoRA giảm tham số bằng cách không cập nhật toàn bộ ma trận gốc có kích thước $d \times d$ (ví dụ $4096 \times 4096 \approx 16.7$ triệu tham số). Thay vào đó, nó xấp xỉ ma trận thay đổi bằng tích của 2 ma trận nhỏ hơn: $A$ có kích thước ($4096 \times r$) và $B$ có kích thước ($r \times 4096$).
> Khi đặt $r = 16$, tổng số tham số cần cập nhật chỉ là khoảng 131 ngàn tham số, tức là chưa tới 1% ma trận gốc. Việc chọn $r=16$ (kèm $\alpha=32$) là điểm cân bằng lý tưởng. Nếu $r$ quá nhỏ ($r=4, 8$) thì mô hình không đủ 'không gian' để học các khái niệm tài chính phức tạp. Nếu $r$ quá lớn ($r=64$) thì lại dễ bị quá khớp (overfitting) và tốn VRAM mà hiệu năng không tăng đáng kể.

---

### Câu hỏi 4: Thông thường người ta chỉ gắn LoRA vào ma trận Query và Value. Tại sao em lại gắn vào cả 7 ma trận (All-linear)?
**Cách trả lời (Kỹ thuật chuyên sâu):**
> Thưa thầy, tiêu chuẩn cũ chỉ gắn LoRA vào `q_proj` và `v_proj` thường chỉ đủ cho các task đơn giản. Với bài toán phân tích tài chính phức tạp đòi hỏi suy luận, mô hình cần cả sự chú ý (Attention) lẫn tư duy logic.
> Việc gắn thêm vào `k_proj` và `o_proj` giúp mô hình linh hoạt hơn trong việc kết nối các thực thể (như công ty và tin tức). Còn việc gắn LoRA vào mạng FFN (`gate_proj`, `up_proj`, `down_proj`) là vì FFN chứa tới 60% tham số và đóng vai trò như bộ nhớ tri thức. Can thiệp vào FFN giúp mô hình 'lưu trữ' các khái niệm tài chính mới cực kỳ hiệu quả. Với VRAM hiện nay, em hoàn toàn có khả năng train 'all-linear' để tối đa hoá hiệu năng.

---

### Câu hỏi 5: Trong cấu hình em có chỉ số LoRA Alpha (Scaling factor). Ý nghĩa của nó là gì? Tại sao đặt Alpha = 32 khi Rank = 16?
**Cách trả lời (Hyperparameter tuning):**
> Dạ, tham số $\alpha$ (Alpha) được dùng để scale (khuếch đại) trọng số của LoRA trước khi cộng nó vào trọng số Base của mô hình. Tỷ lệ khuếch đại được tính bằng $\frac{\alpha}{r}$. 
> Ở đây, em đặt $\alpha = 32$ và $r = 16$, tức là hệ số scaling $\frac{32}{16} = 2$. Điều này có nghĩa là mọi kiến thức tài chính mới mà LoRA học được sẽ được nhân đôi cường độ tác động. Đặt Alpha lớn hơn Rank (thường gấp đôi) là một best practice giúp mô hình học nhanh hơn và các tín hiệu mới đủ mạnh để ghi đè lên những kiến thức chung chung ban đầu của Base model.

---

### Câu hỏi 6: Lượng tử hóa 4-bit (4-bit Quantization / NF4) ảnh hưởng thế nào đến độ chính xác?
**Cách trả lời (Trade-off):**
> Việc lượng tử hóa từ 16-bit (FP16) xuống 4-bit (NF4) để làm tròn số chắc chắn làm giảm độ chính xác của trọng số gốc. Tuy nhiên, kiến trúc **QLoRA** giải quyết bằng cách: Đóng băng bộ trọng số 4-bit này lại để tính Base, nhưng các ma trận Adapter LoRA (lưu trữ những gì đang được học mới) thì lại được tính toán và cập nhật ở định dạng 16-bit độ chính xác cao (FP16 / BF16). Sự kết hợp này giúp việc suy giảm độ chính xác tổng thể là cực kỳ nhỏ (chỉ 1-2%), nhưng bù lại ta tiết kiệm được tới 80% VRAM (từ 32GB xuống 5GB).

---

### Câu hỏi 7: Trong mạng FFN của kiến trúc này, tại sao mô hình lại sử dụng hàm SwiGLU thay vì ReLU truyền thống? Lợi ích khi gắn LoRA vào `gate_proj` là gì?
**Cách trả lời (Kiến trúc Nơ-ron & Toán học):**
> Dạ, thay vì dùng FFN truyền thống với ReLU, Qwen3-4B sử dụng kiến trúc **SwiGLU** (Swish Gated Linear Unit) tích hợp hàm SiLU. Công thức toán học cốt lõi của FFN lúc này là:
> 
> $$
> \text{FFN}(x) = \Big(\text{SiLU}(x \cdot W_{gate}) \otimes (x \cdot W_{up})\Big) \cdot W_{down}
> $$
> 
> Trong đó:
> - Hàm $\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$ là một hàm trơn (smooth) có đạo hàm liên tục, giúp mô hình không bao giờ bị lỗi 'Dead ReLU' (chết nơ-ron).
> - Ký hiệu $\otimes$ là phép nhân từng phần tử (element-wise multiplication).
> 
> **Giải thích vai trò của `gate_proj`:** 
> Nhìn vào công thức, thành phần $\text{SiLU}(x \cdot W_{gate})$ hoạt động chính xác như một cái **Cổng (Gate)** điều tiết luồng dữ liệu. Giá trị của cổng này sẽ nhân trực tiếp với nhánh thông tin đã được khai triển là $x \cdot W_{up}$. 
> Do đó, khi em gắn LoRA vào ma trận `gate_proj` ($W_{gate}$), mô hình sẽ tự động học được cách điều khiển cái cổng này: Khi gặp tín hiệu tài chính có giá trị (như 'Báo cáo thu nhập tăng'), hệ số cổng sẽ lớn để cho phép thông tin đi qua. Ngược lại, nếu gặp tín hiệu nhiễu vô nghĩa trên mạng xã hội, hệ số cổng sẽ triệt tiêu về 0, đóng cổng và chặn đứng thông tin rác đó truyền đi tiếp xuống `down_proj`.

---

### Câu hỏi 8: Mất bao lâu để train mô hình này và làm sao để biết nó không bị hội chứng 'Catastrophic Forgetting' (Quên thảm khốc)?
**Cách trả lời (Validation):**
> Thời gian huấn luyện (train) trên Colab Pro (GPU T4/A100) mất khoảng 2.5 giờ cho 5 Epochs với tập dữ liệu chưng cất. Để đảm bảo mô hình không bị 'Catastrophic Forgetting' (học cái mới quên cái cũ), giải pháp chính của QLoRA là **Đóng băng hoàn toàn (Frozen)** ma trận gốc 4 Tỷ tham số. Do trí tuệ nền tảng nguyên bản không hề bị thay đổi, mô hình chỉ 'đính kèm' kiến thức mới thông qua LoRA. Kèm theo đó em cũng thiết lập tỷ lệ Dropout 5% cho LoRA để mô hình khái quát hoá (generalize) tốt hơn.
