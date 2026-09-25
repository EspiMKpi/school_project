# HƯỚNG DẪN THUYẾT TRÌNH VÀ BẢO VỆ ĐỒ ÁN ASSIGNMENT 4
## Hệ Thống Thông Minh: Đánh Giá Đối Sánh Các Kiến Trúc Deep Learning & Frameworks

> **Tài liệu hướng dẫn nội bộ dành cho sinh viên**: Giúp bạn nắm vững toàn bộ bản chất kỹ thuật, kiến trúc hệ thống, các con số thực nghiệm then chốt và chuẩn bị kịch bản trả lời xuất sắc trước mọi câu hỏi phản biện của Giảng viên.

---

## PHẦN I: TỔNG QUAN HỆ THỐNG & Ý NGHĨA KHOA HỌC

### 1. Mục tiêu cốt lõi của đề tài là gì?
Nếu giảng viên hỏi: *"Đề tài này làm gì và nhằm chứng minh điều gì?"*, bạn trả lời:
> *"Dạ thưa Thầy/Cô, đồ án thực hiện cuộc kiểm chứng thực nghiệm đa chiều nhằm trả lời 3 câu hỏi khoa học nền tảng trong Deep Learning:  
> 1. **Tính tương đương toán học giữa các Framework**: Khi giữ nguyên cùng một kiến trúc toán học và dữ liệu, ba cấp độ triển khai phần mềm gồm **NumPy Scratch (tự viết từ đầu)**, **TensorFlow/Keras (cấp cao)** và **PyTorch (hướng module)** có cho ra cùng một mô hình và kết quả hay không?  
> 2. **Sự tương tác giữa Kiến trúc Mạng và Cấu trúc Dữ liệu (Inductive Bias & Capacity)**: Tại sao mạng MLP kết nối đầy đủ phù hợp cho dữ liệu bảng nhưng bất khả thi trên ảnh lớn; và tại sao kiến trúc LeNet-5 xuất sắc trên chữ số MNIST (98%) nhưng lại suy sụp trên ảnh màu CIFAR-10 (~45%) do nút thắt dung lượng biểu diễn?  
> 3. **Quy luật tiến hóa kiến trúc (Ablation Study $M_1 \to M_4$)**: Việc bổ sung các cơ chế kiến trúc hiện đại (Batch Normalization, Residual Connection, Channel Attention) có luôn làm tăng độ chính xác hay không, và nguyên lý kỹ thuật nào chi phối sự thành bại của chúng?"*

---

### 2. Cấu trúc tổng thể của Project
Dự án được tổ chức chặt chẽ thành các thành phần:

```text
├── 00_inventory.ipynb       : Kiểm kê dữ liệu, đối chiếu số dòng, số cột, phân bố nhãn, seed=42
├── 01_diabetes130.ipynb     : Dữ liệu bảng Diabetes 130-US Hospitals (Tabular MLP 179-128-64-3)
├── 02_mnist.ipynb           : Dữ liệu ảnh MNIST chữ số viết tay (Baseline CNN 2conv+fc)
├── 03_cifar10.ipynb         : Dữ liệu ảnh CIFAR-10 vật thể màu tự nhiên (Baseline SmallCNN)
├── 04_compare.ipynb         : Tổng hợp đối sánh, đo sàn nhiễu (Noise Floor), thang tiến hóa M1->M4
├── 06_mnist_lenet.ipynb     : Khảo sát kiến trúc kinh điển LeNet-5 trên MNIST
├── 07_cifar10_lenet.ipynb   : Khảo sát kiến trúc kinh điển LeNet-5 trên CIFAR-10
├── ass4_utils.py            : Thư viện tiện ích dùng chung (Loader tập trung, Timer, Checkpoint)
├── scratch_nn.py            : Động cơ học sâu tự viết bằng NumPy (Forward, Backward, Adam, GradCheck)
├── variants.py              : Triển khai PyTorch cho 4 biến thể kiến trúc M1, M2, M3, M4
├── results/all_runs.csv     : Bảng kết quả tổng hợp máy đọc từ toàn bộ các đợt chạy thực nghiệm
├── results/models/          : Thư mục lưu trọng số (.npz, .keras, .pt) kèm metadata JSON sidecar
└── pdf/FINAL_REPORT_A4.pdf  : Ấn bản Master Report toàn vẹn 172 trang chuẩn mực
```

---

### 3. Nguyên tắc công bằng cốt lõi (The Fairness Rule)
Đây là luận điểm quan trọng nhất giúp đồ án đạt điểm tối đa về phương pháp luận nghiên cứu:
> **Công thức công bằng (Lecture 04, Slide 29):**  
> $$\text{Same Dataset} + \text{Same Split} + \text{Same Architecture} + \text{Comparable Hyperparameters}$$

1. **Cùng Dữ liệu & Phép chia**: Cả 3 framework đều lấy dữ liệu từ một data loader tập trung (`ass4_utils.py`), dùng chung `Random Seed = 42`.
2. **Cùng Kiến trúc Toán học**: Tổng số tham số học được ($N_{params}$) giữa Scratch NumPy, TensorFlow/Keras và PyTorch **khớp nhau tuyệt đối 100% đến từng tham số đơn lẻ**.
3. **Cùng Siêu tham số**: Cùng Epoch, cùng Batch Size, cùng bộ tối ưu hóa Adam ($\eta=0.001, \beta_1=0.9, \beta_2=0.999$).

---

## PHẦN II: NẮM VỮNG BẢN CHẤT TỪNG BÀI TOÁN THỰC NGHIỆM

### BÀI TOÁN 1: Diabetes 130-US Hospitals (Dữ liệu bảng - Tabular Data)
- **Đặc trưng dữ liệu**: 101,766 hồ sơ bệnh nhân tiểu đường nội trú tại 130 bệnh viện Hoa Kỳ. Sau tiền xử lý (loại bỏ cột rỗng $>95\%$ như `weight`, mã hóa One-Hot các biến hạng mục), tensor đầu vào có **179 đặc trưng** liên tục và nhị phân.
- **Kiến trúc**: MLP 3 tầng ẩn `179 -> 128 -> 64 -> 3` (tổng số tham số: **31,491**).
- **Kết quả 3 Framework**:
  * Scratch NumPy: Accuracy 60.79%, Macro-F1 0.3462 (12.8s)
  * Keras: Accuracy 60.77%, Macro-F1 0.3686 (12.8s)
  * PyTorch: Accuracy 61.05%, Macro-F1 0.3694 (20.9s)
- **Điểm "ăn tiền" cần trình bày với Giảng viên**:
  * **Tại sao dùng MLP mà không dùng CNN?** Dữ liệu bảng không có tính cục bộ không gian (spatial locality). Đảo vị trí cột `BMI` và cột `Age` thì ý nghĩa y tế không đổi, trong khi đảo vị trí điểm ảnh trong ảnh sẽ phá hủy hình dạng vật thể. MLP có Inductive Bias lỏng, không áp đặt giả định không gian.
  * **Nghịch lý Accuracy vs Macro-F1**: Nếu một mô hình ngây thơ luôn đoán tất cả bệnh nhân là `NO` (không tái nhập viện), nó sẽ đạt ngay **53.9% Accuracy** nhưng giá trị y tế bằng 0 vì bỏ sót 100% bệnh nhân nguy cơ cao. Do đó, **Macro-F1 mới là thước đo phản ánh đúng chất lượng**.
  * **Hiện tượng PyTorch chạy chậm hơn NumPy trên CPU**: Do chi phí Overhead quản lý tensor, graph động và đồng bộ bộ nhớ của PyTorch lớn hơn so với tính toán ma trận C-optimized trực tiếp của NumPy khi kích thước mạng quá nhỏ.

---

### BÀI TOÁN 2: MNIST (Chữ số viết tay - Grayscale Images)
- **Đặc trưng dữ liệu**: 70,000 ảnh đơn sắc kích thước $1 \times 28 \times 28$, gồm 10 chữ số (0 đến 9).
- **Kiến trúc**: Baseline `CNN 2conv+fc` (Conv 16 3x3, MaxPool, Conv 32 3x3, MaxPool, Flatten, Dense 128, Output 10). Tổng tham số: **20,490**.
- **Kết quả thực nghiệm**:
  * Trên tập con 10k: Scratch đạt **98.10%**, Keras **97.40%**, PyTorch **98.35%**.
  * Trên toàn bộ 60k: Keras đạt **98.65%**, PyTorch đạt **98.70%**.
- **Điểm "ăn tiền" cần trình bày với Giảng viên**:
  * Cả 3 framework đều hội tụ về cùng một mức chính xác tuyệt hảo (~98.1% - 98.7%). Chênh lệch 0.2% hoàn toàn nằm trong biên độ nhiễu ngẫu nhiên của trọng số khởi tạo (Noise Floor).
  * Khối tích chập chỉ dùng **4,800 tham số** nhưng xử lý được 784 điểm ảnh nhờ cơ chế chia sẻ trọng số (Weight Sharing) và vùng tiếp nhận cục bộ (Local Receptive Fields).

---

### BÀI TOÁN 3: CIFAR-10 (Nhận diện vật thể màu tự nhiên)
- **Đặc trưng dữ liệu**: 60,000 ảnh màu RGB kích thước $3 \times 32 \times 32$, 10 lớp vật thể (máy bay, ô tô, chim, mèo, hươu, chó, ếch, ngựa, tàu thủy, xe tải). Độ biến thiên màu sắc, góc chụp và phông nền cực kỳ phức tạp.
- **Kiến trúc**: `SmallCNN` (Conv 32, MaxPool, Conv 64, MaxPool, Dense 512, Output 10). Tổng tham số: **545,098**.
- **Kết quả thực nghiệm**:
  * Trên tập con 5k: Scratch đạt **51.20%**, Keras **54.50%**, PyTorch **52.90%**.
  * Trên toàn bộ 50k: Keras đạt **70.11%**, PyTorch đạt **71.84%**.
- **Điểm "ăn tiền" cần trình bày với Giảng viên**:
  * Tầng Dense chiếm tới **524,800 tham số (hơn 96% toàn mạng)**, trong khi hai tầng tích chập chỉ chiếm chưa đầy 20k tham số.
  * Phân tích ma trận nhầm lẫn: Cặp nhầm lẫn nặng nhất là **Chó vs Mèo** và **Xe hơi vs Xe tải** vì các đặc trưng hình học cơ bản (tai, mõm, bốn chân hoặc bánh xe, khối hộp) rất gần nhau trong không gian biểu diễn cấp thấp.

---

### BÀI TOÁN 4: Hiện Đại Hóa LeNet-5 — Hai Số Phận Trái Ngược
- **Kiến trúc**: LeNet-5 (LeCun et al., 1998) hiện đại hóa (dùng ReLU + MaxPool + Adam). Tổng tham số: **61,706** trên MNIST và **62,006** trên CIFAR-10.
- **Nghịch lý thực nghiệm**:
  * **Trên MNIST**: LeNet-5 đạt **97.85% (subset)** và **98.68% (full 60k)** — ngang ngửa hoàn toàn với Baseline CNN hiện đại.
  * **Trên CIFAR-10**: LeNet-5 chỉ đạt **45.10% (subset 5k)** và **61.18% (full 50k)** — **TỤT HẬU NẶNG TỪ -9.8% ĐẾN -10.66%** so với SmallCNN.
- **Điểm "ăn tiền" cần giải thích trước Hội đồng**:
  * **Hiện tượng Nút thắt Dung lượng Biểu diễn (Capacity Bottleneck)**: Tầng tích chập C1 của LeNet-5 chỉ có đúng **6 bộ lọc $5 \times 5$**. Sáu bộ lọc này chỉ đủ để bắt các nét thẳng, nét cong của chữ số đen trắng. Khi đối mặt với thế giới tự nhiên của CIFAR-10 (vừa có màu sắc RGB, vừa có vân da, đổ bóng, phông nền), **6 bộ lọc bị cạn kiệt dung lượng biểu diễn**.
  * Khi tăng dữ liệu từ 5k lên 50k, độ chênh lệch của LeNet-5 không hề thu hẹp mà bị giãn rộng thêm (-7.8% -> -10.66%). **Dữ liệu lớn không thể cứu vãn một kiến trúc bị nghẽn dung lượng từ gốc**.
  * **Nghịch lý FLOPs**: Dù LeNet-5 có số tham số gấp 3 lần Baseline trên MNIST (61.7k vs 20.5k), nó lại chạy **nhanh hơn 27% trên CPU** (728s vs 998s) vì số lượng phép tính tích chập (Conv MACs) của LeNet-5 ít hơn tới 2.84 lần.

---

### BÀI TOÁN 5: Chuỗi Cải Tiến Kiến Trúc CNN ($M_1 \to M_4$) trên CIFAR-10
Áp dụng phương pháp nghiên cứu bóc tách thành phần (**Ablation Study**) trên cùng tập dữ liệu 50,000 ảnh CIFAR-10 trong 10 Epochs:

$$\text{Baseline } M_1 \xrightarrow{\text{+ BatchNorm}} M_2 \xrightarrow{\text{+ Residual}} M_3 \xrightarrow{\text{+ SE Attention}} M_4$$

| Mô hình | Cơ chế bổ sung duy nhất | Số tham số | Train Loss | Test Accuracy | Macro-F1 | Đánh giá kỹ thuật |
|:---:|:---|---:|---:|---:|---:|:---|
| **$M_1$** | Conv + ReLU + MaxPool Baseline | 72,730 | 0.8074 | **69.55%** | 0.7001 | Mốc cơ sở tham chiếu |
| **$M_2$** | **+ Batch Normalization** | 72,954 | 0.4579 | **74.86%** | 0.7483 | **Tăng vọt +5.31%** (Hiệu quả cao nhất) |
| **$M_3$** | **+ Residual Shortcut** ($Y = \mathcal{F}(X) + X$) | 75,786 | 0.4479 | **75.74%** | 0.7634 | **Tăng thêm +0.88%** (Đạt đỉnh bảng) |
| **$M_4$** | **+ Squeeze-and-Excitation (Attention)** | 78,614 | **0.4335** | **75.25%** | 0.7557 | **Tụt giảm -0.49%** (Bị Overfitting) |

- **Đúc kết quy luật khoa học cực kỳ sâu sắc**:
  1. **Batch Normalization là cơ chế mang lại giá trị thực tế cao nhất**: Chỉ tốn thêm đúng **224 tham số** (hệ số $\gamma, \beta$) nhưng kéo độ chính xác tăng vọt +5.31% nhờ triệt tiêu hiện tượng Internal Covariate Shift và ổn định phân phối kích hoạt giữa các tầng.
  2. **Residual Connection tạo đường cao tốc gradient**: Phép cộng tắt không vật cản giúp dòng đạo hàm lan truyền thông suốt, giúp mạng đạt đỉnh 75.74%.
  3. **Nghịch lý của Attention ($M_4$)**: $M_4$ đạt **Train Loss thấp nhất toàn bảng (0.4335)** nhưng Test Accuracy lại bị tụt lùi. Do mạng có số kênh hẹp (16/32/64) và huấn luyện trong 10 epochs, các tầng FC trong khối SE bị khớp quá mức (Overfit) vào các tương quan kênh ngẫu nhiên của tập train, làm suy giảm năng lực khái quát hóa trên tập test.
  4. **Quy luật kỹ nghệ**: *"Một cơ chế mới chỉ phát huy tác dụng khi mạng thực sự đối mặt với đúng điểm nghẽn mà cơ chế đó được sinh ra để giải quyết"*. Thêm cơ chế bừa bãi sẽ phản tác dụng.

---

## PHẦN III: THIẾT KẾ ĐỘNG CƠ TỰ VIẾT (SCRATCH NUMPY)

Nếu giảng viên hỏi sâu về phần lập trình tự viết `scratch_nn.py`:
1. **Quy trình tính toán 4 bước**:
   - Bước 1 (Forward): $Z = XW + b$, qua hàm kích hoạt $\text{ReLU}(Z) = \max(0, Z)$.
   - Bước 2 (Loss): Tính Softmax Cross-Entropy với thủ thuật trừ max để tránh tràn số: $\hat{p}_i = \frac{e^{z_i - \max(z)}}{\sum e^{z_j - \max(z)}}$.
   - Bước 3 (Backward): Đạo hàm hàm mất mát đối với logits cực kỳ gọn: $\frac{\partial \mathcal{L}}{\partial Z} = P - Y$. Áp dụng quy tắc chuỗi để tính $dW = X^T \cdot dZ$ và $db = \sum dZ$.
   - Bước 4 (Optimizer): Cài đặt đầy đủ thuật toán **Adam** với hiệu chỉnh độ lệch (Bias Correction) cho moment bậc 1 ($m_t$) và moment bậc 2 ($v_t$).
2. **Kiểm tra đạo hàm số trị (Gradient Checking / Finite Difference)**:
   - Tính đạo hàm số trị bằng công thức sai phân trung tâm: $g_{num} = \frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon}$ với $\epsilon = 10^{-5}$.
   - Đo sai số tương đối: $\text{Relative Error} = \frac{\|g_{analytical} - g_{num}\|_2}{\|g_{analytical}\|_2 + \|g_{num}\|_2}$.
   - Kết quả thực tế đạt mức **$2.95 \times 10^{-7} < 10^{-6}$**, chứng minh toán học rằng toàn bộ công thức lan truyền ngược viết bằng tay là hoàn toàn chính xác!

---

## PHẦN IV: BỘ 10 CÂU HỎI PHẢN BIỆN THƯỜNG GẶP & CÁCH TRẢ LỜI XUẤT SẮC

### Câu 1: Tại sao trong bảng kết quả, PyTorch chạy nhanh hơn Keras hàng chục lần? Có phải PyTorch viết tốt hơn Keras không?
- **Trả lời**: *"Dạ không ạ. Đây là điểm khác biệt cốt tử về nền tảng phần cứng chứ không phải do thiết kế phần mềm. Kể từ bản 2.10, TensorFlow trên Windows chính thức dừng hỗ trợ GPU native và chạy hoàn toàn trên CPU. Trong khi đó, PyTorch được kích hoạt nhân CUDA chạy trên GPU NVIDIA RTX 3060 12GB. Do đó, thời gian của Keras phản ánh năng lực của CPU, còn PyTorch phản ánh sức mạnh tính toán song song hàng nghìn nhân của GPU. Phép so sánh công bằng về mặt thuật toán được thể hiện qua tính tương đồng về độ chính xác và số lượng tham số học được."*

### Câu 2: Tại sao số tham số của 3 framework lại giống nhau tuyệt đối?
- **Trả lời**: *"Dạ vì cả 3 bản cài đặt đều tuân thủ nghiêm ngặt cùng một công thức kiến trúc toán học. Tầng `Dense(in, out)` luôn có $in \times out$ trọng số $W$ và $out$ bias $b$. Tầng `Conv2D(C_in, C_out, k)` luôn có $C_out \times C_in \times k \times k$ trọng số và $C_out$ bias. `S.Dense`, `keras.layers.Dense` và `torch.nn.Linear` chỉ là ba cách gọi tên khác nhau của cùng một hàm ánh xạ đại số tuyến tính."*

### Câu 3: Tại sao trên tập dữ liệu bảng Diabetes, Accuracy đạt 61% nhưng Macro-F1 chỉ đạt ~0.37?
- **Trả lời**: *"Dạ thưa Thầy/Cô, đó là do hiện tượng mất cân bằng nhãn nghiêm trọng (Class Imbalance). Trong tập dữ liệu, lớp không tái nhập viện (`NO`) chiếm đa số áp đảo, trong khi lớp tái nhập viện dưới 30 ngày (`<30`) chỉ chiếm tỷ lệ rất nhỏ (~11%). Khi tính Accuracy, mô hình đoán đúng lớp đa số sẽ kéo chỉ số lên cao. Nhưng khi xét Macro-F1 (tính trung bình không trọng số F1 của từng lớp), điểm F1 của lớp hiếm rất thấp kéo Macro-F1 tụt xuống 0.37. Trong bài toán y tế, bỏ sót bệnh nhân nặng nguy hiểm hơn nhiều so với chẩn đoán nhầm người khỏe, vì vậy Macro-F1 mới là thước đo trung thực."*

### Câu 4: Tại sao LeNet-5 chạy rất tốt trên MNIST nhưng lại tụt hậu gần 10% trên CIFAR-10?
- **Trả lời**: *"Dạ thưa Thầy/Cô, nguyên nhân nằm ở hiện tượng 'Nghẽn dung lượng biểu diễn' (Capacity Bottleneck). Tầng Conv đầu tiên của LeNet-5 chỉ có vỏn vẹn 6 bộ lọc 5x5, được thiết kế năm 1998 cho chữ số đơn sắc viết tay. Chữ số chỉ cần phân biệt nét cong và nét thẳng. Nhưng ảnh màu tự nhiên CIFAR-10 có 3 kênh RGB, nhiều hình khối phức tạp và phông nền đa dạng. Sáu bộ lọc ban đầu không thể trích xuất đủ số lượng đặc trưng cần thiết, làm mất thông tin ngay từ cửa ngõ đầu vào. Ngược lại, SmallCNN có 32 và 64 bộ lọc, gấp hơn 5-10 lần vốn biểu diễn, nên vượt trội hoàn toàn."*

### Câu 5: Dữ liệu lớn có giải quyết được điểm yếu của LeNet-5 trên CIFAR-10 không?
- **Trả lời**: *"Dạ không ạ. Thực nghiệm của nhóm đã chứng minh điều này: Khi tăng tập dữ liệu gấp 10 lần (từ 5,000 ảnh lên 50,000 ảnh), khoảng cách tụt hậu của LeNet-5 so với SmallCNN không hề thu hẹp mà còn bị nới rộng từ -7.8% lên -10.66%. Khi mô hình đã bị nghẽn dung lượng (Underfitting do kiến trúc quá hẹp), việc nạp thêm dữ liệu không thể giúp mô hình thông minh hơn vì nó không còn dung lượng tham số để hấp thụ dữ liệu đó."*

### Câu 6: Trong 4 biến thể M1 đến M4, tại sao M4 có Attention lại có kết quả kém hơn M3?
- **Trả lời**: *"Dạ thưa Thầy/Cô, $M_4$ đạt Train Loss thấp nhất toàn bảng (0.4335) nhưng Test Accuracy lại tụt giảm -0.49% so với $M_3$. Đây là biểu hiện kinh điển của hiện tượng quá khớp (Overfitting). Khối Squeeze-and-Excitation bổ sung thêm hai tầng Fully Connected để học trọng số chú ý kênh. Với số kênh hẹp (16/32/64) và chỉ huấn luyện 10 Epochs, cơ chế Attention đã học thuộc lòng các đặc trưng cục bộ ngẫu nhiên của tập train thay vì học các mẫu hình tổng quát. Bài học rút ra là không phải cứ thêm cơ chế phức tạp là mô hình sẽ tốt lên."*

### Câu 7: Cơ chế Batch Normalization hoạt động ra sao mà lại tăng vọt +5.31% độ chính xác?
- **Trả lời**: *"Dạ Batch Normalization chuẩn hóa đầu ra của các tầng ẩn về phân phối chuẩn (mean=0, variance=1) theo từng mini-batch, sau đó nhân với tham số tỉ lệ $\gamma$ và cộng độ dời $\beta$ để mạng tự học mức độ biểu diễn tối ưu. Cơ chế này triệt tiêu hiện tượng dịch chuyển phân phối nội tại (Internal Covariate Shift), giúp bề mặt hàm mất mát (loss landscape) trở nên trơn tru hơn, cho phép gradient lan truyền ổn định và tránh được tình trạng các nơ-ron rơi vào vùng bão hòa."*

### Câu 8: Nhóm đã kiểm tra tính đúng đắn của phần code tự viết NumPy Scratch như thế nào?
- **Trả lời**: *"Dạ nhóm sử dụng kỹ thuật Kiểm tra Gradient số trị (Finite Difference Gradient Checking). Nhóm so sánh gradient giải tích được tính từ mã nguồn lan truyền ngược viết tay với gradient số trị tính từ sai phân trung tâm với bước nhảy $\epsilon = 10^{-5}$. Sai số tương đối giữa hai phương pháp trên mọi tầng mạng đều đạt mức dưới $10^{-6}$ (thực tế đạt $2.95 \times 10^{-7}$). Điều này bảo đảm mã nguồn tự viết hoàn toàn không có lỗi sai sót công thức toán học."*

### Câu 9: Sàn nhiễu (Noise Floor) là gì và nó chứng minh điều gì?
- **Trả lời**: *"Dạ Sàn nhiễu (Noise Floor) là biên độ dao động tự nhiên của kết quả khi chạy cùng một mô hình với các hạt giống ngẫu nhiên (random seeds) khác nhau. Trong thực nghiệm của nhóm với 5 seed ngẫu nhiên trên Notebook 04, độ lệch chuẩn dao động từ 0.02 đến 0.04 (2% đến 4%). Do đó, sự chênh lệch nhỏ dưới 0.5% giữa NumPy, Keras và PyTorch trên cùng một tập dữ liệu hoàn toàn nằm trong sàn nhiễu thống kê tự nhiên, chứng minh rằng sự khác biệt về framework không làm thay đổi bản chất của mô hình."*

### Câu 10: Mô hình đã huấn luyện được lưu trữ dưới định dạng nào và có thể nạp lại để demo độc lập không?
- **Trả lời**: *"Dạ toàn bộ mô hình đã huấn luyện được lưu trữ đầy đủ trong thư mục `results/models/` theo ba định dạng đặc thù của từng framework:  
1. **NumPy Scratch**: Lưu mảng nén `.npz` chứa từng ma trận trọng số $W, b$.  
2. **TensorFlow/Keras**: Lưu tệp `.keras` tự chứa cả cấu trúc đồ thị tính toán lẫn trọng số (Self-Contained Graph).  
3. **PyTorch**: Lưu `state_dict` dưới dạng `.pt`.  
Mỗi tệp mô hình đều đi kèm một tệp JSON sidecar lưu giữ lý lịch huấn luyện (số tham số, độ chính xác, siêu tham số), cho phép nạp lại và chạy suy diễn (inference) tức thì mà không cần huấn luyện lại."*

---

## PHẦN V: KỊCH BẢN THUYẾT TRÌNH BẢO VỆ GỢI Ý (10 - 15 PHÚT)

- **Phút 1 - 2: Mở đầu & Phương pháp luận**: Giới thiệu đề tài, cấu trúc 3 bài toán (Bảng, Ảnh đơn sắc, Ảnh màu), và nhấn mạnh nguyên tắc công bằng cốt lõi (**The Fairness Rule**).
- **Phút 3 - 5: Động cơ tự viết Scratch NumPy & Chứng minh tương đương toán học**: Trình bày vòng lặp 4 bước, kỹ thuật Gradient Checking với sai số $< 10^{-6}$, và bảng so sánh chứng minh 3 framework có số tham số khớp 100%.
- **Phút 6 - 8: Phân tích 3 bài toán thực tế**: 
  * Diabetes: Phân tích vì sao chọn MLP, mổ xẻ nghịch lý Accuracy 84% vs Macro-F1 0.37 do mất cân bằng nhãn.
  * MNIST & CIFAR-10: Đối chiếu hiệu năng, phân tích sự chi phối của tầng Dense trong SmallCNN và các cặp nhầm lẫn chính.
- **Phút 9 - 11: Nghiên cứu chuyên sâu LeNet-5 & Chuỗi cải tiến M1..M4**:
  * Trình bày hiện tượng nút thắt dung lượng (Capacity Bottleneck) của LeNet-5 trên CIFAR-10.
  * Phân tích bóc tách thành phần $M_1 \to M_4$: Đóng góp vượt trội của BatchNorm (+5.31%) và bài học quá khớp của Attention (-0.49%).
- **Phút 12 - 13: Mô hình hóa & Báo cáo ấn bản 172 trang**: Giới thiệu hệ thống checkpoint `results/models/` và ấn bản Master PDF 172 trang (`FINAL_REPORT_A4.pdf`).
- **Phút 14 - 15: Kết luận & Lời cảm ơn**: Đúc kết 4 bài học lớn về mối quan hệ giữa Toán học, Kiến trúc và Framework. Sẵn sàng nhận câu hỏi từ Hội đồng.
