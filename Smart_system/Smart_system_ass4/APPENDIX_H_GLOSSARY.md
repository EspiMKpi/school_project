# Phụ lục H: Bảng Tra Cứu Thuật Ngữ Chuyên Ngành Việt – Anh

> **Quy ước sư phạm**: Mọi thuật ngữ trong toàn văn báo cáo đều tuân thủ nguyên tắc: giữ nguyên các từ khóa chuyên ngành (keyword/technical word) bằng tiếng Anh khi diễn giải code và bảng thông số, đồng thời cung cấp định nghĩa giải nghĩa tiếng Việt chuẩn xác nhằm đảm bảo tính hàn lâm khoa học và tính ứng dụng kỹ nghệ phần mềm.

| STT | Thuật ngữ Tiếng Anh | Khái niệm Tiếng Việt tương đương | Định nghĩa vắn tắt và Ý nghĩa kỹ thuật |
|:---:|:---|:---|:---|
| 1 | **Accuracy** | Độ chính xác tổng thể | Tỷ lệ giữa số lượng mẫu được phân loại đúng trên tổng số lượng mẫu kiểm thử. Thường gây ngộ nhận nặng khi dữ liệu mất cân bằng. |
| 2 | **Adam (Adaptive Moment Estimation)** | Bộ tối ưu hóa đà thích nghi | Thuật toán cập nhật trọng số kết hợp giữa Moment bậc 1 (đà) và Moment bậc 2 (bình phương độ dốc không định tâm), tự điều chỉnh bước nhảy cho từng tham số. |
| 3 | **Ablation Study** | Phân tích tách biệt thành phần | Phương pháp thực nghiệm cô lập: thêm hoặc bớt duy nhất một thành phần kiến trúc để đo lường chính xác đóng góp biên của cơ chế đó. |
| 4 | **Autograd** | Tự động vi phân | Công cụ tính toán đồ thị tính toán động (computation graph) để suy biến gradient tự động trong PyTorch/TensorFlow. |
| 5 | **Backpropagation** | Lan truyền ngược | Phương pháp áp dụng quy tắc chuỗi vi phân (chain rule) để tính toán gradient hàm mất mát ngược từ tầng cuối về tầng đầu. |
| 6 | **Batch Normalization (BN)** | Chuẩn hóa theo lô | Kỹ thuật chuẩn hóa phân phối đầu ra của các tầng ẩn về mean=0, var=1 theo từng mini-batch, triệt tiêu hiện tượng Internal Covariate Shift. |
| 7 | **Capacity Bottleneck** | Nút thắt dung lượng biểu diễn | Hiện tượng mạng không đủ số lượng kênh hoặc tham số để mã hóa không gian đặc trưng của bài toán phức tạp (như LeNet-5 trên CIFAR-10). |
| 8 | **Categorical Cross-Entropy** | Mất mát entropy chéo đa lớp | Hàm mất mát đo lường khoảng cách Kullback-Leibler giữa phân phối xác suất dự đoán (Softmax) và phân phối nhãn thật (One-Hot). |
| 9 | **Chain Rule** | Quy tắc đạo hàm chuỗi | Cơ sở toán học nền tảng của vi tích phân hàm hợp cho phép tính toán đạo hàm liên tiếp qua nhiều tầng nơ-ron. |
| 10 | **Channel Attention** | Chú ý theo chiều kênh | Cơ chế đánh giá trọng số mức độ quan trọng giữa các bản đồ đặc trưng khác nhau (điển hình là Squeeze-and-Excitation block). |
| 11 | **Confusion Matrix** | Ma trận nhầm lẫn | Bảng biểu diễn số lượng mẫu dự đoán đối sánh với nhãn thật theo từng lớp, phơi bày các điểm mù phân loại. |
| 12 | **Convolutional Layer (Conv2D)** | Tầng tích chập 2D | Tầng áp dụng phép nhân chập giữa các bộ lọc cục bộ (kernels) và bản đồ đặc trưng đầu vào, bảo toàn tính bất biến tịnh tiến. |
| 13 | **Data Augmentation** | Tăng cường dữ liệu | Kỹ thuật biến đổi hình học hoặc quang học mẫu huấn luyện (lật ảnh, xoay, xén ngẫu nhiên) để làm giàu không gian mẫu mà không tốn tham số. |
| 14 | **Dense Layer / Fully Connected** | Tầng kết nối đầy đủ | Tầng nơ-ron truyền thống mà mỗi nơ-ron kết nối với toàn bộ nơ-ron của tầng trước đó thông qua phép nhân ma trận $xW + b$. |
| 15 | **Dropout** | Cơ chế ngắt ngẫu nhiên | Kỹ thuật điều chuẩn (regularization) vô hiệu hóa ngẫu nhiên một tỷ lệ $p$ nơ-ron trong quá trình huấn luyện nhằm chống đồng thích nghi. |
| 16 | **Early Stopping** | Dừng sớm huấn luyện | Chiến lược giám sát loss/độ chính xác trên tập validation để dừng huấn luyện trước khi mô hình rơi vào trạng thái Overfitting. |
| 17 | **Epoch** | Chu kỳ huấn luyện | Một lượt quét trọn vẹn qua toàn bộ các mẫu của tập dữ liệu huấn luyện. |
| 18 | **Fairness Rule** | Quy tắc đối chiếu công bằng | Quy chuẩn thực nghiệm bắt buộc: cùng dữ liệu, cùng phép chia, cùng hạt giống, cùng cấu trúc để đảm bảo kết quả so sánh khách quan. |
| 19 | **Feature Map** | Bản đồ đặc trưng | Tensors đầu ra sau khi một khối tích chập trích xuất các mẫu tín hiệu hình học hoặc ngữ nghĩa. |
| 20 | **Finite Difference** | Sai phân hữu hạn số trị | Phương pháp tính gần đúng đạo hàm bằng công thức đối xứng trung tâm $\frac{f(x+\epsilon)-f(x-\epsilon)}{2\epsilon}$ nhằm kiểm chứng đạo hàm giải tích. |
| 21 | **FLOPs / MACs** | Số phép tính dấu phẩy động | Đơn vị định lượng độ phức tạp tính toán của mô hình (Floating Point Operations / Multiply-Accumulate Operations). |
| 22 | **Forward Pass** | Lan truyền xuôi | Quy trình truyền dữ liệu đầu vào qua chuỗi hàm ánh xạ của các tầng nơ-ron để đưa ra dự đoán xác suất cuối cùng. |
| 23 | **Gradient Checking** | Kiểm tra tính đúng của gradient | Quy trình so sánh đạo hàm tính bằng mã giải tích (NumPy Scratch) với sai phân hữu hạn, yêu cầu sai số tương đối $< 10^{-6}$. |
| 24 | **Global Average Pooling (GAP)** | Gộp trung bình toàn cục | Kỹ thuật tính giá trị trung bình trên toàn bộ không gian $(H, W)$ của từng kênh, nén tensor về vector 1D mà không tốn tham số. |
| 25 | **He / Kaiming Initialization** | Khởi tạo trọng số He | Phương pháp khởi tạo phân phối ngẫu nhiên với phương sai $\sigma^2 = \frac{2}{n_{in}}$, tương thích hoàn hảo với hàm kích hoạt ReLU. |
| 26 | **Inductive Bias** | Thiên lệch quy nạp | Giả định tiên nghiệm mà một mô hình áp đặt lên cấu trúc dữ liệu (ví dụ: CNN áp đặt tính cục bộ không gian và bất biến tịnh tiến). |
| 27 | **Kernel / Filter** | Bộ lọc tích chập | Ma trận trọng số kích thước nhỏ (ví dụ $3 \times 3$ hoặc $5 \times 5$) trượt qua ảnh để trích xuất các đặc trưng biên cạnh, vân và hình khối. |
| 28 | **Learning Rate ($\eta$)** | Tốc độ học | Siêu tham số điều khiển kích thước bước nhảy tham số theo hướng ngược gradient trong không gian tối ưu. |
| 29 | **LeNet-5** | Mạng LeNet-5 (1998) | Kiến trúc mạng tích chập kinh điển của Yann LeCun gồm 2 tầng Conv 5×5 và 3 tầng Dense, đặt nền móng cho thị giác máy tính hiện đại. |
| 30 | **Loss Landscape** | Bề mặt hàm mất mát | Không gian địa hình đa chiều biểu thị hàm mất mát theo không gian tham số trọng số. |
| 31 | **Macro-F1 Score** | Điểm F1 trung bình vĩ mô | Trung bình cộng không trọng số của điểm F1 trên từng lớp. Thước đo trung thực nhất đối với dữ liệu mất cân bằng nghiêm trọng. |
| 32 | **Max Pooling** | Gộp cực đại | Thao tác giảm chiều không gian bản đồ đặc trưng bằng cách giữ lại giá trị lớn nhất trong mỗi cửa sổ trượt (thường $2 \times 2$, stride 2). |
| 33 | **Mini-batch** | Lô dữ liệu con | Tập hợp con các mẫu huấn luyện được nạp đồng thời vào bộ nhớ GPU/CPU để tính toán vector gradient xấp xỉ. |
| 34 | **Model Serialization** | Tuần tự hóa mô hình | Quy trình lưu cấu trúc và vector trọng số đã huấn luyện ra đĩa cứng dưới các định dạng chuẩn (`.npz`, `.keras`, `.pt`). |
| 35 | **Multi-Layer Perceptron (MLP)**| Mạng perceptron đa tầng | Kiến trúc mạng nơ-ron truyền thẳng cổ điển gồm các tầng kết nối đầy đủ (Dense), tối ưu cho dữ liệu bảng không có tính cục bộ không gian. |
| 36 | **Noise Floor** | Sàn nhiễu thống kê | Biên độ dao động tự nhiên của độ chính xác mô hình gây ra thuần túy bởi các hạt giống ngẫu nhiên (random seeds) khác nhau. |
| 37 | **One-Hot Encoding** | Mã hóa 1-nhiệt | Biểu diễn biến phân loại dưới dạng vector nhị phân trong đó chỉ có duy nhất một phần tử mang giá trị 1, các phần tử khác là 0. |
| 38 | **Optimizer** | Bộ tối ưu hóa | Thuật toán toán học chỉ đạo việc cập nhật vector tham số dựa trên đạo hàm tính toán được (ví dụ SGD, Momentum, Adam). |
| 39 | **Overfitting** | Hiện tượng quá khớp | Tình trạng mô hình học thuộc lòng nhiễu thống kê trên tập huấn luyện dẫn đến khả năng tổng quát hóa kém trên tập kiểm thử. |
| 40 | **Padding** | Đệm viền | Thao tác bổ sung các giá trị 0 quanh biên của ảnh/bản đồ đặc trưng để duy trì kích thước không gian sau khi áp dụng tích chập. |
| 41 | **Precision** | Độ chuẩn xác | Tỷ lệ giữa số mẫu dương tính thật (True Positive) trên tổng số mẫu được mô hình dự đoán là dương tính. |
| 42 | **Recall / Sensitivity** | Độ nhạy / Độ bao phủ | Tỷ lệ giữa số mẫu dương tính thật (True Positive) trên tổng số mẫu thực sự mang nhãn dương tính. |
| 43 | **Receptive Field** | Vùng tiếp nhận thông tin | Vùng không gian trên ảnh đầu vào mà một nơ-ron ở tầng sâu có thể bao quát và chịu ảnh hưởng trực tiếp. |
| 44 | **ReLU (Rectified Linear Unit)** | Hàm kích hoạt tuyến tính chỉnh lưu | Hàm kích hoạt $f(x) = \max(0, x)$ giúp chống suy giảm đạo hàm và tăng tốc độ hội tụ so với Sigmoid/Tanh. |
| 45 | **Residual Connection** | Kết nối phần dư / Kết nối tắt | Đường truyền tín hiệu bỏ qua một hoặc nhiều tầng ($y = \mathcal{F}(x) + x$) giúp dòng gradient truyền thẳng không bị triệt tiêu. |
| 46 | **Squeeze-and-Excitation (SE)** | Khối nén và kích hoạt | Mô-đun mạng tính toán trọng số tương quan giữa các kênh bằng cách gộp không gian toàn cục và chiếu qua tầng học phi tuyến. |
| 47 | **Stride** | Bước trượt | Khoảng cách dịch chuyển của cửa sổ bộ lọc tích chập hoặc cửa sổ gộp qua từng bước tính toán. |
| 48 | **Tensor** | Mảng đa chiều | Cấu trúc dữ liệu đại số cơ bản biểu diễn mảng $N$ chiều (ví dụ tensor ảnh $N \times C \times H \times W$). |
| 49 | **Underfitting** | Hiện tượng chưa khớp | Trạng thái mô hình quá đơn giản, chưa đủ năng lực học được mẫu hình tiềm ẩn ngay cả trên tập huấn luyện. |
| 50 | **Vanishing Gradient** | Hiện tượng suy biến độ dốc | Trạng thái vector gradient bị suy giảm cấp số nhân khi lan truyền ngược qua nhiều tầng kích hoạt bão hòa (như Sigmoid). |
