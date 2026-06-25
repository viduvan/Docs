# TỔNG HỢP CÂU HỎI ÔN TẬP TENSORFLOW & DEEP LEARNING (PHẦN 1 - 4)

*Tài liệu ôn tập tổng hợp toàn bộ câu hỏi và giải thích chi tiết từ Phần 1 đến Phần 4.*

## MỤC LỤC
1. [Phần 1: Khái niệm cơ bản TensorFlow & CNNs](#phần-1-khái-niệm-cơ-bản-tensorflow--cnns)
2. [Phần 2: ImageDataGenerator, Transfer Learning & Tránh Overfitting](#phần-2-imagedatagenerator-transfer-learning--tránh-overfitting)
3. [Phần 3: Xử lý ngôn ngữ tự nhiên (NLP) & Chuỗi số nguyên](#phần-3-xử-lý-ngôn-ngữ-tự-nhiên-nlp--chuỗi-số-nguyên)
4. [Phần 4: Chuỗi thời gian (Time Series), Đánh giá hiệu năng & DNN/RNN](#phần-4-chuỗi-thời-gian-time-series-đánh-giá-hiệu-năng--dnnrnn)

---

## PHẦN 1: KHÁI NIỆM CƠ BẢN TENSORFLOW & CNNS
### Câu 1: In machine learning with TensorFlow, how is the relationship between inputs and outputs rearranged compared to traditional programming?
- A. We put rules and data in and get answers out.
- **B. We put answers and data in and get rules out.** *(Đáp án đúng)*
- C. We put rules and answers in and get data out.
- D. We only put data in and get nothing out.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Trong lập trình truyền thống (traditional programming), chúng ta đưa các quy tắc (rules) và dữ liệu (data) vào hệ thống để nhận được câu trả lời (answers). Ngược lại, học máy (machine learning) đảo ngược quy trình này bằng cách đưa câu trả lời (answers) và dữ liệu (data) vào để huấn luyện hệ thống tự tìm ra các quy tắc (rules).

### Câu 2: What does the `Tensor.dtype` attribute represent in TensorFlow?
- A. The size of the tensor along each of its axes.
- B. The number of dimensions of the tensor.
- **C. The type of all the elements in the tensor.** *(Đáp án đúng)*
- D. The memory allocation address of the tensor.

**Đáp án:** **C**

> [!NOTE]
> **Giải thích:** Các đối tượng tf.Tensor đại diện cho các mảng đa chiều. Trong đó, thuộc tính Tensor.dtype thể hiện kiểu dữ liệu của toàn bộ các phần tử bên trong tensor đó. Kích thước dọc theo mỗi trục của tensor được đại diện bởi thuộc tính Tensor.shape.

### Câu 3: Which of the following is the high-level API of the TensorFlow platform designed for modern deep learning and fast experimentation?
- A. NumPy
- **B. Keras** *(Đáp án đúng)*
- C. SciPy
- D. Matplotlib

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Keras là API cấp cao của nền tảng TensorFlow. Keras cung cấp giao diện dễ tiếp cận và hiệu quả cao để thiết lập các mô hình học sâu hiện đại, hỗ trợ các nhà phát triển thử nghiệm ý tưởng của họ một cách nhanh chóng. Các cấu trúc dữ liệu cốt lõi của Keras là các lớp (layers) và mô hình (models).

### Câu 4: What is the main purpose of utilizing a `Flatten` layer as the first layer in a `Sequential` neural network model for image classification?
- A. To apply a 2D convolution over the input pixels.
- B. To compress the image size by selecting the maximum values.
- **C. To transform a multi-dimensional square matrix into a simple linear array.** *(Đáp án đúng)*
- D. To normalize pixel values so they fall between 0 and 1.

**Đáp án:** **C**

> [!NOTE]
> **Giải thích:** Đối với các tập dữ liệu ảnh như MNIST hoặc Fashion-MNIST gồm các ảnh vuông ma trận 28×28 pixel , lớp Flatten sẽ đóng vai trò chuyển đổi ma trận hai chiều này thành một mảng tuyến tính một chiều đơn giản để làm đầu vào phù hợp cho các lớp ẩn tiếp theo.

### Câu 5: Which callback in TensorFlow Keras monitors validation metrics and stops the training process early if there is no improvement after a specified number of epochs?
- A. ModelCheckpoint
- B. TensorBoard
- **C. EarlyStopping** *(Đáp án đúng)*
- D. LearningRateScheduler

**Đáp án:** **C**

> [!NOTE]
> **Giải thích:** Callback EarlyStopping theo dõi các chỉ số quan trọng (như validation loss) trong suốt quá trình huấn luyện và sẽ tự động dừng việc đào tạo sớm nếu mô hình không có sự cải thiện nào sau một số lượng epoch được định nghĩa trước qua tham số patience. Điều này giúp tránh lãng phí tài nguyên tính toán và ngăn chặn hiện tượng quá khớp (overfitting).

### Câu 6: What is the main purpose of a pooling layer (such as `MaxPooling2D` with a 2x2 pool) in a Convolutional Neural Network?
- A. To increase the number of color channels in the input image.
- **B. To compress the image by keeping only the largest value in every four pixels.** *(Đáp án đúng)*
- C. To apply a 3x3 filter that emphasizes vertical lines in the image.
- D. To flatten the multi-dimensional tensor into a linear array.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Pooling là một cách để nén ảnh. Cụ thể với Max Pooling kích thước 2x2, cứ mỗi vùng gồm 4 pixel, giá trị lớn nhất (biggest pixel) sẽ được giữ lại và đi tiếp qua lớp sau.

### Câu 7: How does the `ImageDataGenerator` class in Keras automatically generate labels for your training images?
- A. By analyzing the dominant color patterns of each image.
- B. By reading the file format extensions of the images.
- **C. By pointing it at a directory where the sub-directories automatically represent the class labels.** *(Đáp án đúng)*
- D. By requiring the user to write a custom Python dictionary of image paths and labels.

**Đáp án:** **C**

> [!NOTE]
> **Giải thích:** Khi sử dụng ImageDataGenerator, bạn chỉ cần chỉ định đường dẫn đến thư mục chứa ảnh. Trình tạo ảnh này sẽ tự động nhận diện các thư mục con bên trong (ví dụ: thư mục "horses" và "humans") để tự động gán nhãn tương ứng cho các hình ảnh được tải lên.

### Câu 8: When compiling a Keras model in TensorFlow, what is the primary role of the loss function?
- A. To load and split the training dataset from a directory on disk.
- **B. To measure how well or how badly the model did after making a guess.** *(Đáp án đúng)*
- C. To save the best model weights to a file whenever validation metrics improve.
- D. To convert multi-dimensional matrices into a flat 1D array.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Trong quá trình biên dịch (compile) mô hình, hàm mất mát (loss function) được thiết lập để đo lường mức độ chính xác hoặc sai lệch của mô hình sau khi đưa ra dự đoán so với nhãn thực tế. Từ đó, bộ tối ưu hóa (optimizer) sẽ điều chỉnh lại các tham số.

### Câu 9: What is the primary purpose of the tf.image module in TensorFlow?
- A. To host web-based visualization tools like TensorBoard.
- **B. To contain various functions for image processing and data augmentation, such as flipping, rotating, and adjusting brightness.** *(Đáp án đúng)*
- C. To run distributed processing on mobile platforms like iOS and Android.
- D. To automatically build sequential deep neural networks.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Module tf.image chứa nhiều hàm chức năng hỗ trợ xử lý hình ảnh và xây dựng đường dẫn tăng cường dữ liệu (data augmentation), bao gồm lật ảnh (tf.image.flip_left_right), chuyển ảnh RGB sang xám (rgb_to_grayscale), điều chỉnh độ sáng (adjust_brightness), hay cắt ảnh trung tâm (central_crop).

### Câu 10: Which of the following hardware accelerators was built by Google specifically for machine learning and tailored to optimize TensorFlow?
- A. T4 GPU
- **B. Tensor Processing Unit (TPU)** *(Đáp án đúng)*
- C. Multi-core CPU
- D. CUDA GPU

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Tensor Processing Unit (TPU) là một mạch tích hợp chuyên dụng (ASIC - chip phần cứng) được Google thiết kế dành riêng cho các tác vụ học máy và được tối ưu hóa đặc biệt cho nền tảng TensorFlow.

### Câu 11: Who originally developed TensorFlow and what was its primary initial purpose?
- A. Developed by Facebook AI Research for academic studies.
- B. Developed by Microsoft Research for operating system optimizations.
- **C. Developed by the Google Brain team for internal Google use in research and production.** *(Đáp án đúng)*
- D. Developed by Keras Core team for high-level image editing applications.

**Đáp án:** **C**

> [!NOTE]
> **Giải thích:** TensorFlow ban đầu được phát triển bởi đội ngũ Google Brain phục vụ cho các mục đích nghiên cứu và sản xuất nội bộ của Google.

### Câu 12: In Google Colaboratory, what step is required before you can start coding with TensorFlow?
- A. You must manually install it using !pip install tensorflow.
- B. You need to download the TensorFlow installer file and run it.
- **C. No setup or manual installation is required; you just need to import tensorflow as tf and start coding.** *(Đáp án đúng)*
- D. You must compile the TensorFlow source code from GitHub within the container.

**Đáp án:** **C**

> [!NOTE]
> **Giải thích:** Google Colab là một môi trường hosted Jupyter notebook miễn phí và không yêu cầu thiết lập phức tạp. TensorFlow đã được cài đặt sẵn và tối ưu hóa trên hệ thống này, bạn chỉ cần gọi import tensorflow as tf để bắt đầu lập trình.

### Câu 13: What does the tf.keras.Model.fit method do in the Keras API workflow?
- A. It returns the loss and metrics values for the model.
- B. It generates output predictions for the input samples.
- **C. It trains the model for a fixed number of epochs.** *(Đáp án đúng)*
- D. It configures the model with an optimizer and loss function.

**Đáp án:** **C**

> [!NOTE]
> **Giải thích:** Phương thức Model.fit thực hiện nhiệm vụ huấn luyện mô hình trên tập dữ liệu với một số lượng epoch (chu kỳ) cố định.

### Câu 14: What is the Fashion-MNIST dataset composed of?
- A. 50,000 training and 50,000 testing color images of handwritten numbers.
- **B. 60,000 training and 10,000 testing grayscale images of 10 classes of clothing articles from Zalando.** *(Đáp án đúng)*
- C. 100,000 high-resolution photos of horses and humans.
- D. 10,000 training and 60,000 testing 3D MRI medical scans.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Fashion-MNIST là tập dữ liệu hình ảnh sản phẩm của Zalando, bao gồm 60.000 mẫu huấn luyện và 10.000 mẫu kiểm tra. Mỗi mẫu là một ảnh xám (grayscale) kích thước 28x28 tương ứng với một nhãn thuộc 10 nhóm sản phẩm thời trang khác nhau.

### Câu 15: How should pixel values of images in datasets like Fashion-MNIST be normalized before training, and why?
- A. By multiplying every value by 255 to increase the color depth.
- B. By subtracting the mean and dividing by 255 to change the values between 0 and 1.
- **C. By dividing every pixel value by 255 to scale them to a range between 0 and 1.** *(Đáp án đúng)*
- D. By converting all grayscale values to negative values.

**Đáp án:** **C**

> [!NOTE]
> **Giải thích:** Để chuẩn hóa dữ liệu ảnh đầu vào, ta đưa toàn bộ giá trị điểm ảnh về khoảng từ 0 đến 1 bằng cách chia tất cả các giá trị pixel cho 255.

### Câu 16: What represents the fundamental abstraction of a layer (tf.keras.layers.Layer) in Keras?
- A. It only defines the learning rate scheduler of the model.
- **B. It encapsulates a state (weights) and some computation (defined in the call method).** *(Đáp án đúng)*
- C. It serves strictly as an input container with no trainable parameters.
- D. It acts as a database directory configuration for loading files.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Trong Keras, lớp Layer là một trừu tượng hóa cơ bản. Nó đóng gói trạng thái (các trọng số - weights) và một số phép tính toán cốt lõi được định nghĩa trong phương thức call của lớp đó.

### Câu 17: Which callback in TensorFlow Keras allows you to save the model weights to a file whenever the validation loss improves?
- A. TensorBoard
- **B. ModelCheckpoint** *(Đáp án đúng)*
- C. EarlyStopping
- D. LearningRateScheduler

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Callback ModelCheckpoint chịu trách nhiệm lưu lại các trọng số của mô hình vào một tệp tin (mặc định là 'model_checkpoint.h5') mỗi khi độ lỗi trên tập kiểm định (validation loss) được cải thiện tốt hơn.

### Câu 18: What is a convolution filter designed to do in a Convolutional Neural Network (CNN)?
- A. Randomly shuffle the pixels to augment the dataset.
- B. Compress the image by taking the maximum value in a window.
- **C. Modify the image pixels so that specific features (like vertical or horizontal lines) are emphasized.** *(Đáp án đúng)*
- D. Convert color RGB channels into single-byte representations.

**Đáp án:** **C**

> [!NOTE]
> **Giải thích:** Bộ lọc tích chập (convolution filter) thay đổi hình ảnh theo cách làm nổi bật và nhấn mạnh các đặc trưng cụ thể của ảnh như các đường kẻ dọc hoặc kẻ ngang.

### Câu 19: When configuring a `Conv2D` layer in Keras as tf.keras.layers.`Conv2D`(64, (3,3), activation='relu', input_shape=(28,28,1)), what does the final '1' in input_shape represent?
- A. The number of layers in the sequential model.
- **B. A single byte for color depth (grayscale channel).** *(Đáp án đúng)*
- C. The stride value of the convolution operation.
- D. The index of the first training sample.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Giá trị số 1 ở cuối của input_shape=(28, 28, 1) biểu thị việc mô hình đang xử lý ảnh sử dụng một byte đơn cho chiều sâu màu (tương ứng với ảnh xám thông thường).

### Câu 20: In binary image classification tasks (such as classifying horses vs. humans), what setup is conventionally used for the final dense layer in Keras?
- A. A Dense layer with 10 neurons and 'softmax' activation.
- **B. A Dense layer with 1 neuron and 'sigmoid' activation.** *(Đáp án đúng)*
- C. A Dense layer with 128 neurons and 'relu' activation.
- D. A Dense layer with 2 neurons and 'linear' activation.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Đối với phân loại nhị phân (binary classification), lớp Dense cuối cùng chỉ cần 1 nơ-ron kết hợp với hàm kích hoạt sigmoid để trả ra giá trị xác suất từ 0 đến 1 (ví dụ: giá trị 0 đại diện cho lớp 'horses' và 1 đại diện cho 'humans').

### Câu 21: How are TensorFlow APIs arranged structurally, and which level do machine learning researchers typically use to explore new algorithms?
- A. They are arranged hierarchically, with low-level APIs built on top of high-level APIs; researchers use high-level APIs.
- **B. They are arranged hierarchically, with high-level APIs built on low-level APIs; researchers use low-level APIs.** *(Đáp án đúng)*
- C. They are arranged horizontally, with all APIs having equal abstraction; researchers use both equally.
- D. They are arranged in isolated containers; researchers only write custom APIs from scratch.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Các API của TensorFlow được sắp xếp theo cấu trúc phân tầng (hierarchically), trong đó các API cấp cao (high-level APIs) được xây dựng dựa trên nền tảng của các API cấp thấp (low-level APIs). Các nhà nghiên cứu máy học sử dụng các low-level APIs để tạo ra và khám phá các thuật toán học máy mới.

### Câu 22: In TensorFlow, if we want to write a custom callback to stop training when a specific metric is met (e.g., loss < 0.4), what Keras class must we inherit from, and what method must we override?
- **A. Inherit from tf.keras.callbacks.Callback and override on_epoch_end.** *(Đáp án đúng)*
- B. Inherit from tf.keras.Model and override fit.
- C. Inherit from tf.keras.layers.Layer and override call.
- D. Inherit from tf.keras.optimizers.Optimizer and override minimize.

**Đáp án:** **A**

> [!NOTE]
> **Giải thích:** Để tạo một bộ kiểm soát huấn luyện tự tùy chỉnh (custom callback) nhằm can thiệp vào các giai đoạn huấn luyện, chúng ta kế thừa từ lớp cha tf.keras.callbacks.Callback và ghi đè phương thức on_epoch_end để kiểm tra chỉ số vào cuối mỗi chu kỳ (epoch).

### Câu 23: When creating a custom callback to stop training early, what exact line of code is used inside the callback class to halt the training process?
- A. tf.keras.backend.clear_session()
- **B. self.model.stop_training = True** *(Đáp án đúng)*
- C. model.fit(stop=True)
- D. sys.exit(0)

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Bên trong lớp callback tự định nghĩa, khi điều kiện kiểm tra độ lỗi đạt ngưỡng mong muốn (ví dụ: logs.get('loss') < 0.4), thuộc tính self.model.stop_training = True được gọi để báo hiệu hệ thống lập tức hủy bỏ quá trình huấn luyện hiện tại.

### Câu 24: Why does a 28x28 input image output a tensor size of 26x26 after passing through a `Conv2D` layer with a 3x3 filter without padding?
- A. The MaxPooling layer automatically crops 1 pixel from each side of the image.
- **B. Because you cannot calculate the convolution for the edge pixels since they don't have enough neighbors.** *(Đáp án đúng)*
- C. The Flatten layer compresses the dimensions.
- D. The batch size dimension is subtracted from the image height and width.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Phép toán tích chập (convolution) trên một điểm ảnh (pixel) yêu cầu các điểm ảnh lân cận xung quanh (neighbors) tương ứng với kích thước bộ lọc (ví dụ: bộ lọc 3x3). Vì các điểm ảnh ở rìa ngoài cùng (edges) của bức ảnh 28x28 không có đủ điểm ảnh lân cận bao quanh, bộ lọc không thể thực hiện phép tính tại đó. Kết quả là vùng biên ảnh bị thu hẹp đi 2 pixel ở mỗi chiều, làm kích thước ảnh đầu ra giảm xuống còn 26x26.

### Câu 25: What does the '3' represent in the input_shape=(150, 150, 3) configuration when defining a `Conv2D` layer for a complex image model like Horses vs. Humans?
- A. The stride value of 3 pixels for the convolution filter.
- B. The size of the convolution filter matrix (3x3).
- **C. The color depth channel representing RGB (3 bytes for color depth).** *(Đáp án đúng)*
- D. The number of dense hidden layers in the model.

**Đáp án:** **C**

> [!NOTE]
> **Giải thích:** Giá trị số 3 ở cuối cấu hình input_shape đại diện cho số kênh màu của bức ảnh đầu vào, cụ thể là ảnh màu có đủ 3 kênh màu RGB (Red, Green, Blue) sử dụng 3 byte cho chiều sâu màu. Ngược lại, giá trị 1 (như trong ảnh Fashion-MNIST) biểu thị ảnh xám chỉ sử dụng một byte đơn.

### Câu 26: In a binary classification model (e.g., horses vs. humans) using Keras, if the final layer outputs predictions via a sigmoid activation function, how are the output classes determined during inference?
- A. Values close to 0 represent a human, while values close to 1 represent a horse.
- **B. Any value greater than 0.5 represents a human (class value 1), and values of 0.5 or less represent a horse (class value 0).** *(Đáp án đúng)*
- C. All output values are automatically rounded to the nearest integer index from 0 to 9.
- D. A value of exactly 0.5 means both subjects are present in the image.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Hàm kích hoạt sigmoid đưa ra giá trị đầu ra nằm trong khoảng từ 0 đến 1. Trong đoạn mã dự đoán thực tế của tài liệu, nếu giá trị xác suất trả về từ mô hình classes > 0.5 thì bức ảnh được gán nhãn là "human" (lớp 1); các giá trị nhỏ hơn hoặc bằng 0.5 đại diện cho nhãn "horse" (lớp 0).

### Câu 27: When preparing validation data using `ImageDataGenerator` with a binary crossentropy loss, what should the class_mode parameter be set to in the `flow_from_directory` method?
- A. class_mode='categorical'
- **B. class_mode='binary'** *(Đáp án đúng)*
- C. class_mode='sparse'
- D. class_mode='grayscale'

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Do mô hình Horses vs. Humans sử dụng hàm mất mát phân loại nhị phân (binary_crossentropy), nhãn dữ liệu sinh ra từ bộ phát ảnh tự động cần phải ở dạng nhị phân. Vì vậy, tham số class_mode trong hàm flow_from_directory phải được định cấu hình chính xác là 'binary'.

### Câu 28: What is the impact of reducing the target size of the images generated by `ImageDataGenerator` (e.g., from 300x300 down to 150x150)?
- A. It will have no effect on the neural network architecture or training speed.
- **B. It will affect both the architecture (changing tensor shapes in subsequent layers) and the overall performance of the model.** *(Đáp án đúng)*
- C. It will automatically convert color images to grayscale images.
- D. It requires you to change the optimizer from RMSprop to Adam.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Tài liệu ghi nhận rõ ràng việc giảm kích thước ảnh đích (target size) của trình tạo ảnh sẽ gây ảnh hưởng trực tiếp đến kiến trúc mạng (thay đổi cấu trúc shape của các lớp nơ-ron kế tiếp) cũng như hiệu năng tổng thể của mô hình.

### Câu 29: Which programming languages does TensorFlow support for developing machine learning applications?
- A. Python only.
- **B. Python, JavaScript, C++, and Java.** *(Đáp án đúng)*
- C. Python, R, and Swift only.
- D. Python, HTML, and CSS.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** TensorFlow là một nền tảng mã nguồn mở vô cùng linh hoạt, cho phép các nhà phát triển xây dựng và triển khai mô hình học sâu trên nhiều ngôn ngữ khác nhau bao gồm Python, JavaScript, C++, và Java.

### Câu 30: When running a model in Google Colab and visualizing its graph structure, which tool is integrated directly inside Colab notebooks using a magic command (e.g. %tensorboard --logdir=logs)?
- A. Matplotlib
- **B. TensorBoard** *(Đáp án đúng)*
- C. KerasCV
- D. TensorFlow Datasets

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Công cụ trực quan hóa đồ thị tính toán và đo lường tham số huấn luyện cốt lõi của Google được gọi là TensorBoard. Nó được tích hợp dễ dàng và khởi chạy ngay trong môi trường Colab bằng lệnh magic %tensorboard --logdir=logs.


---

## PHẦN 2: IMAGEDATAGENERATOR, TRANSFER LEARNING & TRÁNH OVERFITTING
### Câu 1: Image Augmentation Parameters
In Keras' `ImageDataGenerator`, which parameter specifies the range in degrees (0–180) within which to randomly rotate images during training to avoid overfitting?
- A. zoom_range
- **B. rotation_range** *(Đáp án đúng)*
- C. shear_range
- D. rotation_degree

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Tham số rotation_range là một giá trị tính bằng độ (từ 0–180) xác định khoảng mà mô hình sẽ tự động xoay các hình ảnh ngẫu nhiên trong quá trình huấn luyện nhằm tăng độ đa dạng của dữ liệu.

### Câu 2: Causes of Overfitting
According to the course materials, which of the following is NOT a common cause of overfitting in machine learning models?
- A. The training data size is too small.
- **B. The model complexity is too low, preventing it from learning any noise.** *(Đáp án đúng)*
- C. The model trains for too long on a single sample set of data.
- D. The training data contains large amounts of irrelevant information (noisy data).

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Overfitting (quá khớp) thường xảy ra khi độ phức tạp của mô hình cao (khiến nó ghi nhớ luôn cả nhiễu trong dữ liệu huấn luyện). Các nguyên nhân chính gây overfitting bao gồm: tập dữ liệu huấn luyện quá nhỏ, dữ liệu chứa nhiều thông tin không liên quan (nhiễu), huấn luyện quá lâu trên một tập dữ liệu, hoặc mô hình có độ phức tạp cao.

### Câu 3: Transfer Learning Settings (include_top)
When loading a pre-trained model like InceptionV3 for transfer learning, what is the purpose of setting the parameter `include_top` to False?
- **A. To specify that we want to ignore the fully-connected layer at the top and get straight to the convolutions.** *(Đáp án đúng)*
- B. To automatically freeze all the convolutional layers of the base model.
- C. To change the required input shape of the network to a smaller size.
- D. To indicate that we want to retrain all the layers from scratch without loading pre-trained weights.

**Đáp án:** **A**

> [!NOTE]
> **Giải thích:** Mô hình InceptionV3 được thiết kế sẵn một lớp kết nối đầy đủ (fully-connected layer) ở trên cùng để phân loại lớp gốc. Khi thiết lập include_top=False, chúng ta muốn loại bỏ lớp phân loại này để đi thẳng tới các tầng trích xuất đặc trưng tích chập (convolutions), phục vụ cho việc thêm lớp phân loại mới phù hợp với bài toán hiện tại.

### Câu 4: The Dropout Technique
Why does the Dropout technique work effectively in preventing overfitting within a neural network?
- A. It physically deletes neurons with low weights during the compilation phase to make the network lighter.
- **B. It prevents neighboring neurons from ending up with similar weights and prevents neurons from over-specializing on inputs from the previous layer.** *(Đáp án đúng)*
- C. It increases the image size from 150x150 to 300x300 pixels automatically.
- D. It permanently freezes all parameters of the deep learning model.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Kỹ thuật Dropout hoạt động bằng cách loại bỏ ngẫu nhiên một số lượng nơ-ron trong quá trình huấn luyện. Điều này mang lại hiệu quả cao vì: (1) giúp các nơ-ron lân cận tránh việc cập nhật trọng số quá giống nhau (dẫn đến overfitting); và (2) tránh việc một nơ-ron bị phụ thuộc quá mức vào đầu ra của nơ-ron ở lớp trước đó và trở nên chuyên biệt hóa quá mức.

### Câu 5: Multi-Class Classification Configuration
When setting up Keras utilities and model compilation parameters for a multi-class classification task (such as the Rock-Paper-Scissors dataset with 3 classes), which of the following combinations should be used?
- A. class_mode='binary', activation='sigmoid', loss='binary_crossentropy'
- **B. class_mode='categorical', activation='softmax', loss='categorical_crossentropy'** *(Đáp án đúng)*
- C. class_mode='categorical', activation='sigmoid', loss='categorical_crossentropy'
- D. class_mode='binary', activation='softmax', loss='binary_crossentropy'

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Đối với bài toán phân loại đa lớp (từ 3 lớp trở lên như Kéo-Búa-Bao) , ImageDataGenerator cần thiết lập class_mode='categorical'. Lớp đầu ra của mô hình nơ-ron cần sử dụng số lượng nơ-ron bằng số lượng lớp (ví dụ: 3) kết hợp với hàm kích hoạt softmax để tính xác suất phân phối. Khi biên dịch mô hình, hàm loss phải được thiết lập là categorical_crossentropy.

### Câu 6: ImageDataGenerator Directory Flow
In Keras, what is the default behavior of the `flow_from_directory` method when reading images from subfolders?
- A. It resizes all images to 150x150 and performs K-fold cross-validation during load time.
- **B. It automatically labels the images based on the subdirectory names.** *(Đáp án đúng)*
- C. It freezes the weights of the layers within the parent directory.
- D. It automatically crops and deletes corrupted images.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Phương thức flow_from_directory được thiết kế để tự động đọc cấu trúc thư mục phân cấp (như thư mục chứa Cats và Dogs riêng biệt) , từ đó tự động gán nhãn cho hình ảnh dựa trên tên của chính thư mục con chứa chúng.

### Câu 7: Batch Size vs. Steps Per Epoch
What does the training parameter `batch_size` represent?
- A. The total number of images included in the validation set.
- **B. The number of samples processed in one single iteration.** *(Đáp án đúng)*
- C. The number of batches to process before considering one epoch complete.
- D. The learning rate multiplier applied to the RMSprop optimizer.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** batch_size đại diện cho số lượng mẫu dữ liệu được mô hình xử lý cùng một lúc trong một lần lặp (iteration). Trọng số của mô hình sẽ được cập nhật liên tục ngay sau khi xử lý xong từng lô (batch) dữ liệu này. Trong khi đó, số lượng batch cần xử lý trước khi kết thúc một epoch được gọi là steps_per_epoch.

### Câu 8: Grounding Concept of Transfer Learning
Why is Transfer Learning exceptionally effective for image classification, even when the new task has very limited data?
- **A. Deep neural networks trained on images share a curious phenomenon: in the early layers, the model learns generic, low-level features like edges, colors, and intensity variations.** *(Đáp án đúng)*
- B. Pre-trained weights automatically eliminate the need to compile the model or specify a loss function.
- C. It forces the training process to halt immediately when validation accuracy drops.
- D. It guarantees that the training process takes less than 1 epoch to achieve 100% accuracy.

**Đáp án:** **A**

> [!NOTE]
> **Giải thích:** Trong học chuyển giao (Transfer Learning), mô hình gốc đã học được các đặc trưng cấp thấp mang tính tổng quát (như phát hiện các đường biên, màu sắc, cường độ sáng) ở các lớp đầu tiên. Những đặc trưng nền tảng này hoàn toàn tương thích và áp dụng được cho các bài toán phân loại hình ảnh mới , giúp tăng tốc thời gian huấn luyện và cải thiện hiệu suất tối ưu trên tập dữ liệu nhỏ.

### Câu 9: Preventing Overfitting (Early Stopping)
How does the "Early stopping" technique prevent a machine learning model from overfitting during training?
- A. It identifies unimportant features and removes them from the architecture.
- **B. It pauses the training phase before the machine learning model starts to learn the noise in the data.** *(Đáp án đúng)*
- C. It merges several weak learners into a single strong algorithm.
- D. It increases the dropout rate incrementally after each iteration.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Kỹ thuật "Early stopping" (dừng sớm) sẽ theo dõi hiệu suất của mô hình trên tập kiểm thử (validation set) và tạm dừng giai đoạn huấn luyện trước khi mô hình bắt đầu học sâu vào các chi tiết thừa thãi hoặc nhiễu của tập dữ liệu huấn luyện chính.

### Câu 10: InceptionV3 Architecture Design
Which module or design technique is utilized in Google's InceptionV3 to capture features at different scales while maintaining computational efficiency?
- A. A fully-connected layer at the very bottom of the model architecture.
- **B. An Inception module that uses various filter sizes (1x1, 3x3, and 5x5 convolutions) in parallel, along with batch normalization and factorized convolutions.** *(Đáp án đúng)*
- C. An exclusive reliance on global average pooling layers instead of standard convolutions.
- D. A strict 2-layer sequential structure with no convolutional operations.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Kiến trúc InceptionV3 thiết kế một khối mô-đun đặc biệt (Inception module) sử dụng song song các bộ lọc tích chập có kích thước khác nhau (1x1, 3x3, và 5x5) để trích xuất đặc trưng đa quy mô. Việc tích hợp chuẩn hóa theo lô (batch normalization) và kỹ thuật tích chập nhân tố hóa (factorized convolutions) giúp giảm thiểu tối đa chi phí tính toán khi mô hình vận hành.

### Câu 11: Effect of Convolution on Image Dimensions
If we feed an image of spatial dimension 150x150 into a `Conv2D` layer with a 3x3 filter and no padding, what is the output spatial dimension of the feature map?
- A. 150x150
- **B. 148x148** *(Đáp án đúng)*
- C. 147x147
- D. 75x75

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Theo tài liệu bài giảng, khi hình ảnh ban đầu có kích thước 150x150 đi qua tầng tích chập Conv2D đầu tiên (với bộ lọc kích thước 3x3 và không sử dụng padding), kích thước của ảnh sẽ bị giảm đi 2 pixel ở mỗi chiều, kết quả đầu ra thu được là 148x148.

### Câu 12: Weight Freezing in Keras
In transfer learning, which line of code is used to freeze the weights of a pre-trained model's layers to prevent them from being updated during the training process?
- **A. layer.trainable = False** *(Đáp án đúng)*
- B. layer.freeze = True
- C. model.trainable = False
- D. layer.lock()

**Đáp án:** **A**

> [!NOTE]
> **Giải thích:** Để đóng băng các trọng số của mô hình gốc (pre-trained model) và tận dụng các đặc trưng chung đã học, chúng ta duyệt qua các tầng tích chập và khóa chúng lại bằng cách thiết lập thuộc tính layer.trainable = False. Điều này đảm bảo các lớp này sẽ không trải qua quá trình cập nhật trọng số khi huấn luyện tác vụ mới.

### Câu 13: Choosing the Extraction Boundary Layer (mixed7)
When implementing transfer learning with the InceptionV3 architecture as described in the slides, which specific intermediate layer is chosen as the boundary to extract features?
- A. mixed10
- B. dense_final
- **C. mixed7** *(Đáp án đúng)*
- D. conv2d_last

**Đáp án:** **C**

> [!NOTE]
> **Giải thích:** Trong thực hành huấn luyện học chuyển giao với InceptionV3, lớp mixed7 được chỉ định làm lớp cuối cùng lấy từ mô hình gốc (last_layer). Chúng ta sẽ lấy đầu ra của lớp này (last_output) để kết nối với các tầng phân loại tùy chỉnh được thêm vào phía sau.

### Câu 14: K-fold Cross-validation
How does the K-fold cross-validation technique function to detect whether a machine learning model is overfitting?
- A. It trains K independent models and averages their predictions to get a single score.
- **B. It divides the training dataset into K equally sized subsets, using one subset as validation data and training the model on the remaining K-1 subsets.** *(Đáp án đúng)*
- C. It permanently drops out K% of the neurons during every single training iteration.
- D. It rescales the images by a factor of 1/K during data flow.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** K-fold cross-validation hoạt động bằng cách chia tập dữ liệu huấn luyện thành K phần (folds) bằng nhau. Quá trình huấn luyện diễn ra qua nhiều lượt, mỗi lượt chọn ra 1 phần làm dữ liệu kiểm thử (validation) và huấn luyện mô hình trên K-1 phần còn lại, giúp đánh giá khách quan khả năng tổng quát hóa của mô hình trên dữ liệu mới.

### Câu 15: Fine-tuning Definition in Transfer Learning
What is the primary definition of "Fine-tuning" in the context of transfer learning?
- A. Retraining all layers of the pre-trained model from scratch with completely random initial weights.
- B. Changing the image resolution dynamically from 300x300 to 150x150 during training.
- **C. Using the dataset from the new challenge to retrain selected layers of the pre-trained model to adapt their parameters.** *(Đáp án đúng)*
- D. Manually adjusting the learning rate after each iteration.

**Đáp án:** **C**

> [!NOTE]
> **Giải thích:** Tinh chỉnh (Fine-tuning) là quá trình sử dụng chính tập dữ liệu của tác vụ mới để huấn luyện lại một số lớp nhất định được lựa chọn từ mô hình pre-trained. Việc này giúp tinh chỉnh các tham số của mô hình gốc để chúng tương thích sâu và hoạt động tối ưu hơn với mục tiêu phân loại mới.

### Câu 16: Disadvantages of Transfer Learning
Which of the following combinations represents the primary disadvantages of utilizing Transfer Learning listed in the course?
- A. Speeding up the training process and better performance.
- B. High storage requirements and slower convergence rates.
- **C. Domain mismatch, overfitting, and complexity.** *(Đáp án đúng)*
- D. Incapability to handle small datasets and automatic loss of pre-trained weights.

**Đáp án:** **C**

> [!NOTE]
> **Giải thích:** Mặc dù mang lại hiệu năng cao và giải quyết bài toán thiếu dữ liệu, học chuyển giao vẫn có các nhược điểm cần lưu ý bao gồm: lệch miền dữ liệu (domain mismatch - dữ liệu gốc quá khác biệt dữ liệu mới), nguy cơ quá khớp (overfitting), và độ phức tạp trong cấu trúc mô hình.

### Câu 17: Ensembling Technique
Which machine learning technique combines predictions from several separate, potentially inaccurate "weak learners" to obtain a more accurate final prediction?
- A. Regularization
- B. Pruning
- **C. Ensembling** *(Đáp án đúng)*
- D. Early stopping

**Đáp án:** **C**

> [!NOTE]
> **Giải thích:** Ensembling (học máy kết hợp/học đồng bộ) là phương pháp kết hợp các dự đoán của nhiều thuật toán độc lập (vốn được gọi là các weak learners - bộ học yếu vì kết quả riêng lẻ dễ sai lệch) để tổng hợp lại thành một kết quả dự đoán chung có độ chính xác cao hơn rất nhiều.

### Câu 18: Rock-Paper-Scissors Dataset Specs
In the multi-class classification lectures, what are the dataset specifications of the Rock-Paper-Scissors dataset?
- A. 25,000 grayscale images of hands, sized 28x28 pixels.
- **B. 2,892 color images of diverse hands posed into Rock/Paper/Scissors, sized 300x300 pixels in 24-bit color.** *(Đáp án đúng)*
- C. 10,000 images of horses and humans, sized 150x150 pixels.
- D. 2,892 binary images of signs, sized 150x150 pixels.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Tập dữ liệu Kéo-Búa-Bao (Rock-Paper-Scissors) bao gồm 2.892 ảnh màu của nhiều bàn tay khác nhau thuộc nhiều chủng tộc, giới tính, độ tuổi khác nhau. Mỗi bức ảnh có kích thước 300x300 pixel định dạng màu 24-bit.

### Câu 19: ImageDataGenerator Parameter (fill_mode)
When randomly rotating or shifting images using Keras' `ImageDataGenerator`, which parameter specifies the strategy used for filling in newly created empty pixels?
- **A. fill_mode** *(Đáp án đúng)*
- B. padding_strategy
- C. nearest_neighbor
- D. interpolation_mode

**Đáp án:** **A**

> [!NOTE]
> **Giải thích:** Khi thực hiện xoay ảnh hoặc dịch chuyển ảnh theo chiều ngang/dọc, các vùng khoảng trống rìa ảnh mới sẽ được tạo ra. Tham số fill_mode (ví dụ: fill_mode='nearest') chỉ định thuật toán hoặc chiến lược để tự động điền giá trị cho các điểm ảnh trống này.

### Câu 20: Steps Per Epoch Definition
What does the training parameter `steps_per_epoch` represent in the model training configuration?
- A. The number of samples processed in one single iteration.
- **B. The number of batches to process before moving on to the next epoch.** *(Đáp án đúng)*
- C. The total number of validation rounds performed during training.
- D. The total number of convolutional layers in the sequential model.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Khác với batch_size là số mẫu xử lý trong một lần lặp , steps_per_epoch định nghĩa số lượng lô dữ liệu (batches) cần được xử lý hoàn tất trước khi kết thúc một epoch (chu kỳ huấn luyện) để chuyển sang epoch tiếp theo.

### Câu 21: RMSprop Learning Rate for Transfer Learning
When compiling the customized transfer learning model with the InceptionV3 base, which specific small learning rate is set for the `RMSprop` optimizer to prevent disrupting the pre-trained weights?
- A. learning_rate=0.1
- **B. learning_rate=0.0001 (or 1e-4)** *(Đáp án đúng)*
- C. learning_rate=0.5
- D. learning_rate=1.0

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Khi thực hiện học chuyển giao, các trọng số ở các tầng tích chập của mô hình cơ sở vốn đã rất tốt. Do đó, khi biên dịch mô hình mới, chúng ta cần thiết lập tốc độ học cực kỳ nhỏ cho RMSprop, cụ thể là 1e-4 (tức 0.0001) để mô hình cập nhật trọng số từ từ và ổn định, không làm phá vỡ các đặc trưng đã học được trước đó.


---

## PHẦN 3: XỬ LÝ NGÔN NGỮ TỰ NHIÊN (NLP) & CHUỖI SỐ NGUYÊN
### PART 1: INTRODUCTION TO WORD ENCODINGS & TOKENIZATION

#### Câu 1: Why is using standard ASCII character values insufficient for helping a computer truly understand the sentiment or meaning of words in a text dataset?
- A. ASCII encodings require too much memory compared to word-based encodings.
- **B. Characters only represent letters, meaning anagram words like "LISTEN" and "SILENT" share the exact same characters but have completely different meanings.** *(Đáp án đúng)*
- C. ASCII values cannot be converted into float tensors for neural network training.
- D. Character encodings can only be applied to image pixel datasets.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Theo tài liệu, mã hóa ký tự (như mã ASCII) gán giá trị số cho từng chữ cái đơn lẻ. Cách này không giúp máy tính hiểu nghĩa của từ. Điển hình là hai từ "LISTEN" và "SILENT" chứa cùng một tập hợp ký tự giống hệt nhau nhưng có ý nghĩa hoàn toàn khác nhau. Nếu chỉ dùng mã hóa ký tự, máy tính không thể phân biệt được sự khác biệt ngữ nghĩa này.

#### Câu 2: If the sentence "I love my dog" is encoded as `[1, 2, 3, 4]`, how would the sentence "I love my cat" be encoded in Keras Tokenizer if "cat" is a newly encountered word?
- A. `[1, 2, 3, 4]`
- **B. `[1, 2, 3, 5]`** *(Đáp án đúng)*
- C. `[1, 3, 2, 5]`
- D. `[1, 2, 3]`

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Tài liệu chỉ ra rằng khi từ "I love my" đã được mã hóa tương ứng là các số 1, 2, 3, ta sẽ tái sử dụng lại các token cũ này. Đối với từ mới xuất hiện lần đầu là "cat", Tokenizer sẽ tự động tạo một token số nguyên mới kế tiếp, cụ thể là số 5. Do đó, chuỗi mã hóa cho câu mới là `[1, 2, 3, 5]`.

#### Câu 3: Which property of the Keras Tokenizer class should you access to retrieve the dictionary mapping of words (keys) to their corresponding integer tokens (values)?
- **A. tokenizer.word_index** *(Đáp án đúng)*
- B. tokenizer.index_word
- C. tokenizer.word_dictionary
- D. tokenizer.fit_on_texts

**Đáp án:** **A**

> [!NOTE]
> **Giải thích:** Lớp Tokenizer trong Keras cung cấp thuộc tính word_index. Thuộc tính này trả về một từ điển (dictionary) chứa các cặp key-value, trong đó key là từ (dạng text) và value là token số nguyên đại diện cho từ đó.

#### Câu 4: What is the purpose of passing the `num_words` parameter when initializing Keras Tokenizer(`num_words`=N)?
- A. It sets the maximum sequence length allowed for padded input arrays.
- **B. It specifies the maximum number of words to keep in the tokenizer’s vocabulary, focusing on the most frequent words.** *(Đáp án đúng)*
- C. It limits the training dataset size to N rows.
- D. It configures the embedding vector dimension size.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Tham số num_words truyền vào lớp khởi tạo Tokenizer đóng vai trò giới hạn số lượng từ vựng tối đa mà mô hình sẽ giữ lại và học từ tập dữ liệu đầu vào. Kế hoạch này giúp mô hình tập trung vào các từ phổ biến nhất và bỏ qua các từ cực kỳ hiếm gặp để tránh mô hình bị quá tải từ vựng.

### PART 2: TEXT TO SEQUENCE & OOV HANDLING

#### Câu 5: By default, if a Tokenizer fits only on "I love my dog" and is then used to encode "I really love my dog" using `texts_to_sequences`, what happens to the word "really"?
- A. It is automatically assigned a token value of 0.
- B. It throws a KeyError runtime exception.
- **C. It is lost/ignored because the word is not present in the Tokenizer's existing word_index.** *(Đáp án đúng)*
- D. It is represented by a random number generator.

**Đáp án:** **C**

> [!NOTE]
> **Giải thích:** Theo tài liệu, câu "I really love my dog" khi chuyển đổi qua texts_to_sequences sẽ vẫn được mã hóa thành `[1, 2, 3, 4]` (tương ứng với "I love my dog"). Từ "really" bị bỏ qua hoàn toàn và biến mất khỏi chuỗi đầu ra vì nó không có sẵn trong thuộc tính word_index.

#### Câu 6: Given a fitted Keras Tokenizer with `word_index` = {'my': 1, 'love': 2, 'dog': 3}, what is the exact output of `texts_to_sequences`(['my dog loves my manatee'])?
- **A. `[1, 3, 1]`** *(Đáp án đúng)*
- B. `[1, 2, 3, 1]`
- C. `[1, 3, 1, 2]`
- D. `[1, 3]`

**Đáp án:** **A**

> [!NOTE]
> **Giải thích:** Vì từ "loves" và "manatee" không nằm trong từ điển word_index của Tokenizer, chúng sẽ bị bỏ qua trong quá trình chuyển văn bản thành chuỗi số nguyên. Chuỗi kết quả đầu ra chỉ giữ lại mã của "my", "dog" và "my", tức là mảng số nguyên `[1, 3, 1]`.

#### Câu 7: How does establishing an Out-Of-Vocabulary token (e.g., oov_token="<OOV>") help prevent the loss of sentence structure when evaluating unseen validation data?
- A. It expands the vocabulary dynamically by searching Google for synonyms.
- **B. It replaces unknown words with a designated special token instead of skipping them, thereby preserving the original length and relative positions of words.** *(Đáp án đúng)*
- C. It automatically flags and filters out grammar mistakes.
- D. It translates foreign words back to English before encoding.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Khi thiết lập oov_token, Tokenizer sẽ không bỏ qua các từ chưa biết nữa mà thay thế chúng bằng token đặc biệt này (thường là <OOV>). Điều này giúp duy trì cấu trúc độ dài và nhịp điệu của câu thay vì làm mất thông tin vị trí các từ như khi bỏ qua mặc định.

### PART 3: ADVANCED TOKENIZATION WITH TENSORFLOW TEXT

#### Câu 8: Which tokenizer provided by the tensorflow_text package is most basic, splitting strings strictly based on ICU-defined spaces, tabs, and newlines?
- A. UnicodeScriptTokenizer
- **B. WhitespaceTokenizer** *(Đáp án đúng)*
- C. BertTokenizer
- D. SentencepieceTokenizer

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Lớp text.WhitespaceTokenizer là bộ tách từ cơ bản nhất trong thư viện tensorflow_text. Nó hoạt động đơn giản bằng cách tách các chuỗi ký tự tại bất kỳ vị trí nào có khoảng trắng được ICU định nghĩa (như phím cách, tab, dòng mới). Công cụ này rất phù hợp để nhanh chóng thử nghiệm và xây dựng các nguyên mẫu mô hình ban đầu.

#### Câu 9: What distinguishes `UnicodeScriptTokenizer` from a simple `WhitespaceTokenizer` during text preprocessing?
- A. It splits text strictly into character ASCII code integers.
- **B. It segments strings at Unicode script boundaries, ensuring punctuation marks (like periods and commas) are separated from adjacent words.** *(Đáp án đúng)*
- C. It ignores all non-English text scripts.
- D. It requires an active internet connection to download pre-trained scripts.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** UnicodeScriptTokenizer chia chuỗi dựa trên ranh giới script của Unicode (tương ứng với các giá trị UScriptCode của ICU). Do đó, nó có thể tách rời các dấu câu (như dấu chấm, dấu phẩy, dấu chấm hỏi) ra khỏi từ đứng trước nó, thay vì dính liền như WhitespaceTokenizer thông thường.

#### Câu 10: Which tensorflow_text tokenizer should be preferred for CJK (Chinese, Japanese, Korean) languages where text characters are not typically separated by spaces?
- A. WhitespaceTokenizer
- **B. UnicodeCharTokenizer** *(Đáp án đúng)*
- C. RegexSplitter
- D. SplitMergeTokenizer

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Các ngôn ngữ như tiếng Trung, tiếng Nhật, tiếng Hàn (CJK) không có khoảng trắng phân tách giữa các từ. Do đó, UnicodeCharTokenizer là lựa chọn lý tưởng vì nó tự động phân chia chuỗi văn bản đầu vào thành các ký tự UTF-8 đơn lẻ.

#### Câu 11: In a `SplitMergeTokenizer`, how are the control labels 0 and 1 interpreted to define string segments?
- A. 0 means standard character, 1 means specialized emoji.
- **B. 0 indicates the start of a new string segment, while 1 indicates that the character should be merged into the current active string segment.** *(Đáp án đúng)*
- C. 0 represents an upper-case word, 1 represents lower-case.
- D. 0 splits punctuation, 1 splits numerical numbers.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Đối với bộ phân tách từ SplitMergeTokenizer, giá trị kiểm soát 0 được quy định dùng để biểu thị điểm bắt đầu của một chuỗi mới. Trong khi đó, giá trị 1 biểu thị rằng ký tự hiện tại vẫn là một phần của chuỗi đang xử lý và cần được gộp vào chuỗi đó.

#### Câu 12: Which tokenizer is designed to accept output probabilities (logit value pairs) from a neural network to predict if characters should start a new word or merge?
- **A. SplitMergeFromLogitsTokenizer** *(Đáp án đúng)*
- B. BertTokenizer
- C. SentencepieceTokenizer
- D. HubModuleTokenizer

**Đáp án:** **A**

> [!NOTE]
> **Giải thích:** Khác với SplitMergeTokenizer nhận nhãn cứng trực tiếp, SplitMergeFromLogitsTokenizer nhận các cặp giá trị logit từ một mạng nơ-ron để dự đoán xem mỗi ký tự nên được tách sang một chuỗi mới (0) hay hợp nhất vào chuỗi hiện tại (1).

#### Câu 13: Which of the following statement is true regarding Keras and TensorFlow Text "Detokenization"?
- A. Detokenization is guaranteed to reconstruct the original string with 100% precision.
- **B. Detokenization combines tokens to reconstruct the string, but the process can be lossy, meaning the output may not always match the original pre-tokenized text exactly.** *(Đáp án đúng)*
- C. Detokenization permanently deletes the embedding layer weights.
- D. Detokenization only works on numeric datasets.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Theo slide 3.3, các Tokenizer triển khai giao diện Detokenizer cung cấp phương thức detokenize để ghép nối các chuỗi. Tuy nhiên, tiến trình này có khả năng bị hao hụt thông tin (lossy), vì thế chuỗi nhận được sau khi giải mã không phải lúc nào cũng trùng khớp hoàn toàn với chuỗi thô ban đầu trước khi phân rã.

### PART 4: PADDING & TRUNCATING SEQUENCES

#### Câu 14: In Keras `pad_sequences`, what is the default value of the padding parameter and where does it insert the zeros?
- A. padding='post', inserting zeros at the end of the sequence.
- **B. padding='pre', inserting zeros at the beginning of the sequence.** *(Đáp án đúng)*
- C. padding='center', placing zeros equally on both sides.
- D. padding='none', raising error for uneven length.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Trong Keras, hành vi mặc định của lớp pad_sequences là thêm các số 0 vào phía trước của chuỗi số nguyên (padding='pre'). Việc này giúp đưa các chuỗi có độ dài khác nhau về cùng kích thước thống nhất để tạo thành một ma trận vuông vắn trước khi nạp vào mạng nơ-ron tuần hoàn.

#### Câu 15: When sentences in your dataset exceed the maximum length specified by the `maxlen` parameter, what is the default truncation behavior?
- A. It raises a ValueError because the sentence exceeds the limits.
- B. It truncates words from the end of the sentence ('post').
- **C. It truncates words from the beginning of the sentence ('pre').** *(Đáp án đúng)*
- D. It deletes the entire sentence sequence.

**Đáp án:** **C**

> [!NOTE]
> **Giải thích:** Theo tài liệu, giá trị mặc định của thuộc tính cắt chuỗi là 'pre'. Nghĩa là nếu câu văn dài hơn độ dài tối đa cấu hình (maxlen), phần thông tin ở đầu câu sẽ bị tự động cắt bỏ và chấp nhận mất thông tin tại đây.

#### Câu 16: If you wish to configure `pad_sequences` to cut off words from the end of sentences that exceed your predefined length, which parameter settings are correct?
- A. padding='post'
- **B. truncating='post'** *(Đáp án đúng)*
- C. truncating='pre'
- D. maxlen='post'

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Để ghi đè cấu hình mặc định (làm mất chữ ở đầu câu), bạn phải chủ động cung cấp thêm tham số truncating='post' để hướng dẫn Keras loại bỏ bớt các từ ở cuối câu khi câu đó vượt quá giới hạn chiều rộng.

#### Câu 17: If you use `tf.pad` in "SYMMETRIC" mode for an input dimension D, what size constraints are applied to the paddings?
- A. Paddings must be no greater than tensor.dim_size(D) - 1.
- **B. Paddings must be no greater than tensor.dim_size(D).** *(Đáp án đúng)*
- C. Paddings must be exactly equal to zero.
- D. Paddings must be twice the tensor size.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Theo Slide 3.4, khi sử dụng hàm tf.pad ở chế độ "SYMMETRIC", cả hai giá trị đệm trước và đệm sau cho chiều D đều phải nhỏ hơn hoặc bằng độ lớn gốc của chiều đó, tức là tensor.dim_size(D). (Lưu ý: Đối với chế độ "REFLECT", giới hạn khắt khe hơn là phải nhỏ hơn hoặc bằng tensor.dim_size(D) - 1).

### PART 5: WORD VECTORS & EMBEDDINGS

#### Câu 18: How does a neural network learn the high-dimensional spatial relationship (semantic vectors) of words during training?
- A. By lookup from hardcoded dictionary files stored on local systems.
- **B. By adjusting continuous coordinate values in multi-dimensional space, clustering words that appear in similar sentiment-labeled contexts close together.** *(Đáp án đúng)*
- C. By converting text letters into character-level ASCII codes.
- D. By sorting all words alphabetically in the memory layout.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Trong quá trình huấn luyện, mạng nơ-ron học các vector từ bằng cách gắn kết chúng với nhãn cảm xúc của tập dữ liệu. Ví dụ, trong các đánh giá phim tiêu cực, hai từ "dull" và "boring" xuất hiện cùng nhau liên tục nên chúng sẽ có xu hướng được gán các vector tương cận và dần kéo lại gần nhau tạo thành các cụm (cluster) có khoảng cách gần trong không gian đa chiều.

#### Câu 19: What is the structural output representation of a 2D array generated after passing a sentence sequence through a trained Keras Embedding layer?
- A. [sentence_length, vocabulary_size]
- **B. [sentence_length, embedding_dimension]** *(Đáp án đúng)*
- C. [embedding_dimension, vocabulary_size]
- D. [batch_size, embedding_dimension]

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Slide 3.5 làm rõ rằng đầu ra của một Embedding layer sau khi tiếp nhận chuỗi sẽ là một ma trận hai chiều (2D array) có cấu trúc dạng: chiều rộng là độ dài tối đa của câu (sentence_length) và chiều sâu là số chiều không gian vector nhúng (embedding_dimension).

#### Câu 20: In the IMDB sentiment classifier implementation, what output activation function and loss function are typically used for binary sentiment prediction (positive vs negative)?
- A. Softmax activation with Categorical Crossentropy.
- **B. Sigmoid activation with Binary Crossentropy.** *(Đáp án đúng)*
- C. ReLu activation with Mean Squared Error.
- D. Linear activation with Sparse Categorical Crossentropy.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Bài toán phân loại đánh giá phim IMDB là bài toán phân loại nhị phân (chỉ gồm 2 nhãn: 1 - tích cực và 0 - tiêu cực). Vì vậy, lớp Dense ngoài cùng của mô hình sẽ áp dụng hàm kích hoạt Sigmoid để cho ra xác suất trong khoảng , đi kèm hàm mất mát Binary Crossentropy phù hợp cho nhãn nhị phân.

### PART 6: SUBWORDS TEXT ENCODING

#### Câu 21: Why is `SubwordTextEncoder` used instead of a standard word-based Tokenizer in modern NLP models?
- A. It is designed to ignore unknown symbols completely.
- **B. It breaks words down into sub-word units (prefixes, suffixes, or roots), allowing the model to process Out-Of-Vocabulary (OOV) words by parsing their sub-components.** *(Đáp án đúng)*
- C. It runs faster because it only works on numbers.
- D. It translates words into different regional languages automatically.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Điểm mạnh của SubwordTextEncoder (như bộ subwords8k của IMDB) là nó không chia nhỏ văn bản thành các từ nguyên bản mà phân mảnh thành các "sub-tokens" (subwords). Nhờ vậy, khi gặp một từ lạ chưa từng thấy trong từ điển huấn luyện, mô hình vẫn có thể ước lượng ý nghĩa của từ đó bằng cách ghép nối ý nghĩa từ các mảnh subwords đã học được.

### PART 7: LONG SHORT-TERM MEMORY (LSTM)

#### Câu 22: What major flaw in standard Recurrent Neural Networks (RNN) are LSTM layers explicitly designed to solve?
- A. RNN layers take too much physical disc storage.
- **B. The vanishing gradient problem, enabling the network to learn long-term dependencies in long texts.** *(Đáp án đúng)*
- C. RNN layers cannot process integer arrays.
- D. The inability to use embedding layers.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Mạng tuần hoàn RNN truyền thống thường gặp khó khăn lớn khi lưu giữ thông tin xa do vấn đề triệt tiêu đạo hàm. Mạng LSTM (Long Short Term Memory) ra đời như một kiến trúc đặc biệt nhằm giải quyết triệt để vấn đề phụ thuộc xa (long-term dependency problem), giúp mô hình ghi nhớ thông tin ngữ cảnh xuyên suốt các câu dài.

#### Câu 23: Unlike standard RNNs which have a single tanh neural network layer in their repeating module, how many neural network layers interact inside an LSTM unit?
- A. Two
- B. Three
- **C. Four** *(Đáp án đúng)*
- D. Five

**Đáp án:** **C**

> [!NOTE]
> **Giải thích:** Theo tài liệu, các khối lặp của RNN thông thường chỉ chứa một lớp mạng đơn giản (như tanh). Khối lặp của LSTM phức tạp hơn rất nhiều khi tích hợp tới bốn (4) lớp mạng nơ-ron hoạt động tương tác với nhau theo một quy trình kiểm soát thông tin rất chặt chẽ.

### PART 8: SEQUENCE MODELS FOR TEXT GENERATION

#### Câu 24: In a next-word prediction model trained on a song or poetry corpus, how are the training features (X) and target labels (Y) extracted from a padded input sequence?
- A. Half of the sentence tokens are used as X and the rest as Y.
- **B. All tokens in the sequence except the last token are taken as the input X, and the last token is used as the target label Y.** *(Đáp án đúng)*
- C. Odd index tokens are X and even index tokens are Y.
- D. The entire sentence is X and the overall song title is Y.

**Đáp án:** **B**

> [!NOTE]
> **Giải thích:** Để chuẩn bị dữ liệu huấn luyện mô hình dự đoán từ tiếp theo, giải pháp phổ biến là trích xuất từ chuỗi n-gram: lấy toàn bộ các token phía trước làm đầu vào dữ liệu (X) và lấy từ/token cuối cùng ngay phía sau làm nhãn cần dự đoán (Y).

#### Câu 25: What target encoding strategy must be applied to labels (Ys) before training a next-word prediction model to match a multi-class classification setup?
- A. Float scaling normalization.
- B. Conversions into a binary matrix format.
- **C. One-hot encoding using tf.keras.utils.to_categorical with a size equal to total_words.** *(Đáp án đúng)*
- D. Logarithmic transformation.

**Đáp án:** **C**

> [!NOTE]
> **Giải thích:** Vì mô hình cần học cách dự đoán từ tiếp theo từ toàn bộ kho từ vựng (bài toán phân loại đa lớp), nhãn Y (dạng số nguyên ban đầu) cần phải được chuyển đổi thành dạng vector "one-hot encoding". Việc này được thực hiện thông qua hàm to_categorical với số lượng lớp bằng tổng số từ vựng trong kho ngữ liệu (total_words) , kết hợp với lớp Dense cuối sử dụng hàm kích hoạt softmax để tìm ra xác suất từ cao nhất.


---

## PHẦN 4: CHUỖI THỜI GIAN (TIME SERIES), ĐÁNH GIÁ HIỆU NĂNG & DNN/RNN
### Part 1: Train, Validation, and Test Sets (Data Partitioning)

#### Câu 1: When splitting a time series dataset with yearly seasonality into training, validation, and test sets, what is the best practice regarding the duration of each split?
- A. Each partition should cover exactly 50% of a season to avoid overfitting.
- **B. Each partition must contain a whole number of seasons (e.g., 1 year, 2 years, or 3 years).** *(Đáp án đúng)*
- C. The splits should be done randomly using k-fold cross-validation without considering temporal order.
- D. The test set should only consist of a half-season to test the model's adaptability.

**Đáp án:** **B**
**Tài liệu tham khảo (Grounding Citation):** [9]

> [!NOTE]
> **Giải thích:** Nếu chuỗi thời gian có tính chu kỳ (seasonality), chúng ta cần đảm bảo rằng mỗi tập dữ liệu (train/validation/test) chứa một số nguyên lần chu kỳ (ví dụ: đúng 1 năm, 2 năm hoặc 3 năm đối với chu kỳ năm) [9]. Việc chia theo kiểu "1 năm rưỡi" (1.5 years) sẽ khiến một số tháng hoặc mùa được đại diện nhiều hơn những tháng khác trong tập dữ liệu, gây ra sự mất cân bằng và sai lệch khi mô hình học và đánh giá các đặc trưng chu kỳ [9].

#### Câu 2: In time series forecasting, what is the primary role of the Validation Set and how does it prevent overfitting compared to the Test Set?
- A. The validation set is used to train the final weights of the model.
- B. The validation set is reserved for the final evaluation of the model's ability to generalize to completely unseen future data.
- **C. The validation set follows the training set and is used to fine-tune hyperparameters and architecture, providing an intermediate evaluation.** *(Đáp án đúng)*
- D. The validation set is shuffled randomly with the training set to create diverse mini-batches.

**Đáp án:** **C**
**Tài liệu tham khảo (Grounding Citation):** [11, 12]

> [!NOTE]
> **Giải thích:** Tập kiểm định (Validation Set) là phần dữ liệu tiếp nối ngay sau tập huấn luyện (Training Set) theo tính liên tục của thời gian [11]. Vai trò chính của nó là giúp người lập trình đánh giá trung gian để tinh chỉnh các siêu tham số (hyperparameters) và cấu trúc mạng (architecture) [11], từ đó giúp ngăn ngừa hiện tượng quá khớp (overfitting) trên tập huấn luyện trước khi mô hình được đánh giá cuối cùng trên tập kiểm thử (Test Set) hoàn toàn chưa được thấy [12, 14].

#### Câu 3: According to the standard process for training time series models presented in the slides, what should you do after evaluating your model's performance on the validation set but before testing on the test set?
- A. Retrain your model using only the validation data.
- B. Discard the validation set and train a completely new model architecture on the test set.
- **C. Retrain your model using both the training and validation data.** *(Đáp án đúng)*
- D. Immediately deploy the model to production without any further retraining.

**Đáp án:** **C**
**Tài liệu tham khảo (Grounding Citation):** [22]

> [!NOTE]
> **Giải thích:** Quy trình huấn luyện chuỗi thời gian chuẩn bao gồm các bước: (1) Huấn luyện trên tập Train để tìm ra kiến trúc phù hợp; (2) Đánh giá trên tập Validation; (3) Tái huấn luyện mô hình sử dụng cả dữ liệu tập Train và tập Validation để tận dụng tối đa lượng thông tin sẵn có trước khi dự báo tương lai; (4) Kiểm tra hiệu năng trên tập Test; (5) Tái huấn luyện lần cuối sử dụng toàn bộ dữ liệu bao gồm cả tập Test [22].

#### Câu 4: What is the core idea of Roll-forward partitioning (also known as walk-forward validation) in time series forecasting?
- A. Splitting the data randomly into $k$ folds and computing the average error.
- **B. Starting with a short training period, gradually increasing it (e.g., by one day or week at a time), training the model, and forecasting the next period.** *(Đáp án đúng)*
- C. Training the model on the most recent data first and evaluating it on the historical past observations.
- D. Using a fixed window of 80% train, 10% validation, and 10% test without ever retraining.

**Đáp án:** **B**
**Tài liệu tham khảo (Grounding Citation):** [24]

> [!NOTE]
> **Giải thích:** Phân hoạch cuốn chiếu (Roll-forward partitioning) là một chiến lược kiểm định chuỗi thời gian cực kỳ mạnh mẽ [24]. Ta bắt đầu với một giai đoạn huấn luyện ngắn, sau đó tăng dần thời gian huấn luyện lên (từng ngày hoặc từng tuần) [24]. Tại mỗi bước lặp, ta huấn luyện lại mô hình trên toàn bộ dữ liệu lịch sử đã tích lũy và sử dụng nó để dự báo cho ngày hoặc tuần tiếp theo ngay sau đó trong tập validation [24]. Điều này mô phỏng chính xác cách mô hình hoạt động trong thực tế khi liên tục nhận dữ liệu mới.

### Part 2: Metrics for Evaluating Performance

#### Câu 5: Why is the Mean Squared Error (MSE) often preferred over Mean Absolute Error (MAE) when training neural networks, and how does squaring the errors affect the penalty?
- A. MSE does not square the errors, making it simpler to compute.
- **B. Squaring the errors in MSE penalizes large errors more heavily than small ones, which helps guide the gradient descent optimizer to avoid large mistakes.** *(Đáp án đúng)*
- C. MSE measures errors as a percentage, which is easier to interpret for business executives.
- D. MSE is insensitive to outliers, so it ignores anomalously large errors.

**Đáp án:** **B**
**Tài liệu tham khảo (Grounding Citation):** [43, 47, 59]

> [!NOTE]
> **Giải thích:** MSE tính toán bằng cách bình phương các lỗi trước khi lấy trung bình [43, 47]. Việc bình phương này làm cho các lỗi lớn bị phóng đại và phạt nặng hơn nhiều so với các lỗi nhỏ (ví dụ: lỗi bằng 2 phạt 4, lỗi bằng 10 phạt 100) [43]. Điều này thúc đẩy mô hình tối ưu hóa để loại bỏ hoàn toàn các lỗi lớn, rất phù hợp khi các lỗi lớn mang lại rủi ro hoặc chi phí rất cao. Ngược lại, MAE chỉ lấy trị tuyệt đối nên phạt tuyến tính và đều đặn [59, 63].

#### Câu 6: If you need an evaluation metric that is in the same scale and units as your target variable (e.g., US Dollars or Temperature) while still heavily penalizing large outliers, which metric should you use?
- A. Mean Squared Error (MSE)
- B. Mean Absolute Percentage Error (MAPE)
- C. Forecast Bias
- **D. Root Mean Squared Error (RMSE)** *(Đáp án đúng)*

**Đáp án:** **D**
**Tài liệu tham khảo (Grounding Citation):** [51, 55]

> [!NOTE]
> **Giải thích:** RMSE là căn bậc hai của MSE [51, 55]. Nhờ việc khai căn này, RMSE đưa giá trị đo lường lỗi quay trở lại cùng thang đo và đơn vị vật lý với biến mục tiêu ban đầu [51, 55], giúp kết quả đánh giá dễ diễn giải hơn nhiều so với MSE (vốn có đơn vị bị bình phương) mà vẫn giữ nguyên đặc tính phạt nặng các lỗi lớn từ MSE [51].

#### Câu 7: Which of the following formulas correctly implements the Mean Absolute Percentage Error (MAPE) using NumPy, and what is its primary advantage?
- A. mape = np.square(errors).mean() - Provides a squared penalty scale.
- B. mape = np.abs(errors).mean() - Measures average absolute distance.
- **C. mape = np.abs(errors / x_valid).mean() - Expresses errors as a percentage of actual values, offering a relative measure of accuracy.** *(Đáp án đúng)*
- D. mape = errors.mean() - Measures the average forecast bias.

**Đáp án:** **C**
**Tài liệu tham khảo (Grounding Citation):** [67, 71]

> [!NOTE]
> **Giải thích:** Công thức tính MAPE trong NumPy là np.abs(errors / x_valid).mean() [67] (hoặc nhân với 100 để đổi thành phần trăm [71]). Ưu điểm lớn nhất của MAPE là thể hiện độ lớn của sai số tương đối so với giá trị thực tế dưới dạng phần trăm [67, 71]. Điều này cho phép chúng ta so sánh hiệu năng của mô hình trên các chuỗi thời gian có quy mô (scale) khác nhau (ví dụ: sai số 10 đơn vị trên sản lượng bán ra 100 là lớn, nhưng trên sản lượng 10,000 thì lại rất nhỏ).

#### Câu 8: What does a Forecast Bias value of -2.5 indicate about the performance of your forecasting model?
- A. The model's predictions are, on average, 2.5 units higher than the actual values (overforecasting).
- **B. The model's predictions are, on average, 2.5 units lower than the actual values (underforecasting).** *(Đáp án đúng)*
- C. The model has a 2.5% absolute accuracy rate.
- D. The model has exactly 2.5 unstable gradient points.

**Đáp án:** **B**
**Tài liệu tham khảo (Grounding Citation):** [75]

> [!NOTE]
> **Giải thích:** Forecast Bias (Độ lệch dự báo) đo lường xu hướng trung bình của mô hình là dự báo quá cao (too high) hay quá thấp (too low) [75]. Công thức tính Bias thường là trung bình của sai số (Dự báo - Thực tế) hoặc ngược lại tùy thuộc quy ước [75]. Theo ký hiệu chuẩn trong tài liệu, sai số được định nghĩa là $Y_i - \hat{Y}_i$ (Thực tế - Dự báo) hoặc $ ext{forecasts} - ext{actual}$. Nếu bias mang giá trị âm, điều đó có nghĩa là giá trị dự báo trung bình có xu hướng thấp hơn giá trị thực tế (underforecasting), hoặc ngược lại tùy thuộc vào hướng định nghĩa phép trừ lỗi [75]. Do đó, bias cho biết xu hướng lệch một chiều của mô hình chứ không phải độ lớn sai số tuyệt đối.

### Part 3: Preparing Features and Labels

#### Câu 9: In the TensorFlow dataset pipeline for time series windowing, what is the difference between setting `drop_remainder=False` and `drop_remainder=True` in the `dataset.window`() method?
- A. drop_remainder=True will randomly drop 50% of the training data to speed up epochs.
- **B. drop_remainder=True ensures that all generated windows have exactly the specified window size, discarding any incomplete windows at the end.** *(Đáp án đúng)*
- C. drop_remainder=False automatically pads shorter windows with zeros.
- D. drop_remainder=True reshapes the data to 3D automatically.

**Đáp án:** **B**
**Tài liệu tham khảo (Grounding Citation):** [110]

> [!NOTE]
> **Giải thích:** Khi cắt dữ liệu chuỗi thời gian bằng phương thức dataset.window(size, shift, drop_remainder), các điểm dữ liệu ở cuối chuỗi sẽ không đủ số lượng để tạo thành một cửa sổ có kích thước đầy đủ (size) [110]. Việc cấu hình drop_remainder=True sẽ bỏ qua những cửa sổ thiếu này [110], đảm bảo mọi batch dữ liệu đưa vào mô hình Neural Network đều có kích thước cố định và đồng nhất, tránh lỗi sai lệch kích thước đầu vào (shape mismatch) khi huấn luyện mô hình.

#### Câu 10: After calling `dataset.window`(size=5, shift=1, `drop_remainder=True`) on a TensorFlow dataset, the output contains window datasets of type _VariantDataset. What is the purpose of applying `dataset.flat_map`() with lambda window: `window.batch`(5)?
- A. It scales the window values by dividing them by 5.
- B. It shuffles the data inside each window.
- **C. It flattens the nested dataset structure, converting each window from a Dataset object into a single tensor of size 5.** *(Đáp án đúng)*
- D. It splits the windows into 80% train and 20% validation.

**Đáp án:** **C**
**Tài liệu tham khảo (Grounding Citation):** [115, 119]

> [!NOTE]
> **Giải thích:** Phương thức dataset.window() trả về một tập hợp các đối tượng Dataset con (nested dataset) [98, 102]. Chúng ta không thể đưa trực tiếp kiểu dữ liệu này vào huấn luyện mô hình. Bằng cách sử dụng flat_map() kết hợp với window.batch(size) [115], chúng ta làm phẳng cấu trúc lồng nhau này và gom các phần tử của mỗi cửa sổ thành một tensor (mảng) có kích thước cố định (ví dụ: kích thước là 5) [115, 119].

#### Câu 11: You want to prepare features (inputs) and labels (targets) from a flattened window dataset of size 5, where the first 4 steps are used to predict the 5th step. Which `dataset.map`() operation correctly implements this partition?
- A. dataset.map(lambda window: (window[:], window[-1]))
- **B. dataset.map(lambda window: (window[:-1], window[-1]))** *(Đáp án đúng)*
- C. dataset.map(lambda window: (window[1:], window[0]))
- D. dataset.map(lambda window: (window[:-1], window[:-1]))

**Đáp án:** **B**
**Tài liệu tham khảo (Grounding Citation):** [124, 128]

> [!NOTE]
> **Giải thích:** Để chia một cửa sổ kích thước 5 thành đặc trưng (X) và nhãn (y): ta lấy 4 phần tử đầu tiên làm đặc trưng đầu vào và phần tử cuối cùng làm nhãn dự báo [124]. Sử dụng cú pháp slicing trong Python: window[:-1] lấy từ đầu đến sát phần tử cuối (tương đương index 0, 1, 2, 3) đại diện cho $X$ [124, 128]. window[-1] lấy duy nhất phần tử cuối cùng (index 4) đại diện cho $y$ [124, 128]. Công thức chính xác là: dataset.map(lambda window: (window[:-1], window[-1])) [124, 128].

### Part 4: Building & Training DNNs

#### Câu 12: When building a single-layer neural network for linear regression on time series data using Keras, what are the parameters and structure of the layer?
- **A. tf.keras.layers.Dense(1, input_shape=[window_size])** *(Đáp án đúng)*
- B. tf.keras.layers.SimpleRNN(1, input_shape=[window_size])
- C. tf.keras.layers.Dense(window_size, input_shape=[1])
- D. tf.keras.layers.Dense(1, activation="relu")

**Đáp án:** **A**
**Tài liệu tham khảo (Grounding Citation):** [223]

> [!NOTE]
> **Giải thích:** Một mạng thần kinh đơn lớp thực hiện hồi quy tuyến tính bao gồm duy nhất một lớp Dense với 1 unit đầu ra (không có hàm kích hoạt phi tuyến) [158, 223]. Lớp này nhận vào một cửa sổ dữ liệu có kích thước window_size đại diện cho các bước thời gian trong quá khứ [223]. Mã nguồn Keras chuẩn để khởi tạo là: tf.keras.layers.Dense(1, input_shape=[window_size]) [223].

#### Câu 13: Why must we add a batch dimension (e.g., using `np.newaxis` or `np.expand_dims`) before passing a single window slice of shape (20,) into a trained Keras model for prediction?
- A. Keras models can only run predictions if the input data is scaled between 0 and 1.
- **B. Keras layers are designed to process inputs in batches; therefore, the model expects an input shape of 3D or 2D where the first dimension is the batch size (1, 20).** *(Đáp án đúng)*
- C. Adding a batch dimension increases the learning rate of the model.
- D. It prevents the weights of the model from being modified during prediction.

**Đáp án:** **B**
**Tài liệu tham khảo (Grounding Citation):** [234]

> [!NOTE]
> **Giải thích:** Trong TensorFlow/Keras, các lớp mô hình luôn được thiết kế để xử lý dữ liệu theo dạng lô (batch) để tối ưu hiệu năng tính toán [234]. Do đó, ngay cả khi dự báo cho một mẫu dữ liệu đơn lẻ có kích thước (20,), chúng ta vẫn phải thêm chiều batch để chuyển đổi hình dạng của dữ liệu thành (1, 20) [234]. Việc này được thực hiện thông qua series[0:20][np.newaxis] hoặc np.expand_dims [234].

#### Câu 14: During Deep Neural Network training in slide 4.8, why is it recommended to plot the loss starting from epoch 10 onwards (`plot_loss` = loss[10:]) rather than epoch 0?
- A. The first 10 epochs are computed incorrectly due to a bug in TensorFlow.
- **B. The loss values in the first few epochs are extremely high, compressing the scale of the rest of the plot and making it difficult to see the convergence pattern of later epochs.** *(Đáp án đúng)*
- C. TensorFlow automatically discards the weights trained during the first 10 epochs.
- D. The learning rate schedule only starts working after epoch 10.

**Đáp án:** **B**
**Tài liệu tham khảo (Grounding Citation):** [385]

> [!NOTE]
> **Giải thích:** Trong những epoch đầu tiên, mô hình bắt đầu học với các trọng số ngẫu nhiên nên giá trị loss thường cực kỳ cao [385]. Nếu vẽ toàn bộ đồ thị từ epoch 0, trục Y sẽ bị kéo giãn ra rất lớn, khiến cho phần đồ thị từ epoch 10 trở đi (khi loss đã giảm và hội tụ nhỏ) trông giống như một đường thẳng nằm ngang sát đáy [385]. Loại bỏ 10 epoch đầu giúp chúng ta thu nhỏ thang đo trục Y và quan sát rõ nét sự cải thiện chi tiết của mô hình ở các giai đoạn sau [385].

#### Câu 15: How can you use the Keras `LearningRateScheduler` callback to tune the learning rate of your optimizer during a test training run of 100 epochs?
- A. Set the learning rate to a constant value of 0.01.
- **B. Define a callback that increases the learning rate exponentially at each epoch (e.g., 1e-8 * 10**(epoch / 20)), then plot loss vs. learning rate to find the optimal point.** *(Đáp án đúng)*
- C. Decrement the learning rate by 0.1 at each step until it reaches 0.
- D. Change the optimizer from SGD to Adam every 10 epochs.

**Đáp án:** **B**
**Tài liệu tham khảo (Grounding Citation):** [349, 353, 367]

> [!NOTE]
> **Giải thích:** Để tìm ra tốc độ học (learning rate) tối ưu, kỹ thuật tuning hiệu quả là sử dụng LearningRateScheduler để tăng dần learning rate theo cấp số nhân qua từng epoch (bắt đầu từ rất nhỏ như $10^{-8}$ đến lớn như $10^{-3}$) [349, 353, 367]. Bằng cách vẽ biểu đồ Loss theo Learning Rate, ta có thể chọn ra giá trị learning rate nằm ở vùng dốc giảm nhanh nhất trước khi loss bắt đầu dao động mạnh hoặc bùng nổ (unstable) [367, 371].

### Part 5: Recurrent Neural Networks (RNN) Concepts & Implementation

#### Câu 16: What is a unique characteristic of Recurrent Neural Networks (RNNs) compared to standard Deep Neural Networks (DNNs) when processing sequences?
- A. RNNs do not use weights or biases during training.
- B. RNNs use different sets of weights at each time step to process elements.
- **C. RNNs contain connections that loop back, enabling them to maintain an internal hidden state (memory) of previous sequence steps, and they share the same weights across all time steps.** *(Đáp án đúng)*
- D. RNNs process all inputs in parallel, ignoring the temporal order of elements.

**Đáp án:** **C**
**Tài liệu tham khảo (Grounding Citation):** [416, 420, 422]

> [!NOTE]
> **Giải thích:** Sự khác biệt cốt lõi là mạng RNN có các liên kết phản hồi (recurrent connections) tự lặp lại [416, 420]. Điều này cho phép mạng duy trì một trạng thái ẩn (hidden state) đóng vai trò như bộ nhớ lưu trữ thông tin từ các bước trước đó trong chuỗi [416, 420]. Đồng thời, RNN sử dụng cơ chế chia sẻ trọng số (shared weights) - tức là cùng một bộ trọng số và bias được áp dụng nhất quán tại mọi bước thời gian, giúp mô hình học các mẫu độc lập với vị trí xuất hiện của chúng trong chuỗi [422].

#### Câu 17: When building a stacked RNN with two `SimpleRNN` layers in Keras, why must the first layer have `return_sequences=True`?
- A. It indicates that the layer should output a single scalar value.
- **B. The subsequent RNN layer requires a 3D sequence input (batch, time steps, units) rather than just a 2D tensor representing the final step output.** *(Đáp án đúng)*
- C. It activates the learning rate scheduler callback.
- D. It forces the model to use Huber loss instead of MSE.

**Đáp án:** **B**
**Tài liệu tham khảo (Grounding Citation):** [438, 503]

> [!NOTE]
> **Giải thích:** Lớp hồi quy tiếp theo trong chuỗi xếp chồng (stacked RNN) cần nhận đầu vào là một chuỗi các bước thời gian (sequence) đầy đủ để tiếp tục tính toán [438, 503]. Nếu lớp SimpleRNN đầu tiên không cấu hình return_sequences=True (mặc định là False), nó sẽ chỉ trả về kết quả của bước thời gian cuối cùng (mảng 2D), điều này sẽ gây lỗi cấu trúc khi truyền vào lớp SimpleRNN thứ hai [438]. Do đó, lớp RNN đầu tiên phải trả về chuỗi 3D đầy đủ [438, 503].

#### Câu 18: What is the purpose of using a Keras Lambda layer at the beginning of a Simple RNN model, as shown below? `tf.keras.layers.Lambda`(lambda x: `tf.expand_dims`(x, axis=-1), input_shape=[window_size])
- A. It calculates the square root of the inputs to match the scale of RMSE.
- B. It automatically normalizes the input dataset to have a mean of 0 and a variance of 1.
- **C. It expands the dimensions of the input tensor, adding a single-dimensional axis at the end to convert a 2D input (batch_size, window_size) into a 3D tensor (batch_size, window_size, 1) required by RNN layers.** *(Đáp án đúng)*
- D. It drops random weights to prevent overfitting.

**Đáp án:** **C**
**Tài liệu tham khảo (Grounding Citation):** [438, 503]

> [!NOTE]
> **Giải thích:** Lớp hồi quy trong Keras (như SimpleRNN) yêu cầu dữ liệu đầu vào phải có 3 chiều (3D) dạng [batch_size, time_steps, features] [438, 503]. Tuy nhiên, cửa sổ dữ liệu chuẩn bị từ tập dữ liệu thông thường chỉ có 2 chiều [batch_size, window_size] [500]. Lớp Lambda này sử dụng hàm tf.expand_dims(x, axis=-1) để thêm một chiều đơn lẻ ở cuối [503], biến đổi dữ liệu thành 3D một cách nhanh chóng và tự động ngay trong đồ thị luồng tính toán của mô hình (model graph) mà không cần tiền xử lý thủ công bên ngoài [438, 503].

### Part 6: Advanced Multi-Step Forecasting, Stationarity & Training Tweaks

#### Câu 19: If a time series exhibits non-stationarity (i.e., its statistical properties change over time), what preprocessing steps should be considered before splitting and training?
- A. No preprocessing is needed, as deep neural networks automatically handle non-stationarity.
- B. Randomly shuffle the entire dataset to eliminate temporal dependencies.
- **C. Apply strategies such as differencing or detrending to make the series stationary.** *(Đáp án đúng)*
- D. Discard all historical data and only keep the most recent contiguous year of observations.

**Đáp án:** **C**
**Tài liệu tham khảo (Grounding Citation):** [16]

> [!NOTE]
> **Giải thích:** Khi làm việc với chuỗi thời gian không dừng (non-stationary time series - tức là các thuộc tính thống kê như trung bình, phương sai thay đổi theo thời gian), ta cần cân nhắc áp dụng các chiến lược như lấy vi phân (differencing) hoặc khử xu hướng (detrending) trước khi thực hiện phân chia và đưa vào huấn luyện mô hình [16]. Điều này giúp ổn định các thuộc tính thống kê, làm cho mô hình dễ học và dự báo chính xác hơn [16].

#### Câu 20: Why does the standard time series training workflow suggest performing a final retraining step using ALL available data (train, validation, and test sets) before forecasting the future?
- A. To prevent the model from memorizing the test set during evaluation.
- **B. Because time series patterns are highly dynamic, and utilizing the most recent data (found in the test set) is crucial for accurate future forecasts.** *(Đáp án đúng)*
- C. To reduce the number of trainable parameters of the model.
- D. Because Keras models cannot run predictions unless the dataset contains at least three splits.

**Đáp án:** **B**
**Tài liệu tham khảo (Grounding Citation):** [22]

> [!NOTE]
> **Giải thích:** Trong chuỗi thời gian, dữ liệu gần nhất thường chứa thông tin có giá trị nhất về xu hướng hiện tại của hệ thống. Do đó, sau khi kiểm tra mô hình trên tập Test để chắc chắn nó có khả năng tổng quát hóa tốt [22], quy trình chuẩn khuyên ta nên tái huấn luyện lại một lần nữa sử dụng toàn bộ dữ liệu (gồm cả Train, Validation và Test) để mô hình hấp thụ được các mẫu dữ liệu mới nhất sát thời điểm cần dự báo [22].

#### Câu 21: What is the formula for calculating Percentage Forecast Accuracy (Accuracy) as presented in the metrics slides?
- **A. $Accuracy = 100 imes \left(1 - \frac{\sum_{i=1}^n |Y_i - \hat{Y}i|}{\sum{i=1}^n Y_i}\right)$** *(Đáp án đúng)*
- B. $Accuracy = 100 imes \frac{\sum_{i=1}^n (Y_i - \hat{Y}i)^2}{n}$
- C. $Accuracy = \frac{1}{n}\sum{i=1}^n |Y_i - \hat{Y}_i| \times 100$
- D. $Accuracy = \text{np.square}(\text{errors}).\text{mean}() \times 100$

**Đáp án:** **A**
**Tài liệu tham khảo (Grounding Citation):** [79]

> [!NOTE]
> **Giải thích:** Công thức đo lường độ chính xác tổng thể của dự báo dưới dạng phần trăm (Percentage Forecast Accuracy) được trình bày trong Slide 13 là: $Accuracy = 100 \times \left(1 - \frac{\sum |Y_i - \hat{Y}_i|}{\sum Y_i}\right)$ [79]. Công thức này tính toán tỷ lệ tổng sai số tuyệt đối so với tổng giá trị thực tế, sau đó lấy 1 trừ đi tỷ lệ này và nhân với 100 để có phần trăm độ chính xác [79].

#### Câu 22: In slide 4.6, what does the command dataset = `tf.data.Dataset.range`(10) generate, and what will be printed when looping through it and running print(val.numpy())?
- A. A dataset containing 10 random float numbers between 0 and 1.
- B. An array of integers starting from 1 to 10.
- **C. Ten elements representing numbers from 0 to 9 in sequential order.** *(Đáp án đúng)*
- D. A single tensor with value 10.

**Đáp án:** **C**
**Tài liệu tham khảo (Grounding Citation):** [92]

> [!NOTE]
> **Giải thích:** Hàm tf.data.Dataset.range(10) trong TensorFlow tạo ra một Dataset chứa 10 phần tử kiểu số nguyên chạy liên tiếp từ 0 đến 9 [92]. Khi lặp qua dataset này và gọi phương thức.numpy(), chương trình sẽ in ra các số từ 0 đến 9 [92]. Đây là bước cơ bản đầu tiên trong bài học minh họa tiền xử lý dữ liệu trước khi cắt cửa sổ.

#### Câu 23: When preparing windowed datasets using `dataset.window`(size=5, shift=1), how does the shift parameter affect the generation of windowed datasets?
- A. It specifies how many epochs the model will skip during training.
- **B. It determines the step size (stride) by which the window moves forward to generate the next window (e.g., shift=1 means windows overlap and start at indices 0, then 1, then 2, etc.).** *(Đáp án đúng)*
- C. It drops 1 elements from the beginning of the series.
- D. It adds 1 dummy element to pad each window.

**Đáp án:** **B**
**Tài liệu tham khảo (Grounding Citation):** [106, 110]

> [!NOTE]
> **Giải thích:** Tham số shift quy định bước nhảy khi cửa sổ di chuyển dọc theo chuỗi thời gian [106]. Với shift=1, cửa sổ tiếp theo sẽ dịch chuyển đi đúng 1 điểm dữ liệu [110], tạo ra độ chồng lặp tối đa giữa các cửa sổ kế tiếp (ví dụ: cửa sổ 1 chứa [0, 1, 2, 3, 4], cửa sổ 2 chứa [1, 2, 3, 4, 5],...) [110]. Nếu tăng shift, các cửa sổ sẽ ít chồng lặp hơn và số lượng cửa sổ thu được sẽ giảm đi.

#### Câu 24: In the single-layer neural network model defined as:
```python
l0 = tf.keras.layers.Dense(1, input_shape=[20])
model = tf.keras.models.Sequential([l0])
```
How many trainable parameters does this model have, and why?
- A. 1 parameter (the single bias).
- B. 20 parameters (one weight for each input step).
- **C. 21 parameters (20 weights plus 1 bias).** *(Đáp án đúng)*
- D. 40 parameters (20 weights plus 20 biases).

**Đáp án:** **C**
**Tài liệu tham khảo (Grounding Citation):** [223]

> [!NOTE]
> **Giải thích:** Lớp Dense(1, input_shape=[20]) thực hiện hồi quy tuyến tính trên cửa sổ đầu vào có kích thước 20 [223]. Do đó, nó cần học 20 trọng số (weights) tương ứng với 20 bước dữ liệu lịch sử đầu vào, cộng thêm 1 trọng số định thiên (bias) cho unit đầu ra [223]. Tổng số tham số có thể huấn luyện (trainable parameters) của mô hình này là $20 \times 1 + 1 = 21$ tham số [223].

#### Câu 25: In the code implementation in slide 4.7 and 4.8, how is the synthetic time series dataset generated using NumPy?
- A. By randomly generating numbers from a Uniform distribution between 0 and 120.
- **B. series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude) + noise(time, noise_level, seed=42)** *(Đáp án đúng)*
- C. By training an auto-encoder model on actual weather data and generating predictions.
- D. series = trend(time, slope) * seasonality(time, period=365) / noise(time)

**Đáp án:** **B**
**Tài liệu tham khảo (Grounding Citation):** [189, 294]

> [!NOTE]
> **Giải thích:** Dữ liệu tổng hợp (synthetic data) được tạo ra một cách có hệ thống bằng cách cộng tuyến tính các thành phần: Đường cơ sở (baseline) + Xu hướng (trend) + Tính chu kỳ (seasonality) + Nhiễu ngẫu nhiên (noise) [189, 294]. Cụ thể, nhiễu được tạo từ phân phối chuẩn với hạt giống ngẫu nhiên cố định (seed=42) để đảm bảo tính lặp lại của thí nghiệm [189, 294].

#### Câu 26: When tuning the learning rate using the semilog plot of Loss vs. Learning Rate, how should the optimal learning rate be selected?
- A. Select the learning rate where the loss achieves its absolute minimum point, even if it is near the upward spike.
- **B. Select a learning rate close to the minimum loss point but slightly to the left, where the loss is decreasing rapidly and training is highly stable.** *(Đáp án đúng)*
- C. Choose the largest possible learning rate on the right side of the plot to speed up training.
- D. Select the learning rate on the far left ($10^{-8}$), because smaller learning rates are always better.

**Đáp án:** **B**
**Tài liệu tham khảo (Grounding Citation):** [367, 371]

> [!NOTE]
> **Giải thích:** Trên đồ thị biểu diễn Loss theo tốc độ học (ở thang đo semilog), điểm đáy (minimum loss) thường nằm ngay sát vùng huấn luyện bắt đầu mất ổn định (unstable) và tăng vọt [367, 371]. Do đó, lựa chọn tốt nhất là một tốc độ học nằm trong vùng dốc giảm nhanh nhất (ở bên trái điểm cực tiểu một chút, ví dụ là 4e-6 hoặc 1e-6) [371]. Điều này đảm bảo mô hình học nhanh mà vẫn giữ được sự ổn định, tránh hiện tượng bùng nổ đạo hàm (exploding gradients) [367].

#### Câu 27: Slide 4.9 suggests compiling the Simple RNN model with `Huber loss` during training. What is the primary advantage of `Huber loss` in sequence training?
- A. It is extremely sensitive to outliers, ensuring that outliers are heavily penalized.
- **B. It acts like MAE for large errors (less sensitive to outliers) and like MSE for small errors (smooth gradients near zero), combining the best of both metrics.** *(Đáp án đúng)*
- C. It is computationally 10 times faster to calculate than MSE.
- D. It eliminates the need for backpropagation through time.

**Đáp án:** **B**
**Tài liệu tham khảo (Grounding Citation):** [438]

> [!NOTE]
> **Giải thích:** Hàm mất mát Huber (Huber loss) là sự kết hợp thông minh giữa MSE và MAE [438]. Với các sai số nhỏ, Huber loss hoạt động giống như MSE (bình phương sai số), giúp đạo hàm mượt mà gần điểm tối ưu để hội tụ tốt hơn [438]. Với các sai số lớn (do nhiễu hoặc outliers), nó hoạt động giống như MAE (sai số tuyệt đối tuyến tính), giúp mô hình bớt nhạy cảm với các dị biệt và tránh hiện tượng đạo hàm bị dao động quá mạnh [438].

#### Câu 28: What does Backpropagation Through Time (BPTT) represent in the context of Recurrent Neural Networks (RNNs)?
- A. A method that predicts the future first, and then backward predicts the past.
- **B. A variation of backpropagation where error gradients are propagated backward through the entire sequential timeline (unrolled steps) of the network to update the shared weights.** *(Đáp án đúng)*
- C. An optimization algorithm that operates only on the forward pass of the neural network.
- D. A data cleaning technique used to detrend a non-stationary time series dataset.

**Đáp án:** **B**
**Tài liệu tham khảo (Grounding Citation):** [534]

> [!NOTE]
> **Giải thích:** BPTT là thuật toán lan truyền ngược qua thời gian dành riêng cho RNN [534]. Vì RNN xử lý chuỗi tuần tự theo từng bước thời gian và chia sẻ chung trọng số, thuật toán BPTT phải mở rộng cấu trúc mạng theo thời gian (unroll) và lan truyền đạo hàm sai số ngược dòng thời gian từ bước cuối cùng về bước đầu tiên để tính toán tổng đạo hàm và cập nhật chính xác cho bộ trọng số dùng chung [534].
