# 🚦 Traffic Sign Recognition System

[![GitHub](https://img.shields.io/badge/GitHub-PHUPHU2310-blue?logo=github)](https://github.com/PHUPHU2310/Machine-Learning)

Hệ thống nhận diện biển báo giao thông sử dụng Machine Learning với thuật toán KNN (K-Nearest Neighbors) và đặc trưng HOG (Histogram of Oriented Gradients).

## 📋 Mô tả dự án

Dự án này xây dựng một hệ thống web để phân loại biển báo giao thông dựa trên:
- **ClassId**: Loại biển báo (ví dụ: biển cấm, biển báo nguy hiểm, biển chỉ dẫn)
- **ShapeId**: Hình dạng biển báo (tròn, tam giác, vuông)
- **ColorId**: Màu sắc chủ đạo (đỏ, xanh, vàng)

## 🛠️ Công nghệ sử dụng

- **Python 3.11+**
- **Flask**: Web framework
- **OpenCV**: Xử lý ảnh
- **scikit-learn**: Thuật toán KNN
- **scikit-image**: Trích xuất đặc trưng HOG
- **NumPy & Pandas**: Xử lý dữ liệu

## 📁 Cấu trúc thư mục

```
archive/
├── Train.csv              # Dữ liệu huấn luyện
├── Test.csv               # Dữ liệu kiểm tra
├── Meta.csv               # Metadata về các lớp
├── train_model.py         # Script huấn luyện mô hình
├── app.py                 # Ứng dụng Flask
├── requirements.txt       # Thư viện cần thiết
├── knn_class_model.joblib # Mô hình dự đoán ClassId
├── knn_shape_model.joblib # Mô hình dự đoán ShapeId
├── knn_color_model.joblib # Mô hình dự đoán ColorId
├── static/                # Thư mục lưu ảnh upload
└── templates/
    └── index.html         # Giao diện web
```

## 🚀 Cài đặt

### 1. Clone hoặc tải project

```bash
cd c:\Users\Jang\Downloads\archive
```

### 2. Cài đặt Python

Khuyến nghị sử dụng **Python 3.11** hoặc **3.12** để tránh lỗi tương thích:
- Tải tại: https://www.python.org/downloads/

### 3. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv venv
venv\Scripts\activate
```

### 4. Cài đặt thư viện

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Lưu ý:** Nếu gặp lỗi khi cài đặt, thử:
```bash
pip install --only-binary :all: -r requirements.txt
```

## 🎓 Huấn luyện mô hình

Chạy script để huấn luyện 3 mô hình KNN:

```bash
python train_model.py
```

Quá trình này sẽ:
1. Đọc dữ liệu từ `Train.csv`, `Test.csv`, `Meta.csv`
2. Trích xuất đặc trưng HOG từ ảnh
3. Huấn luyện 3 mô hình KNN riêng biệt
4. Lưu mô hình vào file `.joblib`

**Thời gian:** Tùy thuộc vào số lượng ảnh (khoảng 5-30 phút)

## 🌐 Chạy ứng dụng Web

Sau khi đã có các file mô hình, khởi động Flask:

```bash
python app.py
```

Mở trình duyệt và truy cập:
```
http://127.0.0.1:5000
```

## 📝 Cách sử dụng

1. Truy cập trang web
2. Click "Choose File" để chọn ảnh biển báo
3. Click "Upload" để tải ảnh lên
4. Hệ thống sẽ hiển thị kết quả dự đoán:
   - Loại biển báo (ClassId)
   - Hình dạng (ShapeId)
   - Màu sắc (ColorId)

## 🧪 Cách hoạt động

### 1. Trích xuất đặc trưng HOG
```python
# Chuyển ảnh sang grayscale
# Resize về kích thước chuẩn (64x64)
# Trích xuất histogram của gradient hướng
```

### 2. Phân loại bằng KNN
```python
# So sánh với k=5 ảnh gần nhất trong tập huấn luyện
# Vote để xác định nhãn cuối cùng
```

## 📊 Kết quả

- **Độ chính xác**: Phụ thuộc vào chất lượng dataset
- **Tốc độ dự đoán**: ~0.1-0.5 giây/ảnh
- **Số lượng mô hình**: 3 (class, shape, color)

## 🔧 Tùy chỉnh

### Thay đổi số lượng láng giềng (k)

Trong `train_model.py`:
```python
knn_class = KNeighborsClassifier(n_neighbors=5)  # Thay đổi giá trị này
```

### Điều chỉnh kích thước ảnh

Trong cả `train_model.py` và `app.py`:
```python
def extract_hog(img_path, image_size=(64, 64)):  # Thay đổi kích thước
```

## ⚠️ Xử lý lỗi thường gặp

### Lỗi: "No module named 'cv2'"
```bash
pip install opencv-python-headless
```

### Lỗi: "Could not find a version that satisfies the requirement numpy"
- Sử dụng Python 3.11 hoặc 3.12
- Hoặc cài Visual Studio Build Tools

### Lỗi: "No such file or directory: Train.csv"
- Đảm bảo các file CSV và ảnh nằm đúng thư mục

## 📄 License

MIT License - Tự do sử dụng cho mục đích học tập và nghiên cứu.

## 👥 Đóng góp

Mọi đóng góp đều được chào đón! Hãy tạo Pull Request hoặc mở Issue.

## 📧 Liên hệ

Thiều Khánh Phú - 0916345323

---# Virtual Environment
venv/
env/
ENV/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Uploaded images
static/uploads/
static/*.jpg
static/*.png
static/*.ppm

# Model files (nếu quá lớn, có thể bỏ comment)
# *.joblib

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# CSV data (nếu quá lớn)
# Train/
# Test/

**Lưu ý:** Đây là project học tập, không nên sử dụng trong sản phẩm thực tế mà chưa được kiểm tra kỹ lưỡng.
