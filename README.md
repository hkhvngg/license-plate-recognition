# Hệ thống nhận diện biển số xe Việt Nam

> **YOLOv8 Detection** • **Real-ESRGAN Enhancement** • **EasyOCR Recognition**

Hệ thống nhận diện biển số xe Việt Nam sử dụng pipeline AI hoàn chỉnh: phát hiện biển số bằng YOLOv8, tăng độ nét bằng Real-ESRGAN (hoặc OpenCV), và đọc ký tự bằng EasyOCR. Giao diện web được xây dựng trên Streamlit.

---

## 📋 Pipeline xử lý

```
Upload ảnh/video → YOLOv8 Detect → Crop biển số → ESRGAN/OpenCV Enhance → EasyOCR → Hiển thị kết quả
```

| Bước | Công nghệ | Mô tả |
|------|-----------|-------|
| 1 | Streamlit | Upload ảnh xe (JPG, JPEG, PNG) hoặc video (MP4, AVI, MOV, MKV) |
| 2 | YOLOv8 | Phát hiện vùng biển số trong ảnh hoặc từng frame video |
| 3 | OpenCV | Crop vùng biển số từ ảnh gốc |
| 4 | Real-ESRGAN / OpenCV | Tăng độ nét ảnh biển số |
| 5 | EasyOCR | Đọc ký tự biển số |
| 6 | Streamlit | Hiển thị kết quả trên giao diện web |

---

## 🗂️ Cấu trúc project

```
license-plate-app/
│
├── app.py                  # Giao diện Streamlit chính
├── requirements.txt        # Danh sách thư viện cần thiết
├── README.md               # File này
├── weights/
│   └── best.pt             # Model YOLOv8 đã train (BẠN CẦN TỰ ĐẶT VÀO)
├── inputs/                 # Thư mục chứa ảnh đầu vào (tùy chọn)
├── outputs/                # Thư mục chứa kết quả (tùy chọn)
├── utils/
│   ├── __init__.py
│   ├── detector.py         # Module phát hiện biển số (YOLOv8)
│   ├── enhancer.py         # Module làm nét ảnh (ESRGAN/OpenCV)
│   ├── ocr.py              # Module đọc ký tự (EasyOCR)
│   └── preprocess.py       # Module tiền xử lý ảnh
└── models/                 # Thư mục chứa model bổ sung (tùy chọn)
```

---

## 🚀 Hướng dẫn cài đặt

### Yêu cầu hệ thống

- Python 3.8 trở lên
- GPU (khuyến nghị, không bắt buộc) - hỗ trợ CUDA cho YOLOv8 và EasyOCR
- Windows / Linux / macOS

### Bước 1: Clone hoặc tải project

```bash
git clone <repository-url>
cd license-plate-app
```

### Bước 2: Tạo virtual environment (khuyến nghị)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### Bước 3: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Bước 4: Đặt file model YOLOv8

⚠️ **QUAN TRỌNG**: Bạn cần đặt file `best.pt` (model YOLOv8 đã train) vào thư mục `weights/`:

```
license-plate-app/
└── weights/
    └── best.pt    ← Đặt file model vào đây
```

File `best.pt` là model YOLOv8 đã được train để phát hiện biển số xe Việt Nam. Nếu chưa có, bạn cần train model trước.

### Bước 5 (Tùy chọn): Cài đặt Real-ESRGAN

Real-ESRGAN cho chất lượng làm nét tốt hơn nhưng có thể khó cài trên Windows:

```bash
pip install realesrgan basicsr
```

> 💡 Nếu không cài được Real-ESRGAN, hệ thống sẽ tự động fallback sang OpenCV (vẫn hoạt động tốt).

---

## ▶️ Chạy ứng dụng

### Windows / PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py
```

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Mở trình duyệt tại: `http://localhost:8501`

> Không cần GPU. Nếu máy không có GPU, YOLO/EasyOCR sẽ chạy bằng CPU nhưng chậm hơn.

### For AI Agents

Nếu bạn dùng AI agent để dựng môi trường, hãy làm đúng thứ tự:

1. Kiểm tra Python 3.10 hoặc 3.11.
2. Tạo `.venv` nếu chưa có.
3. Cài dependencies bằng `pip install -r requirements.txt`.
4. Kiểm tra file `weights/best.pt` tồn tại.
5. Chạy app bằng `streamlit run app.py` hoặc `.\.venv\Scripts\streamlit.exe run app.py` trên Windows.
6. Không commit hoặc upload `.venv/`, `outputs/`, video output, cache Python.

---

## 🖥️ Hướng dẫn sử dụng

1. **Chọn chế độ**: Chọn `Ảnh` hoặc `Video` trên giao diện.
2. **Upload dữ liệu**:
   - Ảnh: JPG, JPEG, PNG.
   - Video: MP4, AVI, MOV, MKV.
3. **Nếu dùng video**: Có thể xem lại lịch sử xử lý trong mục `Lịch sử xử lý`, gồm video kết quả, CSV và nút xóa từng run.
4. **Bấm nút nhận diện**:
   - Ảnh: Hệ thống detect, crop, làm nét và OCR từng biển số.
   - Video: Hệ thống dùng YOLO + ByteTrack để gán Track ID, OCR theo tần suất mặc định trong code, dùng OpenCV enhancement nhẹ cho tốc độ, chỉ khóa/lưu kết quả khi cả YOLO và OCR đủ tin cậy, gộp các lượt trùng hoặc biển đọc lệch 1 ký tự, rồi xuất video kết quả.
5. **Xem kết quả**:
   - Ảnh hoặc video đã annotate bounding box.
   - Text biển số nhận diện được.
   - Confidence của YOLO và OCR.
   - Với video có thêm bảng kết quả theo Track ID và đường dẫn file MP4 kết quả.

### Cấu trúc output video
Mỗi lần bấm `Detect video`, app tạo một thư mục riêng:
```text
outputs/
└── runs/
    └── YYYYmmdd_HHMMSS_ten_video/
        ├── source_original.<ext>      # Video gốc upload
        ├── source_web.mp4             # Video gốc H.264 để xem trên web
        ├── result_web.mp4             # Video kết quả H.264 để xem trên web
        └── results.csv                # Kết quả chắc chắn đã khóa (nếu có)
```

---

## 🔧 Xử lý lỗi thường gặp

### ❌ Lỗi: Không tìm thấy `weights/best.pt`

```
FileNotFoundError: Không tìm thấy file model tại 'weights/best.pt'
```

**Cách sửa:**
- Kiểm tra file `best.pt` đã có trong thư mục `weights/` chưa
- Đường dẫn đúng: `license-plate-app/weights/best.pt`
- Hoặc thay đổi đường dẫn trong sidebar của ứng dụng

---

### ❌ Lỗi cài EasyOCR

```
ImportError: No module named 'easyocr'
```

**Cách sửa:**
```bash
pip install easyocr --upgrade

# Nếu vẫn lỗi, thử:
pip install easyocr torch torchvision --upgrade
```

> Lần đầu chạy EasyOCR sẽ tự download model nhận dạng (~100MB). Cần có kết nối internet.

---

### ❌ Lỗi cài Real-ESRGAN

```
ImportError: No module named 'realesrgan'
```

**Cách sửa:**
```bash
pip install realesrgan basicsr

# Nếu lỗi trên Windows:
pip install realesrgan basicsr --no-build-isolation

# Nếu vẫn lỗi, bỏ qua ESRGAN:
# → Hệ thống sẽ tự động dùng OpenCV để làm nét (kết quả vẫn tốt)
```

> 💡 Real-ESRGAN là **tùy chọn**. Nếu không cài được, hệ thống dùng OpenCV fallback với pipeline: Resize + Bilateral Filter + Unsharp Mask + CLAHE.

---

### ❌ Lỗi: Không detect được biển số

**Nguyên nhân có thể:**
- Ảnh không chứa biển số xe
- Biển số bị che khuất, quá nhỏ, hoặc quá mờ
- Ngưỡng confidence quá cao
- Model YOLOv8 chưa được train tốt cho loại biển số này

**Cách sửa:**
1. Giảm ngưỡng confidence trong sidebar xuống **0.1 - 0.15**
2. Dùng ảnh rõ nét hơn, biển số không bị che
3. Kiểm tra lại model `best.pt` đã train đúng loại biển số chưa
4. Thử với ảnh khác để xác nhận model hoạt động

---

### ❌ Lỗi CUDA / GPU

```
RuntimeError: CUDA out of memory
```

**Cách sửa:**
```bash
# Chạy với CPU (chậm hơn nhưng ổn định):
set CUDA_VISIBLE_DEVICES=-1
streamlit run app.py
```

---

## 📊 Đánh giá Thực nghiệm (Evaluation Module)

Để hỗ trợ viết báo cáo/đồ án, dự án có kèm theo script đánh giá hiệu năng `evaluate.py` giúp so sánh trực tiếp chất lượng giữa:
- **Mô hình truyền thống (YOLO + OCR)**
- **Mô hình đề xuất (YOLO + ESRGAN + OCR)**

**Tính năng đánh giá:**
- **Accuracy & CER**: Đánh giá độ chính xác đọc chữ.
- **PSNR**: Đánh giá độ phục hồi nét ảnh (nếu có ảnh Ground Truth).

### Cách sử dụng
**1. Chuẩn bị tập dữ liệu (Dataset)**
Bạn cần cấu trúc dữ liệu như sau trong thư mục `dataset/`:
```
dataset/
├── LR/                 # Ảnh mờ/chất lượng thấp (Bắt buộc)
├── HR/                 # Ảnh nét/Ground truth (Tùy chọn - Dùng tính PSNR)
└── labels.txt          # Chứa đáp án. Định dạng: tên_ảnh.jpg [Tab] BIEN_SO
```
*Ví dụ nội dung `labels.txt`:*
```text
test1.jpg	51F88888
test2.jpg	29A12345
```

**2. Chạy đánh giá**
```bash
python evaluate.py --dataset dataset --output benchmark_results.csv
```

Hệ thống sẽ chạy qua toàn bộ ảnh, so sánh 2 pipeline và xuất ra báo cáo tóm tắt trên terminal, đồng thời lưu chi tiết từng ảnh vào file `benchmark_results.csv` (có thể dùng Excel để mở và vẽ biểu đồ).

---

## 📦 Công nghệ sử dụng

| Thư viện | Phiên bản | Mục đích |
|----------|-----------|----------|
| Python | ≥ 3.8 | Ngôn ngữ chính |
| Streamlit | ≥ 1.28 | Giao diện web |
| Ultralytics | ≥ 8.0 | YOLOv8 detection |
| OpenCV | ≥ 4.8 | Xử lý ảnh |
| EasyOCR | ≥ 1.7 | Đọc ký tự |
| Pillow | ≥ 10.0 | Xử lý ảnh |
| NumPy | ≥ 1.24 | Xử lý mảng số |
| Real-ESRGAN | (tùy chọn) | Làm nét ảnh AI |
| PyTorch | ≥ 2.0 | Deep learning framework |

---

## 📝 License

MIT License - Free to use and modify.
