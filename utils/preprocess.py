"""
preprocess.py - Module hỗ trợ đọc ảnh
======================================
Chức năng:
- Kiểm tra file upload hợp lệ
- Đọc ảnh từ file upload thành numpy array
- Chuyển đổi màu BGR <-> RGB
- Resize ảnh nếu quá lớn (tránh tràn bộ nhớ)

Lưu ý: KHÔNG tiền xử lý dữ liệu (đã xử lý trước khi train).
"""

import cv2
import numpy as np
from PIL import Image


# Các định dạng ảnh được hỗ trợ
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png')

# Kích thước tối đa cho ảnh đầu vào (để tránh quá tải bộ nhớ)
MAX_IMAGE_SIZE = 1920


def validate_image(uploaded_file):
    """
    Kiểm tra file upload có phải ảnh hợp lệ không.

    Args:
        uploaded_file: File được upload từ Streamlit

    Returns:
        tuple: (is_valid: bool, message: str)
    """
    try:
        # Kiểm tra file có tồn tại không
        if uploaded_file is None:
            return False, "Chưa chọn file ảnh nào."

        # Kiểm tra định dạng file
        file_name = uploaded_file.name.lower()
        if not any(file_name.endswith(fmt) for fmt in SUPPORTED_FORMATS):
            return False, f"Định dạng ảnh không được hỗ trợ. Chỉ chấp nhận: {', '.join(SUPPORTED_FORMATS)}"

        # Kiểm tra kích thước file (tối đa 10MB)
        file_size = uploaded_file.size
        if file_size > 10 * 1024 * 1024:
            return False, "File ảnh quá lớn. Vui lòng chọn ảnh nhỏ hơn 10MB."

        # Thử đọc ảnh để kiểm tra tính hợp lệ
        image = Image.open(uploaded_file)
        image.verify()

        # Reset lại con trỏ file sau khi verify
        uploaded_file.seek(0)

        return True, "Ảnh hợp lệ."

    except Exception as e:
        return False, f"File ảnh bị lỗi hoặc không đọc được: {str(e)}"


def load_image_from_upload(uploaded_file):
    """
    Đọc ảnh từ file upload và chuyển thành numpy array (BGR cho OpenCV).

    Args:
        uploaded_file: File được upload từ Streamlit

    Returns:
        numpy.ndarray: Ảnh ở dạng BGR (OpenCV format)
    """
    try:
        # Đọc bytes từ file upload
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

        # Decode thành ảnh BGR (OpenCV mặc định)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Không thể decode ảnh từ file upload.")

        # Reset con trỏ file
        uploaded_file.seek(0)

        return image

    except Exception as e:
        raise RuntimeError(f"Lỗi khi đọc ảnh: {str(e)}")


def resize_image(image, max_size=MAX_IMAGE_SIZE):
    """
    Resize ảnh về kích thước tối đa nếu quá lớn, giữ nguyên tỷ lệ.
    Chỉ để tránh tràn bộ nhớ, KHÔNG phải tiền xử lý.

    Args:
        image: numpy array (BGR)
        max_size: Kích thước cạnh dài nhất cho phép

    Returns:
        numpy.ndarray: Ảnh đã resize (hoặc ảnh gốc nếu đủ nhỏ)
    """
    h, w = image.shape[:2]

    # Nếu ảnh nhỏ hơn max_size thì giữ nguyên
    if max(h, w) <= max_size:
        return image

    # Tính tỷ lệ scale
    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def bgr_to_rgb(image):
    """
    Chuyển ảnh từ BGR (OpenCV) sang RGB (Pillow/Streamlit).

    Args:
        image: numpy array BGR

    Returns:
        numpy.ndarray: Ảnh RGB
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
