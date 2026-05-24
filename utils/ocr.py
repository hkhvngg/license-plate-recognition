"""
ocr.py - Module đọc ký tự biển số bằng EasyOCR
================================================
Chức năng:
- Load reader EasyOCR (hỗ trợ tiếng Anh - phù hợp cho biển số VN)
- Đọc ký tự từ ảnh biển số bằng chiến lược multi-attempt
- Sửa lỗi OCR dựa trên vị trí dấu gạch ngang (an toàn, không đoán)
"""

from PIL import Image

# Fix cho lỗi "module 'PIL.Image' has no attribute 'ANTIALIAS'"
if not hasattr(Image, 'ANTIALIAS'):
    setattr(Image, 'ANTIALIAS', getattr(Image, 'LANCZOS', getattr(Image, 'Resampling', None) and getattr(Image.Resampling, 'LANCZOS', 1)))

import easyocr
import cv2
import numpy as np
import re


def load_ocr_reader(languages=None):
    """Load EasyOCR reader."""
    if languages is None:
        languages = ['en']

    try:
        reader = easyocr.Reader(languages, gpu=True, verbose=False)
        print(f"[INFO] Đã load EasyOCR reader với ngôn ngữ: {languages}")
        return reader
    except Exception as e:
        try:
            reader = easyocr.Reader(languages, gpu=False, verbose=False)
            print(f"[INFO] Đã load EasyOCR reader (CPU mode)")
            return reader
        except Exception as e2:
            raise RuntimeError(
                f"Không thể khởi tạo EasyOCR.\n"
                f"Lỗi GPU: {str(e)}\n"
                f"Lỗi CPU: {str(e2)}\n"
                f"Hãy thử: pip install easyocr --upgrade"
            )


PLATE_ALLOWLIST = '0123456789ABCDEFGHKLMNPRSTUVWXYZ-.'


def _prepare_variants(image):
    """
    Tạo 2 phiên bản ảnh để thử OCR:
    1. Ảnh gốc (BGR)
    2. Grayscale
    """
    variants = [("original", image)]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variants.append(("grayscale", gray))

    return variants


def _run_ocr_single(reader, image):
    try:
        results = reader.readtext(
            image, detail=1, paragraph=False,
            allowlist=PLATE_ALLOWLIST,
        )
        ocr_results = []
        for (bbox, text, confidence) in results:
            cleaned = text.strip()
            if cleaned:
                ocr_results.append({
                    'text': cleaned,
                    'confidence': float(confidence),
                    'bbox': bbox
                })
        return ocr_results
    except Exception:
        return []


def _calc_avg_confidence(ocr_results):
    if not ocr_results:
        return 0.0
    return sum(r['confidence'] for r in ocr_results) / len(ocr_results)


def read_plate_text(reader, image):
    """
    Đọc ký tự biển số bằng chiến lược Multi-Attempt OCR.
    Thử trên 2 phiên bản ảnh, chọn kết quả confidence cao nhất.
    """
    try:
        variants = _prepare_variants(image)
        best_results = []
        best_conf = 0.0

        for name, variant_img in variants:
            results = _run_ocr_single(reader, variant_img)
            if not results:
                continue
            avg_conf = _calc_avg_confidence(results)
            if avg_conf > best_conf:
                best_conf = avg_conf
                best_results = results
                
            # Early stopping để tối ưu tốc độ nếu kết quả đã đủ tốt
            if best_conf >= 0.85:
                break

        return best_results

    except Exception as e:
        raise RuntimeError(f"Lỗi khi chạy OCR: {str(e)}")


def format_plate_text(ocr_results, aspect_ratio=None):
    if not ocr_results:
        return "Không đọc được", 0.0

    # Sắp xếp các bounding box: từ trên xuống dưới, từ trái qua phải
    for r in ocr_results:
        bbox = r['bbox']
        y_center = sum(p[1] for p in bbox) / 4.0
        x_center = sum(p[0] for p in bbox) / 4.0
        r['y_center'] = y_center
        r['x_center'] = x_center
    
    # Thuật toán gom nhóm dòng (Line Grouping) siêu chuẩn:
    # Bước 1: Sắp xếp tất cả theo chiều dọc (y_center)
    ocr_results.sort(key=lambda r: r['y_center'])
    
    lines = []
    current_line = []
    for r in ocr_results:
        if not current_line:
            current_line.append(r)
        else:
            # Nếu chênh lệch chiều cao với phần tử đầu dòng < 20 pixel -> Cùng 1 dòng
            if abs(r['y_center'] - current_line[0]['y_center']) < 20:
                current_line.append(r)
            else:
                lines.append(current_line)
                current_line = [r]
    if current_line:
        lines.append(current_line)
        
    # Bước 2: Trong mỗi dòng, sắp xếp từ trái qua phải (x_center)
    sorted_results = []
    for line in lines:
        line.sort(key=lambda r: r['x_center'])
        sorted_results.extend(line)
        
    all_texts = [r['text'] for r in sorted_results]
    all_confidences = [r['confidence'] for r in sorted_results]

    raw_text = ' '.join(all_texts)
    cleaned = clean_plate_text(raw_text)
    
    # Khôi phục dấu gạch ngang bị EasyOCR nhận diện nhầm thành dấu chấm hoặc khoảng trắng
    if '-' not in cleaned:
        dot_idx = cleaned.find('.')
        if dot_idx > 0:
            if aspect_ratio and aspect_ratio > 2.5:
                # 1 dòng (ô tô) -> gạch ngang thường ở index 3 hoặc 4 (hoặc 2 nếu mất 1 ký tự đầu)
                if 2 <= dot_idx <= 4:
                    cleaned = cleaned[:dot_idx] + '-' + cleaned[dot_idx+1:]
            else:
                # 2 dòng (xe máy) -> gạch ngang thường ở index 2
                if 2 <= dot_idx <= 3:
                    cleaned = cleaned[:dot_idx] + '-' + cleaned[dot_idx+1:]
        elif ' ' in cleaned:
            space_idx = cleaned.find(' ')
            if 2 <= space_idx <= 4:
                cleaned = cleaned[:space_idx] + '-' + cleaned[space_idx+1:]
                
    cleaned = cleaned.replace(' ', '')
    fixed = fix_by_dash_position(cleaned, aspect_ratio=aspect_ratio)
    avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0

    return fixed, avg_conf


def clean_plate_text(text):
    """Làm sạch chuỗi OCR thô: giữ lại A-Z, 0-9, dấu gạch ngang và dấu chấm."""
    text = text.upper().strip()
    # Giữ lại khoảng trắng ban đầu để phân biệt dòng trên/dưới của biển 2 dòng
    text = re.sub(r'[^A-Z0-9\-\.\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def fix_by_dash_position(text, aspect_ratio=None):
    """
    Sửa lỗi OCR dựa trên cấu trúc biển số Việt Nam và aspect_ratio (tỷ lệ khung hình).
    """
    DIGIT_TO_LETTER = {
        '0': 'D', '1': 'T', '2': 'Z', '4': 'A',
        '5': 'S', '6': 'G', '7': 'T', '8': 'B',
    }
    LETTER_TO_DIGIT = {
        'D': '0', 'O': '0', 'Q': '0', 'I': '1',
        'Z': '2', 'A': '4', 'L': '4', 'S': '5',
        'G': '6', 'T': '7', 'B': '8',
    }

    dash_idx = text.find('-')
    
    if dash_idx < 0:
        return text

    chars = list(text)

    # ===== SỬA MÃ TỈNH (2 ký tự đầu luôn là SỐ) =====
    for i in range(min(2, dash_idx)):
        if chars[i].isalpha():
            chars[i] = LETTER_TO_DIGIT.get(chars[i], chars[i])

    is_one_line = aspect_ratio is not None and aspect_ratio > 2.5

    # ===== XÁC ĐỊNH LOẠI BIỂN DỰA TRÊN VỊ TRÍ DẤU GẠCH NGANG =====
    if dash_idx == 2:
        if is_one_line:
            # Nếu biển 1 dòng mà dash ở vị trí 2 -> chắc chắn bị miss ký tự mã tỉnh hoặc sê-ri
            # Không được biến số thành chữ (không đổi 5 thành S), chỉ ép các chữ sau gạch thành số.
            for i in range(3, len(chars)):
                if chars[i].isalpha():
                    chars[i] = LETTER_TO_DIGIT.get(chars[i], chars[i])
        else:
            # BIỂN XE MÁY (2 dòng)
            if len(chars) > 3 and chars[3].isdigit():
                chars[3] = DIGIT_TO_LETTER.get(chars[3], chars[3])
            if len(chars) > 4:
                if chars[3] not in ['A', 'M']:
                    if chars[4].isalpha():
                        chars[4] = LETTER_TO_DIGIT.get(chars[4], chars[4])
            for i in range(5, len(chars)):
                if chars[i].isalpha():
                    chars[i] = LETTER_TO_DIGIT.get(chars[i], chars[i])

    elif dash_idx == 3:
        # BIỂN Ô TÔ 1 CHỮ
        if chars[2].isdigit():
            chars[2] = DIGIT_TO_LETTER.get(chars[2], chars[2])
        for i in range(4, len(chars)):
            if chars[i].isalpha():
                chars[i] = LETTER_TO_DIGIT.get(chars[i], chars[i])

    elif dash_idx == 4:
        # BIỂN Ô TÔ 2 CHỮ
        if len(chars) > 2 and chars[2].isdigit():
            chars[2] = DIGIT_TO_LETTER.get(chars[2], chars[2])
        if len(chars) > 3 and chars[3].isdigit():
            chars[3] = DIGIT_TO_LETTER.get(chars[3], chars[3])
        for i in range(5, len(chars)):
            if chars[i].isalpha():
                chars[i] = LETTER_TO_DIGIT.get(chars[i], chars[i])

    return ''.join(chars)
