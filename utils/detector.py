"""
detector.py - Module phát hiện biển số xe
==========================================
Chức năng:
- Load model YOLOv8 từ file weights/best.pt
- Phát hiện biển số xe trong ảnh
- Crop vùng biển số từ ảnh gốc
- Vẽ bounding box lên ảnh
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import functools


def load_yolo_model(model_path="weights/best.pt"):
    """
    Load model YOLOv8 đã train.
    Hàm này sẽ được cache bởi Streamlit để không load lại nhiều lần.

    Args:
        model_path: Đường dẫn tới file weights (.pt)

    Returns:
        YOLO: Model YOLOv8 đã load

    Raises:
        FileNotFoundError: Nếu không tìm thấy file model
    """
    # Kiểm tra file model có tồn tại không
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Không tìm thấy file model tại '{model_path}'.\n"
            f"Vui lòng đặt file best.pt vào thư mục weights/\n"
            f"Đường dẫn đầy đủ cần có: {os.path.abspath(model_path)}"
        )

    try:
        # Fix cho PyTorch >= 2.6: tạm thời đặt weights_only=False
        # An toàn vì đây là model do chính bạn train
        _original_load = torch.load
        torch.load = functools.partial(_original_load, weights_only=False)

        try:
            # Load model YOLOv8
            model = YOLO(model_path)
            print(f"[INFO] Đã load model YOLOv8 từ: {model_path}")
        finally:
            # Khôi phục torch.load gốc
            torch.load = _original_load

        return model

    except Exception as e:
        raise RuntimeError(f"Lỗi khi load model YOLOv8: {str(e)}")


def detect_plates(model, image, confidence=0.25):
    """
    Phát hiện biển số xe trong ảnh bằng YOLOv8.

    Args:
        model: Model YOLOv8 đã load
        image: numpy array (BGR) - ảnh đầu vào
        confidence: Ngưỡng confidence tối thiểu (0.0 - 1.0)

    Returns:
        list: Danh sách các biển số phát hiện được, mỗi biển số là dict gồm:
            - 'bbox': (x1, y1, x2, y2) tọa độ bounding box
            - 'confidence': Độ tin cậy của detection
            - 'class_name': Tên class (nếu có)
    """
    try:
        # Chạy inference với YOLOv8
        results = model(image, conf=confidence, verbose=False)

        detections = []

        # Duyệt qua từng kết quả
        for result in results:
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                # Lấy tọa độ bounding box (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Lấy confidence score
                conf = float(box.conf[0].cpu().numpy())

                # Lấy class name (nếu model có nhiều class)
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names.get(class_id, "license_plate")

                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'class_name': class_name
                })

        return detections

    except Exception as e:
        raise RuntimeError(f"Lỗi khi chạy detection: {str(e)}")


def crop_plate(image, bbox, padding=5):
    """
    Crop vùng biển số từ ảnh gốc.

    Args:
        image: numpy array (BGR) - ảnh gốc
        bbox: tuple (x1, y1, x2, y2) - tọa độ bounding box
        padding: Số pixel padding thêm xung quanh biển số

    Returns:
        numpy.ndarray: Ảnh biển số đã crop
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    # Thêm padding nhưng không vượt quá biên ảnh
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    # Crop vùng biển số
    cropped = image[y1:y2, x1:x2]

    return cropped


def draw_bounding_boxes(image, detections):
    """
    Vẽ bounding box và thông tin lên ảnh.

    Args:
        image: numpy array (BGR) - ảnh gốc
        detections: list - danh sách các detection từ hàm detect_plates()

    Returns:
        numpy.ndarray: Ảnh đã vẽ bounding box
    """
    # Copy ảnh để không thay đổi ảnh gốc
    annotated = image.copy()

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']

        # Màu xanh lá cho bounding box
        color = (0, 255, 0)
        thickness = 2

        # Vẽ bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Tạo label với confidence
        label = f"Bien so #{i + 1}: {conf:.1%}"

        # Tính kích thước text
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        # Vẽ nền cho text (để dễ đọc)
        cv2.rectangle(
            annotated,
            (x1, y1 - text_h - baseline - 5),
            (x1 + text_w, y1),
            color,
            -1  # Filled rectangle
        )

        # Vẽ text
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),  # Màu đen cho text
            2
        )

    return annotated
