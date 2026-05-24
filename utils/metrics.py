"""
metrics.py - Module đánh giá kết quả (PSNR, SSIM, Accuracy)
==========================================================
Chức năng:
- Tính PSNR (Peak Signal-to-Noise Ratio): Đo chất lượng ảnh phục hồi so với ảnh gốc.
- Tính SSIM (Structural Similarity Index): Đo độ tương đồng cấu trúc (tuỳ chọn).
- Tính Accuracy: So sánh kết quả OCR với Ground Truth.
"""

import cv2
import numpy as np
import Levenshtein

def calculate_psnr(img1, img2):
    """
    Tính PSNR giữa 2 ảnh.
    Ảnh phải cùng kích thước. Nếu khác, sẽ resize img2 về bằng img1.
    """
    if img1 is None or img2 is None:
        return 0.0

    # Đảm bảo cùng kích thước
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Tính MSE (Mean Squared Error)
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_cer(pred_text, gt_text):
    """
    Tính Character Error Rate (CER).
    Càng thấp càng tốt.
    """
    if not gt_text:
        return 1.0 if pred_text else 0.0
    
    distance = Levenshtein.distance(pred_text, gt_text)
    cer = distance / len(gt_text)
    return cer

def is_match_exact(pred_text, gt_text):
    """Kiểm tra khớp chính xác hoàn toàn."""
    return pred_text.replace("-", "").replace(".", "") == gt_text.replace("-", "").replace(".", "")
