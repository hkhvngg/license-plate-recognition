import os
import sys
import io
import cv2
import pandas as pd
from tqdm import tqdm
import argparse

# Đảm bảo stdout dùng UTF-8 để in tiếng Việt không bị lỗi trên Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from utils.detector import detect_plates, crop_plate
from utils.enhancer import enhance_with_opencv
from utils.ocr import load_ocr_reader, read_plate_text, format_plate_text
from utils.metrics import calculate_psnr, calculate_cer, is_match_exact

def load_labels(label_path):
    labels = {}
    if not os.path.exists(label_path):
        return labels
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                filename, text = parts[0], parts[1]
                labels[filename] = text.upper()
    return labels

def evaluate_dataset(dataset_dir, output_csv="benchmark_results.csv"):
    lr_dir = os.path.join(dataset_dir, "LR")
    hr_dir = os.path.join(dataset_dir, "HR")
    label_path = os.path.join(dataset_dir, "labels.txt")
    
    if not os.path.exists(lr_dir):
        print(f"Lỗi: Không tìm thấy thư mục {lr_dir}")
        return

    labels = load_labels(label_path)
    if not labels:
        print("Cảnh báo: Không tìm thấy labels.txt. Hệ thống sẽ không thể tính Accuracy/CER.")
        
    has_hr = os.path.exists(hr_dir)
    if not has_hr:
        print("Cảnh báo: Không tìm thấy thư mục HR/. Hệ thống sẽ không thể tính PSNR.")

    # Load mô hình YOLO
    from utils.detector import load_yolo_model
    model_path = 'weights/best.pt'
    try:
        detector = load_yolo_model(model_path)
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return
    
    # Load OCR
    ocr_reader = load_ocr_reader()
    
    # Load ESRGAN (nếu có)
    try:
        from utils.enhancer import load_esrgan_model, enhance_with_esrgan
        esrgan_model = load_esrgan_model()
    except Exception:
        esrgan_model = None
        print("Cảnh báo: Không có Real-ESRGAN, sẽ dùng OpenCV Enhance để thay thế.")

    results = []

    image_files = [f for f in os.listdir(lr_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Bắt đầu đánh giá {len(image_files)} ảnh...")

    for filename in tqdm(image_files):
        lr_path = os.path.join(lr_dir, filename)
        lr_img = cv2.imread(lr_path)
        if lr_img is None:
            continue
            
        gt_text = labels.get(filename, "")
        
        # Đọc HR image nếu có
        hr_img = None
        if has_hr:
            hr_path = os.path.join(hr_dir, filename)
            if os.path.exists(hr_path):
                hr_img = cv2.imread(hr_path)
                
        # 1. Phát hiện biển số (Dùng chung cho cả 2 pipeline)
        boxes = detect_plates(detector, lr_img, confidence=0.15)
        
        yolo_detected = True
        if boxes:
            # Lấy biển số đầu tiên
            box = boxes[0]['bbox']
            cropped_lr = crop_plate(lr_img, box)
            
            cropped_hr = None
            if hr_img is not None:
                # Scale box lên nếu kích thước HR/LR khác nhau
                hr_h, hr_w = hr_img.shape[:2]
                lr_h, lr_w = lr_img.shape[:2]
                scale_x, scale_y = hr_w / lr_w, hr_h / lr_h
                
                x1, y1, x2, y2 = box
                hr_box = [int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)]
                cropped_hr = crop_plate(hr_img, hr_box)
        else:
            # Nếu không detect được, giả sử ảnh đã được crop sẵn
            yolo_detected = False
            cropped_lr = lr_img
            cropped_hr = hr_img

        # ---------------------------------------------
        # Pipeline 1: Truyền thống (Không phục hồi)
        # ---------------------------------------------
        trad_ocr = read_plate_text(ocr_reader, cropped_lr)
        trad_text, _ = format_plate_text(trad_ocr)
        
        trad_psnr = calculate_psnr(cropped_hr, cropped_lr) if cropped_hr is not None else None
        trad_cer = calculate_cer(trad_text, gt_text) if gt_text else None
        trad_match = is_match_exact(trad_text, gt_text) if gt_text else None

        # ---------------------------------------------
        # Pipeline 2: Phục hồi (ESRGAN / OpenCV)
        # ---------------------------------------------
        if esrgan_model:
            enhanced_lr = enhance_with_esrgan(esrgan_model, cropped_lr)
        else:
            enhanced_lr = enhance_with_opencv(cropped_lr)
            
        enh_ocr = read_plate_text(ocr_reader, enhanced_lr)
        enh_text, _ = format_plate_text(enh_ocr)
        
        enh_psnr = calculate_psnr(cropped_hr, enhanced_lr) if cropped_hr is not None else None
        enh_cer = calculate_cer(enh_text, gt_text) if gt_text else None
        enh_match = is_match_exact(enh_text, gt_text) if gt_text else None
        
        # Lưu kết quả
        results.append({
            "Filename": filename,
            "GT_Text": gt_text,
            "YOLO_Detected": yolo_detected,
            # Truyền thống
            "Trad_Text": trad_text,
            "Trad_ExactMatch": trad_match,
            "Trad_CER": trad_cer,
            "Trad_PSNR": trad_psnr,
            # Phục hồi
            "Enh_Text": enh_text,
            "Enh_ExactMatch": enh_match,
            "Enh_CER": enh_cer,
            "Enh_PSNR": enh_psnr
        })

    # Tổng hợp thành DataFrame
    df = pd.DataFrame(results)
    
    # Tính toán trung bình
    print("\n--- BÁO CÁO KẾT QUẢ (BENCHMARK) ---")
    detected_count = df["YOLO_Detected"].sum()
    print(f"Tổng số ảnh: {len(df)}")
    print(f"YOLO phát hiện được biển số: {detected_count}/{len(df)} ảnh")
    
    if "Trad_ExactMatch" in df and not df["Trad_ExactMatch"].isna().all():
        trad_acc = df["Trad_ExactMatch"].mean() * 100
        enh_acc = df["Enh_ExactMatch"].mean() * 100
        trad_cer_avg = df["Trad_CER"].mean()
        enh_cer_avg = df["Enh_CER"].mean()
        print(f"\n[Accuracy] Khớp hoàn toàn (Exact Match):")
        print(f" - Truyền thống (Không phục hồi): {trad_acc:.2f}%")
        print(f" - Phục hồi (ESRGAN/OpenCV)     : {enh_acc:.2f}%")
        print(f"\n[CER] Tỷ lệ lỗi ký tự (Character Error Rate - Thấp hơn là tốt hơn):")
        print(f" - Truyền thống (Không phục hồi): {trad_cer_avg:.4f}")
        print(f" - Phục hồi (ESRGAN/OpenCV)     : {enh_cer_avg:.4f}")
        
    if "Trad_PSNR" in df and not df["Trad_PSNR"].isna().all():
        # Bỏ qua các giá trị vô cực nếu có
        trad_psnr_avg = df.loc[df["Trad_PSNR"] != float('inf'), "Trad_PSNR"].mean()
        enh_psnr_avg = df.loc[df["Enh_PSNR"] != float('inf'), "Enh_PSNR"].mean()
        print(f"\n[PSNR] Độ nét so với Ground Truth (Cao hơn là tốt hơn):")
        print(f" - Truyền thống (Không phục hồi): {trad_psnr_avg:.2f} dB")
        print(f" - Phục hồi (ESRGAN/OpenCV)     : {enh_psnr_avg:.2f} dB")

    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\nĐã lưu chi tiết kết quả vào file: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá thực nghiệm mô hình nhận diện biển số.")
    parser.add_argument("--dataset", type=str, default="dataset", help="Đường dẫn tới thư mục dataset")
    parser.add_argument("--output", type=str, default="benchmark_results.csv", help="File CSV đầu ra")
    args = parser.parse_args()
    
    evaluate_dataset(args.dataset, args.output)
