"""
enhancer.py - Module làm nét ảnh biển số (Real-ESRGAN / OpenCV)
"""
import cv2
import numpy as np

# Thử import Real-ESRGAN (cần cài đặt: pip install realesrgan)
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    ESRGAN_AVAILABLE = True
except ImportError:
    ESRGAN_AVAILABLE = False


def is_esrgan_available():
    return ESRGAN_AVAILABLE


def load_esrgan_model():
    if not ESRGAN_AVAILABLE:
        raise ImportError("Chưa cài đặt Real-ESRGAN. Hãy chạy: pip install realesrgan")
    
    try:
        import torch
        # Khởi tạo kiến trúc model Real-ESRGAN x4 plus
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
        use_half = torch.cuda.is_available()
        print(f"[INFO] Khởi tạo Real-ESRGAN với FP16 (half) = {use_half}")
        
        upsampler = RealESRGANer(
            scale=4,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            dni_weight=None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=use_half, # Đặt True nếu dùng GPU hỗ trợ FP16 (chỉ hoạt động trên CUDA)
            gpu_id=None # Để mặc định tự chọn GPU/CPU
        )
        return upsampler
    except Exception as e:
        raise RuntimeError(f"Lỗi khi load Real-ESRGAN model: {str(e)}")


def enhance_with_esrgan(model, image):
    """
    Làm nét ảnh bằng Real-ESRGAN.
    """
    try:
        # 1. Chạy ESRGAN để lấy độ sắc nét
        esrgan_output, _ = model.enhance(image, outscale=4)
        return esrgan_output
    except Exception as e:
        print(f"[LỖI] Real-ESRGAN thất bại: {str(e)}")
        # Trả về ảnh resize đơn giản nếu lỗi
        h, w = image.shape[:2]
        return cv2.resize(image, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)





def align_plate(image):
    """
    Tự động căn chỉnh góc nghiêng của biển số (Deskewing) bằng HoughLinesP.
    Cách này ổn định hơn MinAreaRect vì nó tìm các đường thẳng ngang thực tế
    của viền biển số hoặc viền chữ, không bị nhiễu bởi bố cục 2 dòng (bị thiếu góc).
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Tìm các đoạn thẳng
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=10)
        
        if lines is None:
            return image
            
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Tính góc của đoạn thẳng
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # Chuẩn hóa góc về khoảng [-90, 90]
            if angle > 90: angle -= 180
            elif angle < -90: angle += 180
            
            # Chỉ lấy các đường ngang (nghiêng từ -30 đến +30 độ)
            # Bỏ qua các đường sát 0 độ (có thể là cạnh vuông góc của chính bức ảnh bị crop)
            if abs(angle) < 30 and abs(angle) > 0.5:
                angles.append(angle)
                
        if not angles:
            return image
            
        # Lấy trung vị (median) để loại bỏ các đoạn thẳng nhiễu
        median_angle = np.median(angles)
        
        if abs(median_angle) < 1.0:
            return image
            
        print(f"[INFO] Tự động xoay biển số {median_angle:.2f} độ (HoughLines)")
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        
        # Xoay ảnh, nhân bản viền
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
    except Exception as e:
        print(f"[CẢNH BÁO] Lỗi khi align_plate: {str(e)}")
        return image


def enhance_with_opencv(image, scale=4):
    """
    Làm nét ảnh bằng OpenCV fallback pipeline:
    Resize -> Bilateral Filter -> Unsharp Mask -> CLAHE
    """
    if image is None or image.size == 0:
        return image

    # 1. Resize (upscale)
    h, w = image.shape[:2]
    resized = cv2.resize(image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # 2. Bilateral Filter (khử nhiễu giữ cạnh)
    filtered = cv2.bilateralFilter(resized, d=9, sigmaColor=75, sigmaSpace=75)

    # 3. Unsharp Mask (làm nét cạnh)
    gaussian = cv2.GaussianBlur(filtered, (5, 5), 0)
    sharpened = cv2.addWeighted(filtered, 1.5, gaussian, -0.5, 0)

    # 4. CLAHE (tăng tương phản cục bộ trên kênh L của không gian màu LAB)
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced


def enhance_plate(image, use_esrgan=False, esrgan_model=None, scale=4, align=True):
    """
    Hàm chính để làm nét biển số.
    """
    # Bước 1: Xoay thẳng biển số (Deskewing)
    if align:
        image = align_plate(image)

    # Bước 2: Làm nét (Bằng ESRGAN hoặc OpenCV fallback)
    if use_esrgan and esrgan_model is not None:
        try:
            enhanced = enhance_with_esrgan(esrgan_model, image)
            return enhanced, "Real-ESRGAN"
        except Exception as e:
            print(f"[CẢNH BÁO] ESRGAN thất bại: {str(e)}")
            enhanced_cv = enhance_with_opencv(image, scale=scale)
            return enhanced_cv, "OpenCV (ESRGAN fallback)"

    # Nếu không dùng ESRGAN hoặc không có model, dùng OpenCV làm nét
    enhanced_cv = enhance_with_opencv(image, scale=scale)
    return enhanced_cv, "OpenCV"
