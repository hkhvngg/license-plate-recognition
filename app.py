"""
app.py - Giao diện Streamlit cho hệ thống nhận diện biển số xe Việt Nam
========================================================================
Pipeline:
1. Upload ảnh xe
2. YOLOv8 phát hiện biển số
3. Crop vùng biển số
4. Làm nét bằng Real-ESRGAN hoặc OpenCV
5. EasyOCR đọc ký tự
6. Hiển thị kết quả

Chạy: streamlit run app.py
"""

import streamlit as st
import cv2
import os
import time
import html
import subprocess
import shutil
from pathlib import Path
import csv

# Import các module xử lý
from utils.preprocess import (
    validate_image,
    load_image_from_upload,
    resize_image,
    bgr_to_rgb,
)
from utils.detector import (
    load_yolo_model,
    detect_plates,
    crop_plate,
    draw_bounding_boxes,
)
from utils.enhancer import (
    is_esrgan_available,
    load_esrgan_model,
    enhance_plate,
)
from utils.ocr import (
    load_ocr_reader,
    read_plate_text,
    format_plate_text,
)


# ============================================
# CẤU HÌNH TRANG
# ============================================
st.set_page_config(
    page_title="Nhận diện biển số xe Việt Nam",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================
# CUSTOM CSS - Giao diện đẹp, hiện đại
# ============================================
st.markdown("""
<style>
    /* === Font và nền chung === */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* === Header chính === */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.7);
        margin: 0.5rem 0 0 0;
        font-size: 0.95rem;
        font-weight: 300;
    }

    /* === Card kết quả === */
    .result-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
    }

    /* === Badge biển số === */
    .plate-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-size: 1.6rem;
        font-weight: 700;
        padding: 0.7rem 1.8rem;
        border-radius: 10px;
        letter-spacing: 3px;
        font-family: 'Courier New', monospace;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        text-align: center;
    }

    /* === Confidence badge === */
    .confidence-high {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .confidence-medium {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        color: #333;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .confidence-low {
        background: linear-gradient(135deg, #eb3349, #f45c43);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }

    /* === Info box === */
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
        border: 1px solid rgba(102, 126, 234, 0.25);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        color: #c0c4e0;
        font-size: 0.9rem;
    }

    /* === Pipeline step === */
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.5rem 0;
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.85rem;
    }
    .step-icon {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
        flex-shrink: 0;
    }
    .step-active {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
    }
    .step-done {
        background: #00b09b;
        color: white;
    }

    /* === Sidebar styling === */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e);
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #c0c4e0;
    }

    /* === Divider === */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
        margin: 1.5rem 0;
    }

    /* === Metric cards === */
    .metric-row {
        display: flex;
        gap: 12px;
        margin: 0.8rem 0;
    }
    .metric-item {
        flex: 1;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.5);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 2px;
    }

    /* === Upload area === */
    .stFileUploader > div > div {
        border: 2px dashed rgba(102, 126, 234, 0.4) !important;
        border-radius: 12px !important;
        background: rgba(102, 126, 234, 0.05) !important;
    }

    /* === Button styling === */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.5px;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
    }

    /* Ẩn footer mặc định */
    footer { visibility: hidden; }

    /* === Responsive Images (Zoom fix) === */
    [data-testid="stImage"] img, .stImage img {
        width: 100% !important;
        max-width: 100% !important;
        height: auto !important;
        object-fit: contain !important;
    }

    /* === Balanced layout refinement === */
    [data-testid="stAppViewContainer"] {
        background: #0d1117;
    }
    .main .block-container {
        max-width: 1720px;
        padding-top: 1.25rem;
        padding-bottom: 2.5rem;
    }
    .main-header {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 12px;
        box-shadow: none;
        padding: 1.35rem 1.75rem;
        margin-bottom: 1rem;
        text-align: left;
    }
    .main-header h1 {
        font-size: 1.65rem;
        line-height: 1.2;
        letter-spacing: 0;
        color: #f8fafc;
        background: none;
        -webkit-text-fill-color: currentColor;
    }
    .main-header p {
        margin-top: 0.35rem;
        color: #94a3b8;
        font-size: 0.9rem;
    }
    h3 {
        margin-top: 0.55rem !important;
        margin-bottom: 0.75rem !important;
        font-size: 1.25rem !important;
        letter-spacing: 0 !important;
    }
    .custom-divider {
        margin: 1.05rem 0;
        background: #253044;
    }
    .metric-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 10px;
        align-items: stretch;
    }
    .metric-item {
        min-height: 74px;
        border-radius: 8px;
        padding: 0.85rem 0.75rem;
        background: #151a23;
        border: 1px solid #283244;
        box-shadow: none;
    }
    .metric-value {
        font-size: 1.25rem;
        line-height: 1.15;
        color: #7aa2ff;
    }
    .metric-label {
        font-size: 0.72rem;
        line-height: 1.25;
        color: #94a3b8;
        letter-spacing: 0.35px;
    }
    .result-card {
        background: #111827;
        border: 1px solid #263244;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.75rem 0 1rem;
        box-shadow: none;
    }
    .result-card:hover {
        transform: none;
        box-shadow: none;
    }
    .plate-badge {
        border-radius: 8px;
        padding: 0.55rem 1.25rem;
        font-size: 1.35rem;
        letter-spacing: 2px;
        box-shadow: none;
    }
    .stFileUploader > div > div {
        border: 1px dashed #40516b !important;
        border-radius: 10px !important;
        background: #111827 !important;
    }
    .stButton > button {
        border-radius: 8px !important;
        min-height: 42px !important;
        box-shadow: none !important;
        letter-spacing: 0 !important;
    }
    .stButton > button:hover {
        transform: none !important;
    }
    [data-testid="stImage"] {
        background: #0f141d;
        border: 1px solid #263244;
        border-radius: 10px;
        padding: 8px;
    }
    [data-testid="stImage"] img, .stImage img {
        border-radius: 6px;
        max-height: 560px;
        object-fit: contain !important;
    }
    [data-testid="stVideo"] {
        background: #0f141d;
        border: 1px solid #263244;
        border-radius: 10px;
        padding: 8px;
    }
    .result-table-wrap {
        overflow-x: auto;
        border: 1px solid #263244;
        border-radius: 8px;
        margin: 0.75rem 0;
    }
    .result-table {
        width: 100%;
        border-collapse: collapse;
        table-layout: auto;
        background: #0f141d;
    }
    .result-table th,
    .result-table td {
        border-bottom: 1px solid #263244;
        padding: 0.62rem 0.75rem;
        text-align: left;
        font-size: 0.9rem;
        white-space: nowrap;
    }
    .result-table th {
        background: #1b2540;
        color: #dbe7ff;
        font-weight: 700;
    }
    .result-table td {
        color: #e5e7eb;
    }
    .result-table tr:last-child td {
        border-bottom: 0;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# CACHE CÁC MODEL
# ============================================
@st.cache_resource(show_spinner=False)
def get_yolo_model(model_path):
    """Cache model YOLOv8 để không load lại mỗi lần rerun."""
    return load_yolo_model(model_path)


@st.cache_resource(show_spinner=False)
def get_ocr_reader():
    """Cache EasyOCR reader."""
    return load_ocr_reader()


@st.cache_resource(show_spinner=False)
def get_esrgan_model():
    """Cache Real-ESRGAN model (nếu cài được)."""
    if is_esrgan_available():
        return load_esrgan_model()
    return None


def get_confidence_badge(confidence):
    """Tạo badge HTML cho confidence score."""
    pct = confidence * 100
    if confidence >= 0.8:
        css_class = "confidence-high"
    elif confidence >= 0.5:
        css_class = "confidence-medium"
    else:
        css_class = "confidence-low"
    return f'<span class="{css_class}">{pct:.1f}%</span>'


def sync_cuda_if_available():
    """Synchronize CUDA before timing GPU work so elapsed time is accurate."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def start_timer(sync_gpu=False):
    if sync_gpu:
        sync_cuda_if_available()
    return time.perf_counter()


def stop_timer(start_time, sync_gpu=False):
    if sync_gpu:
        sync_cuda_if_available()
    return time.perf_counter() - start_time


DETECT_LOCK_CONFIDENCE = 0.55
OCR_LOCK_CONFIDENCE = 0.70
MAX_OCR_ATTEMPTS_PER_TRACK = 5
DEFAULT_VIDEO_OCR_INTERVAL = 5
USE_ESRGAN_FOR_IMAGE = True
USE_ESRGAN_FOR_VIDEO = False
OUTPUT_ROOT = Path("outputs")
RUN_OUTPUT_DIR = OUTPUT_ROOT / "runs"


def normalize_plate_key(plate_text):
    """Create a stable key so the same plate is listed once in video results."""
    if not plate_text:
        return None

    text = str(plate_text).strip()
    if text.startswith("Không đọc") or text.startswith("Lỗi"):
        return None

    key = "".join(ch for ch in text.upper() if ch.isalnum())
    return key if len(key) >= 4 else None


def plate_edit_distance(a, b):
    """Small Levenshtein implementation for merging near-duplicate plate reads."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ca != cb)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def are_similar_plates(left, right):
    left_key = normalize_plate_key(left)
    right_key = normalize_plate_key(right)
    if not left_key or not right_key:
        return False

    if left_key == right_key:
        return True

    min_len = min(len(left_key), len(right_key))
    max_len = max(len(left_key), len(right_key))
    if min_len < 5 or max_len > min_len + 1:
        return False

    return plate_edit_distance(left_key, right_key) <= 1


def parse_percent_value(value):
    try:
        return float(str(value).strip().replace("%", "")) / 100.0
    except Exception:
        return 0.0


def merge_locked_results(unique_results):
    groups = []

    locked_results = [
        result.copy()
        for result in unique_results.values()
        if result.get("_locked") and normalize_plate_key(result.get("Biển số"))
    ]
    locked_results.sort(key=lambda row: row.get("Frame đầu", 0))

    for result in locked_results:
        matched_group = None
        for group in groups:
            if are_similar_plates(result["Biển số"], group["Biển số"]):
                matched_group = group
                break

        if matched_group is None:
            result["_track_ids"] = [str(result.get("Track ID", "-"))]
            groups.append(result)
            continue

        matched_group["_track_ids"].append(str(result.get("Track ID", "-")))
        matched_group["Frame đầu"] = min(matched_group["Frame đầu"], result["Frame đầu"])
        matched_group["Frame cuối"] = max(matched_group["Frame cuối"], result["Frame cuối"])
        matched_group["Số frame thấy"] += result.get("Số frame thấy", 0)
        matched_group["Số lần OCR"] += result.get("Số lần OCR", 0)

        if result.get("_best_score", 0.0) > matched_group.get("_best_score", 0.0):
            matched_group["Biển số"] = result["Biển số"]
            matched_group["YOLO Conf tốt nhất"] = result["YOLO Conf tốt nhất"]
            matched_group["OCR Conf tốt nhất"] = result["OCR Conf tốt nhất"]
            matched_group["_best_score"] = result.get("_best_score", 0.0)
            matched_group["_best_yolo_conf"] = result.get("_best_yolo_conf", 0.0)
            matched_group["_best_ocr_conf"] = result.get("_best_ocr_conf", 0.0)
        else:
            if parse_percent_value(result["YOLO Conf tốt nhất"]) > parse_percent_value(matched_group["YOLO Conf tốt nhất"]):
                matched_group["YOLO Conf tốt nhất"] = result["YOLO Conf tốt nhất"]
            if parse_percent_value(result["OCR Conf tốt nhất"]) > parse_percent_value(matched_group["OCR Conf tốt nhất"]):
                matched_group["OCR Conf tốt nhất"] = result["OCR Conf tốt nhất"]

    rows = []
    for group in groups:
        clean_result = group.copy()
        track_ids = sorted(
            set(clean_result.pop("_track_ids", [])),
            key=lambda item: int(item) if str(item).isdigit() else 10**9
        )
        clean_result["Track ID"] = ",".join(track_ids)
        clean_result.pop("_best_score", None)
        clean_result.pop("_best_yolo_conf", None)
        clean_result.pop("_best_ocr_conf", None)
        clean_result.pop("_has_text", None)
        clean_result.pop("_locked", None)
        rows.append(clean_result)

    return rows


def render_results_table(rows):
    """Render a lightweight HTML table instead of Streamlit's JS table component."""
    if not rows:
        return

    columns = list(rows[0].keys())
    header = "".join(f"<th>{html.escape(str(col))}</th>" for col in columns)
    body_rows = []

    for row in rows:
        cells = "".join(
            f"<td>{html.escape(str(row.get(col, '')))}</td>"
            for col in columns
        )
        body_rows.append(f"<tr>{cells}</tr>")

    st.markdown(
        f"""
        <div class="result-table-wrap">
            <table class="result-table">
                <thead>
                    <tr>{header}</tr>
                </thead>
                <tbody>
                    {''.join(body_rows)}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )


def safe_filename(name):
    stem = Path(name).stem if name else "video"
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return safe.strip("_") or "video"


def create_video_run_dir(uploaded_video):
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = safe_filename(uploaded_video.name)
    run_dir = RUN_OUTPUT_DIR / f"{timestamp}_{base_name}"
    suffix = 1
    while run_dir.exists():
        run_dir = RUN_OUTPUT_DIR / f"{timestamp}_{base_name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_results_csv(rows, output_path):
    if not rows:
        return None

    with open(output_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return str(output_path)


def read_results_csv(csv_path):
    if not csv_path or not Path(csv_path).exists():
        return []

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def is_within_directory(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def find_first_existing(paths):
    for path in paths:
        if path and Path(path).exists():
            return str(path)
    return None


def list_video_history():
    if not RUN_OUTPUT_DIR.exists():
        return []

    runs = []
    for run_dir in RUN_OUTPUT_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        result_video = find_first_existing([
            run_dir / "result_web.mp4",
            run_dir / "result_raw.mp4",
        ])
        source_video = find_first_existing(
            sorted(run_dir.glob("source_original.*"))
        )
        results_csv = find_first_existing([run_dir / "results.csv"])
        rows = read_results_csv(results_csv)

        modified_at = run_dir.stat().st_mtime
        runs.append({
            "name": run_dir.name,
            "run_dir": str(run_dir),
            "result_video": result_video,
            "source_video": source_video,
            "results_csv": results_csv,
            "result_count": len(rows),
            "modified_at": modified_at,
            "modified_label": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(modified_at)),
        })

    return sorted(runs, key=lambda run: run["modified_at"], reverse=True)


def delete_video_history_run(run_dir):
    run_path = Path(run_dir)
    if not is_within_directory(run_path, RUN_OUTPUT_DIR):
        raise ValueError("Đường dẫn lịch sử không hợp lệ.")
    if run_path.exists() and run_path.is_dir():
        shutil.rmtree(run_path)


def render_video_history_manager():
    st.markdown("### 🕘 Lịch sử xử lý")
    runs = list_video_history()
    if not runs:
        st.info("Chưa có lịch sử video nào trong `outputs/runs`.")
        return

    options = [
        f"{run['modified_label']} | {run['name']} | {run['result_count']} kết quả"
        for run in runs
    ]
    selected_label = st.selectbox(
        "Chọn lần xử lý",
        options,
        label_visibility="collapsed",
    )
    selected_run = runs[options.index(selected_label)]

    st.markdown(f"**Run:** `{selected_run['name']}`")
    st.markdown(f"- Thư mục: `{os.path.abspath(selected_run['run_dir'])}`")
    if selected_run["source_video"]:
        st.markdown(f"- Video gốc: `{os.path.abspath(selected_run['source_video'])}`")
    if selected_run["result_video"]:
        st.markdown(f"- Video kết quả: `{os.path.abspath(selected_run['result_video'])}`")
    if selected_run["results_csv"]:
        st.markdown(f"- CSV: `{os.path.abspath(selected_run['results_csv'])}`")

    if selected_run["result_video"] and st.checkbox("Hiển thị video kết quả", key=f"show_video_{selected_run['name']}"):
        show_video_file(selected_run["result_video"])

    rows = read_results_csv(selected_run["results_csv"])
    if rows:
        st.markdown("**Kết quả đã khóa:**")
        render_results_table(rows)
    else:
        st.info("Run này chưa có `results.csv` hoặc chưa có biển số đạt ngưỡng chắc chắn.")

    confirm_delete = st.checkbox(
        "Xác nhận xóa run này",
        key=f"confirm_delete_{selected_run['name']}",
    )
    if st.button(
        "Xóa run đã chọn",
        disabled=not confirm_delete,
        key=f"delete_run_{selected_run['name']}",
        use_container_width=True,
    ):
        delete_video_history_run(selected_run["run_dir"])
        st.success("Đã xóa run lịch sử.")
        st.rerun()


def recognize_single_plate(image_bgr, detection, ocr_reader, esrgan_model=None, use_esrgan_for_plate=True):
    """Crop, enhance, and OCR one detected plate."""
    cropped = crop_plate(image_bgr, detection['bbox'])

    try:
        enhanced, enhance_method = enhance_plate(
            cropped,
            use_esrgan=use_esrgan_for_plate,
            esrgan_model=esrgan_model
        )
    except Exception as e:
        enhanced = cropped
        enhance_method = f"Lỗi: {str(e)}"

    try:
        h, w = enhanced.shape[:2]
        aspect_ratio = w / h if h > 0 else 1.0
        ocr_results = read_plate_text(ocr_reader, enhanced)
        plate_text, ocr_confidence = format_plate_text(ocr_results, aspect_ratio=aspect_ratio)
        enhance_method = enhance_method.split('(')[0].strip()
    except Exception as e:
        ocr_results = []
        plate_text = f"Lỗi OCR: {str(e)}"
        ocr_confidence = 0.0

    return {
        "cropped": cropped,
        "enhanced": enhanced,
        "enhance_method": enhance_method,
        "ocr_results": ocr_results,
        "plate_text": plate_text,
        "ocr_confidence": ocr_confidence,
    }


def track_plates(model, image, confidence=0.25):
    """Detect plates and assign stable ByteTrack IDs for video frames."""
    try:
        results = model.track(
            image,
            conf=confidence,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            track_ids = getattr(boxes, "id", None)

            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = result.names.get(class_id, "license_plate")

                track_id = None
                if track_ids is not None:
                    track_id = int(track_ids[idx].cpu().numpy())

                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'class_name': class_name,
                    'track_id': track_id,
                })

        return detections

    except Exception as e:
        raise RuntimeError(f"Lỗi khi chạy ByteTrack: {str(e)}")


def get_video_result_key(detection, plate_text):
    """Prefer ByteTrack ID for deduplication, fallback to OCR text when needed."""
    track_id = detection.get("track_id")
    if track_id is not None:
        return f"track:{track_id}"

    plate_key = normalize_plate_key(plate_text)
    if plate_key is not None:
        return f"text:{plate_key}"

    return None


def get_track_key(detection):
    track_id = detection.get("track_id")
    if track_id is not None:
        return f"track:{track_id}"
    return None


def make_pending_plate_result(detection, track_state=None):
    track_id = detection.get("track_id")
    if track_state and track_state.get("_has_text"):
        return {
            "plate_text": track_state["Biển số"],
            "ocr_confidence": track_state["_best_ocr_conf"],
        }

    return {
        "plate_text": "Đang đọc..." if track_id is not None else "Tracking...",
        "ocr_confidence": 0.0,
    }


def should_ocr_track(track_state, should_run_ocr_frame):
    if not should_run_ocr_frame:
        return False
    if track_state is None:
        return True
    if track_state.get("_locked"):
        return False
    return track_state.get("Số lần OCR", 0) < MAX_OCR_ATTEMPTS_PER_TRACK


def is_plate_text_readable(plate_text):
    return normalize_plate_key(plate_text) is not None


def is_confident_result(yolo_confidence_value, ocr_confidence_value, plate_text):
    return (
        yolo_confidence_value >= DETECT_LOCK_CONFIDENCE
        and ocr_confidence_value >= OCR_LOCK_CONFIDENCE
        and is_plate_text_readable(plate_text)
    )


def get_ffmpeg_executable():
    """Find ffmpeg from imageio-ffmpeg, PATH, or the project .venv."""
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_path and os.path.exists(ffmpeg_path):
            return ffmpeg_path, "imageio-ffmpeg"
    except Exception:
        pass

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path, "PATH"

    project_root = Path(__file__).resolve().parent
    search_roots = [
        project_root / ".venv" / "Lib" / "site-packages" / "imageio_ffmpeg" / "binaries",
        project_root / ".venv" / "lib" / "site-packages" / "imageio_ffmpeg" / "binaries",
    ]
    for root in search_roots:
        if root.exists():
            candidates = sorted(root.glob("ffmpeg*"))
            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate), ".venv imageio-ffmpeg"

    return None, None


def transcode_video_for_browser(input_path):
    """Convert OpenCV mp4v output to H.264 so Streamlit/browser can play it."""
    output_path = os.path.splitext(input_path)[0] + "_web.mp4"
    ffmpeg_path, ffmpeg_source = get_ffmpeg_executable()
    if not ffmpeg_path:
        return (
            input_path,
            False,
            "Không tìm thấy ffmpeg. Hãy chạy: .\\.venv\\Scripts\\python.exe -m pip install imageio-ffmpeg"
        )

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        input_path,
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        output_path,
    ]

    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
        return output_path, True, f"Đã chuyển video sang H.264 bằng {ffmpeg_source} để phát trên web."
    except Exception as e:
        return input_path, False, f"Không thể chuyển H.264: {str(e)}"


def show_video_file(video_path):
    """Render a local video through bytes to avoid browser path/codec edge cases."""
    with open(video_path, "rb") as video_file:
        st.video(video_file.read())


def reset_yolo_tracker(model):
    """Reset cached Ultralytics trackers before starting a new uploaded video."""
    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", None) if predictor is not None else None

    if not trackers:
        return

    for tracker in trackers:
        tracker.tracked_stracks = []
        tracker.lost_stracks = []
        tracker.removed_stracks = []
        tracker.frame_id = 0
        if hasattr(tracker, "get_kalmanfilter"):
            tracker.kalman_filter = tracker.get_kalmanfilter()
        if hasattr(tracker, "reset_id"):
            tracker.reset_id()


def draw_plate_text_on_frame(frame, detections, plate_results):
    """Draw OCR text under each detected plate in a video frame."""
    annotated = frame.copy()

    for det, plate_result in zip(detections, plate_results):
        x1, y1, x2, y2 = det['bbox']
        text = plate_result["plate_text"]
        track_id = det.get("track_id")
        track_label = f"ID {track_id} | " if track_id is not None else ""
        label = f"{track_label}{text} | OCR {plate_result['ocr_confidence']:.0%}"

        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        )
        label_y1 = min(frame.shape[0] - text_h - baseline - 6, y2 + 6)
        label_y2 = label_y1 + text_h + baseline + 6
        label_x2 = min(frame.shape[1] - 1, x1 + text_w + 8)

        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), (0, 255, 0), -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 4, label_y2 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2
        )

    return annotated


def save_uploaded_video_to_run(uploaded_video, run_dir):
    suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
    input_path = Path(run_dir) / f"source_original{suffix}"
    with open(input_path, "wb") as video_file:
        video_file.write(uploaded_video.getbuffer())
    return str(input_path)


def transcode_to_named_video(input_path, output_path):
    converted_path, ready, note = transcode_video_for_browser(input_path)
    converted_path = Path(converted_path)
    output_path = Path(output_path)

    if converted_path != output_path and converted_path.exists():
        if output_path.exists():
            output_path.unlink()
        converted_path.rename(output_path)
        converted_path = output_path

    return str(converted_path), ready, note


def process_video_file(uploaded_video, ocr_interval):
    """Run plate detection/OCR on a video and return the annotated output path."""
    run_dir = create_video_run_dir(uploaded_video)
    input_path = save_uploaded_video_to_run(uploaded_video, run_dir)
    output_path = str(run_dir / "result_raw.mp4")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        try:
            os.remove(input_path)
        except OSError:
            pass
        raise RuntimeError("Không thể mở video đã upload.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        try:
            os.remove(input_path)
        except OSError:
            pass
        raise RuntimeError("Không đọc được kích thước video.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        try:
            os.remove(input_path)
        except OSError:
            pass
        raise RuntimeError("Không thể tạo file video kết quả.")

    progress_bar = st.progress(0, text="Đang khởi tạo xử lý video...")
    status_box = st.empty()
    source_web_path = None
    source_video_note = None
    source_browser_ready = False

    try:
        progress_bar.progress(3, text="Đang chuẩn bị video gốc để hiển thị...")
        source_web_path, source_browser_ready, source_video_note = transcode_to_named_video(
            input_path,
            run_dir / "source_web.mp4"
        )

        progress_bar.progress(5, text="🔄 Đang load model YOLOv8...")
        yolo_model = get_yolo_model(model_path)
        reset_yolo_tracker(yolo_model)

        progress_bar.progress(10, text="🔤 Đang load EasyOCR...")
        ocr_reader = get_ocr_reader()

        esrgan_model = None
        if USE_ESRGAN_FOR_VIDEO:
            progress_bar.progress(15, text="✨ Đang load Real-ESRGAN...")
            esrgan_model = get_esrgan_model()
    except Exception:
        cap.release()
        writer.release()
        try:
            os.remove(input_path)
        except OSError:
            pass
        raise

    frame_index = 0
    tracked_frames = 0
    ocr_frames = 0
    tracked_boxes = 0
    ocr_reads = 0
    duplicate_reads = 0
    unique_results = {}
    started_at = start_timer(sync_gpu=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            annotated = frame
            tracked_frames += 1
            detections = track_plates(yolo_model, frame, confidence=yolo_confidence)

            if detections:
                tracked_boxes += len(detections)
                annotated = draw_bounding_boxes(frame, detections)
                plate_results = []
                should_run_ocr = frame_index % ocr_interval == 0

                if should_run_ocr:
                    ocr_frames += 1

                for det in detections:
                    track_key = get_track_key(det)
                    existing = None
                    if track_key is not None:
                        existing = unique_results.get(track_key)
                        if existing is None:
                            existing = {
                                "Track ID": det.get("track_id", "-"),
                                "Biển số": "Đang đọc...",
                                "Frame đầu": frame_index,
                                "Thời gian đầu": f"{frame_index / fps:.2f}s",
                                "Frame cuối": frame_index,
                                "Số frame thấy": 0,
                                "Số lần OCR": 0,
                                "Trạng thái OCR": "Đang đọc",
                                "YOLO Conf tốt nhất": f"{det['confidence']:.1%}",
                                "OCR Conf tốt nhất": "0.0%",
                                "_best_score": 0.0,
                                "_best_yolo_conf": det['confidence'],
                                "_best_ocr_conf": 0.0,
                                "_has_text": False,
                                "_locked": False,
                            }
                            unique_results[track_key] = existing
                        else:
                            existing["Frame cuối"] = frame_index

                        existing["Số frame thấy"] += 1

                    run_ocr_for_track = should_ocr_track(existing, should_run_ocr)

                    if run_ocr_for_track:
                        plate_result = recognize_single_plate(
                            frame,
                            det,
                            ocr_reader,
                            esrgan_model,
                            use_esrgan_for_plate=USE_ESRGAN_FOR_VIDEO,
                        )
                        ocr_reads += 1
                        readable_text = is_plate_text_readable(plate_result["plate_text"])
                        confident_result = is_confident_result(
                            det['confidence'],
                            plate_result['ocr_confidence'],
                            plate_result["plate_text"],
                        )

                        result_key = get_video_result_key(det, plate_result["plate_text"])
                        if result_key is not None:
                            existing = unique_results.get(result_key)
                            if existing is None:
                                unique_results[result_key] = {
                                    "Track ID": det.get("track_id", "-"),
                                    "Biển số": plate_result["plate_text"],
                                    "Frame đầu": frame_index,
                                    "Thời gian đầu": f"{frame_index / fps:.2f}s",
                                    "Frame cuối": frame_index,
                                    "Số frame thấy": 1,
                                    "Số lần OCR": 1,
                                    "Trạng thái OCR": (
                                        "Đã khóa" if confident_result else "Đang đọc"
                                    ),
                                    "YOLO Conf tốt nhất": f"{det['confidence']:.1%}",
                                    "OCR Conf tốt nhất": f"{plate_result['ocr_confidence']:.1%}",
                                    "_best_score": det['confidence'] + plate_result['ocr_confidence'],
                                    "_best_yolo_conf": det['confidence'],
                                    "_best_ocr_conf": plate_result['ocr_confidence'],
                                    "_has_text": readable_text,
                                    "_locked": confident_result,
                                }
                            else:
                                if existing["Số lần OCR"] > 0:
                                    duplicate_reads += 1
                                existing["Frame cuối"] = frame_index
                                existing["Số lần OCR"] += 1

                                score = det['confidence'] + plate_result['ocr_confidence']
                                if readable_text and score > existing["_best_score"]:
                                    existing["Biển số"] = plate_result["plate_text"]
                                    existing["YOLO Conf tốt nhất"] = f"{det['confidence']:.1%}"
                                    existing["OCR Conf tốt nhất"] = f"{plate_result['ocr_confidence']:.1%}"
                                    existing["_best_score"] = score
                                    existing["_best_yolo_conf"] = det['confidence']
                                    existing["_best_ocr_conf"] = plate_result['ocr_confidence']
                                    existing["_has_text"] = True
                                if confident_result:
                                    existing["_locked"] = True
                                    existing["Trạng thái OCR"] = "Đã khóa"
                                elif existing["Số lần OCR"] >= MAX_OCR_ATTEMPTS_PER_TRACK:
                                    existing["Trạng thái OCR"] = "Dừng OCR"
                    else:
                        existing = unique_results.get(track_key) if track_key is not None else None
                        plate_result = make_pending_plate_result(det, existing)

                    plate_results.append(plate_result)

                annotated = draw_plate_text_on_frame(annotated, detections, plate_results)

            writer.write(annotated)
            frame_index += 1

            if total_frames > 0:
                pct = min(95, 15 + int(80 * frame_index / total_frames))
                progress_bar.progress(
                    pct,
                    text=f"🎬 Đang xử lý video: {frame_index}/{total_frames} frame..."
                )
            elif frame_index % 25 == 0:
                status_box.info(f"Đã xử lý {frame_index} frame...")
    except Exception:
        cap.release()
        writer.release()
        for path in (input_path, output_path):
            try:
                os.remove(path)
            except OSError:
                pass
        raise

    cap.release()
    writer.release()
    try:
        os.remove(input_path)
    except OSError:
        pass

    elapsed = stop_timer(started_at, sync_gpu=True)
    progress_bar.progress(98, text="Đang tối ưu video để phát trên web...")
    final_output_path, browser_ready, video_note = transcode_video_for_browser(output_path)
    if final_output_path != output_path:
        final_path = Path(final_output_path)
        target_final_path = run_dir / "result_web.mp4"
        if final_path != target_final_path:
            if target_final_path.exists():
                target_final_path.unlink()
            final_path.rename(target_final_path)
            final_output_path = str(target_final_path)
        try:
            os.remove(output_path)
        except OSError:
            pass

    progress_bar.progress(100, text="✅ Hoàn tất xử lý video!")
    results_summary = merge_locked_results(unique_results)
    results_csv_path = write_results_csv(results_summary, run_dir / "results.csv")

    return {
        "run_dir": str(run_dir),
        "source_path": input_path,
        "source_web_path": source_web_path,
        "source_browser_ready": source_browser_ready,
        "source_video_note": source_video_note,
        "output_path": final_output_path,
        "results_csv_path": results_csv_path,
        "browser_ready": browser_ready,
        "video_note": video_note,
        "total_frames": frame_index,
        "processed_frames": tracked_frames,
        "ocr_frames": ocr_frames,
        "detected_plates": tracked_boxes,
        "ocr_reads": ocr_reads,
        "detect_lock_confidence": DETECT_LOCK_CONFIDENCE,
        "ocr_lock_confidence": OCR_LOCK_CONFIDENCE,
        "max_ocr_attempts": MAX_OCR_ATTEMPTS_PER_TRACK,
        "unique_plates": len(results_summary),
        "duplicate_reads": duplicate_reads,
        "elapsed": elapsed,
        "results_summary": results_summary,
    }


# ============================================
# CẤU HÌNH MẶC ĐỊNH (Không dùng Sidebar)
# ============================================
yolo_confidence = 0.15
use_esrgan = USE_ESRGAN_FOR_IMAGE
show_crop = True
model_path = "weights/best.pt"


# ============================================
# MAIN PAGE
# ============================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🚗 Hệ thống nhận diện biển số xe Việt Nam</h1>
    <p>YOLOv8 Detection • Real-ESRGAN Enhancement • EasyOCR Recognition</p>
</div>
""", unsafe_allow_html=True)

# Upload ảnh/video
st.markdown("### 📤 Upload dữ liệu")
input_mode = st.radio(
    "Chọn chế độ nhận diện",
    ["Ảnh", "Video"],
    horizontal=True,
)

if input_mode == "Ảnh":
    uploaded_file = st.file_uploader(
        "Chọn ảnh chứa biển số xe (JPG, JPEG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Hỗ trợ ảnh JPG, JPEG, PNG. Tối đa 10MB.",
        label_visibility="collapsed"
    )
    video_ocr_interval = DEFAULT_VIDEO_OCR_INTERVAL
else:
    uploaded_file = st.file_uploader(
        "Chọn video chứa biển số xe (MP4, AVI, MOV, MKV)",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Video càng dài sẽ xử lý càng lâu. ByteTrack chạy mọi frame, OCR chạy theo tần suất mặc định.",
        label_visibility="collapsed"
    )
    video_ocr_interval = DEFAULT_VIDEO_OCR_INTERVAL

# Nút nhận diện
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    recognize_button = st.button(
        "🔍 Nhận diện biển số" if input_mode == "Ảnh" else "🎬 Detect video",
        use_container_width=True,
        disabled=(uploaded_file is None),
    )

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

if input_mode == "Video" and not recognize_button:
    render_video_history_manager()
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# ============================================
# XỬ LÝ KHI BẤM NÚT NHẬN DIỆN
# ============================================
if uploaded_file is not None and recognize_button:
    if input_mode == "Video":
        try:
            video_result = process_video_file(uploaded_file, video_ocr_interval)
        except FileNotFoundError as e:
            st.error(f"❌ {str(e)}")
            st.info(
                "💡 **Hướng dẫn:** Hãy đặt file `best.pt` (model YOLOv8 đã train) "
                "vào thư mục `weights/` trong project."
            )
            st.stop()
        except Exception as e:
            st.error(f"❌ Lỗi xử lý video: {str(e)}")
            st.stop()

        video_col_original, video_col_result = st.columns(2)
        with video_col_original:
            st.markdown("### 🎞️ Video gốc")
            if video_result["source_web_path"]:
                show_video_file(video_result["source_web_path"])
                if video_result["source_browser_ready"]:
                    st.success(video_result["source_video_note"])
                else:
                    st.warning(video_result["source_video_note"])

        with video_col_result:
            st.markdown("### 🎯 Video kết quả")
            show_video_file(video_result["output_path"])
            if video_result["browser_ready"]:
                st.success(video_result["video_note"])
            else:
                st.warning(video_result["video_note"])

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-item">
                <div class="metric-value">{video_result['total_frames']}</div>
                <div class="metric-label">Tổng frame</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{video_result['processed_frames']}</div>
                <div class="metric-label">Frame đã tracking</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{video_result['detected_plates']}</div>
                <div class="metric-label">Box đã tracking</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{video_result['ocr_reads']}</div>
                <div class="metric-label">Lượt OCR</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{video_result['unique_plates']}</div>
                <div class="metric-label">Track duy nhất</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{video_result['duplicate_reads']}</div>
                <div class="metric-label">Lượt trùng đã gộp</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{video_result['elapsed']:.1f}s</div>
                <div class="metric-label">Thời gian xử lý</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if video_result["results_summary"]:
            st.markdown("### 📊 Kết quả nhận diện")
            render_results_table(video_result["results_summary"])
        else:
            st.warning("⚠️ Chưa có track nào đạt đủ ngưỡng chắc chắn để lưu vào kết quả nhận diện.")

        st.info(
            f"Kết quả chỉ được lưu khi YOLO >= {video_result['detect_lock_confidence']:.0%} "
            f"và OCR >= {video_result['ocr_lock_confidence']:.0%}; "
            f"mỗi track OCR tối đa {video_result['max_ocr_attempts']} lần rồi dùng kết quả tốt nhất."
        )
        st.markdown("### 📁 File output")
        st.markdown(f"- Thư mục run: `{os.path.abspath(video_result['run_dir'])}`")
        st.markdown(f"- Video gốc: `{os.path.abspath(video_result['source_path'])}`")
        if video_result["source_web_path"]:
            st.markdown(f"- Video gốc H.264: `{os.path.abspath(video_result['source_web_path'])}`")
        st.markdown(f"- Video kết quả: `{os.path.abspath(video_result['output_path'])}`")
        if video_result["results_csv_path"]:
            st.markdown(f"- CSV kết quả: `{os.path.abspath(video_result['results_csv_path'])}`")

        st.stop()

    # === Bước 0: Validate ảnh ===
    is_valid, msg = validate_image(uploaded_file)
    if not is_valid:
        st.error(f"❌ {msg}")
        st.stop()

    # Load ảnh
    try:
        image_bgr = load_image_from_upload(uploaded_file)
        image_bgr = resize_image(image_bgr)
    except Exception as e:
        st.error(f"❌ Không thể đọc ảnh: {str(e)}")
        st.stop()

    # === Progress bar ===
    progress_bar = st.progress(0, text="Đang khởi tạo...")

    # === Bước 1: Load model YOLOv8 ===
    progress_bar.progress(10, text="🔄 Đang load model YOLOv8...")
    try:
        yolo_model = get_yolo_model(model_path)
    except FileNotFoundError as e:
        st.error(f"❌ {str(e)}")
        st.info(
            "💡 **Hướng dẫn:** Hãy đặt file `best.pt` (model YOLOv8 đã train) "
            "vào thư mục `weights/` trong project."
        )
        st.stop()
    except Exception as e:
        st.error(f"❌ Lỗi load model: {str(e)}")
        st.stop()

    # === Bước 2: Detect biển số ===
    progress_bar.progress(30, text="🔍 Đang phát hiện biển số xe...")
    try:
        start_time = start_timer(sync_gpu=True)
        detections = detect_plates(yolo_model, image_bgr, confidence=yolo_confidence)
        detect_time = stop_timer(start_time, sync_gpu=True)
    except Exception as e:
        st.error(f"❌ Lỗi detection: {str(e)}")
        st.stop()

    # Kiểm tra có phát hiện được biển số không
    if len(detections) == 0:
        progress_bar.progress(100, text="⚠️ Hoàn tất - Không tìm thấy biển số")
        st.warning(
            "⚠️ **Không phát hiện được biển số xe trong ảnh.**\n\n"
            "**Nguyên nhân có thể:**\n"
            "- Ảnh không chứa biển số xe\n"
            "- Biển số bị che khuất hoặc quá mờ\n"
            "- Ảnh quá mờ hoặc nhiễu\n"
            "- Model chưa được train với loại biển số này\n\n"
            "**Gợi ý:** Thử dùng ảnh rõ nét hơn để đạt kết quả tốt nhất."
        )
        st.stop()

    # === Bước 3: Vẽ bounding box ===
    progress_bar.progress(45, text="📦 Đang vẽ bounding box...")
    annotated_image = draw_bounding_boxes(image_bgr, detections)

    st.markdown(f"### 🎯 Kết quả Detection — Tìm thấy **{len(detections)}** biển số")
    col_original, col_detected = st.columns(2)
    with col_original:
        st.markdown("**📷 Ảnh gốc**")
        st.image(bgr_to_rgb(image_bgr), use_container_width=True)
    with col_detected:
        st.markdown("**🎯 Sau detect**")
        st.image(bgr_to_rgb(annotated_image), use_container_width=True)

    # Metrics tổng quan
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-item">
            <div class="metric-value">{len(detections)}</div>
            <div class="metric-label">Biển số tìm thấy</div>
        </div>
        <div class="metric-item">
            <div class="metric-value">{detect_time:.2f}s</div>
            <div class="metric-label">Thời gian detect</div>
        </div>
        <div class="metric-item">
            <div class="metric-value">{max(d['confidence'] for d in detections):.1%}</div>
            <div class="metric-label">Confidence cao nhất</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    post_detect_start_time = start_timer(sync_gpu=True)

    # === Bước 4: Load OCR ===
    progress_bar.progress(55, text="🔤 Đang load EasyOCR...")
    try:
        ocr_reader = get_ocr_reader()
    except Exception as e:
        st.error(f"❌ Lỗi load EasyOCR: {str(e)}")
        st.stop()

    # === Bước 5: Load ESRGAN (nếu dùng) ===
    esrgan_model = None
    if use_esrgan:
        progress_bar.progress(60, text="✨ Đang load Real-ESRGAN...")
        esrgan_model = get_esrgan_model()

    # === Bước 6: Xử lý từng biển số ===
    st.markdown("### 📋 Chi tiết từng biển số")

    total_plates = len(detections)
    results_summary = []

    for i, det in enumerate(detections):
        progress_pct = 65 + int(30 * (i + 1) / total_plates)
        progress_bar.progress(
            progress_pct,
            text=f"🔄 Đang xử lý biển số {i + 1}/{total_plates}..."
        )

        # Crop biển số
        cropped = crop_plate(image_bgr, det['bbox'])

        # Làm nét
        try:
            enhanced, enhance_method = enhance_plate(
                cropped,
                use_esrgan=use_esrgan,
                esrgan_model=esrgan_model
            )
        except Exception as e:
            enhanced = cropped
            enhance_method = f"Lỗi: {str(e)}"

        # OCR
        try:
            # Tính aspect_ratio để xử lý lỗi thiếu/sai ký tự chuẩn xác hơn
            h, w = enhanced.shape[:2]
            aspect_ratio = w / h if h > 0 else 1.0

            # 1. Chỉ thực hiện OCR trên ảnh đã làm nét bằng ESRGAN (theo yêu cầu)
            ocr_results = read_plate_text(ocr_reader, enhanced)
            plate_text, ocr_confidence = format_plate_text(ocr_results, aspect_ratio=aspect_ratio)
            
            # Dọn dẹp tên method để hiển thị
            enhance_method = enhance_method.split('(')[0].strip()
            
        except Exception as e:
            ocr_results = []
            plate_text = f"Lỗi OCR: {str(e)}"
            ocr_confidence = 0.0

        # === HIỂN THỊ KẾT QUẢ ===
        st.markdown(f"""
        <div class="result-card">
            <h4 style="color: #667eea; margin-top: 0;">🔖 Biển số #{i + 1}</h4>
            <div style="text-align: center; margin: 1rem 0;">
                <div class="plate-badge">{plate_text}</div>
            </div>
            <div class="metric-row">
                <div class="metric-item">
                    <div class="metric-value">{det['confidence']:.1%}</div>
                    <div class="metric-label">YOLO Confidence</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{ocr_confidence:.1%}</div>
                    <div class="metric-label">OCR Confidence</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">{enhance_method.split('(')[0].strip()}</div>
                    <div class="metric-label">Phương pháp làm nét</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Hiển thị ảnh crop và ảnh sau làm nét
        if show_crop:
            col1, col2 = st.columns(2)
            with col1:
                h_orig, w_orig = cropped.shape[:2]
                h_enh, w_enh = enhanced.shape[:2]
                st.markdown(f"**🖼️ Ảnh crop biển số (Phóng to từ {w_orig}x{h_orig} lên {w_enh}x{h_enh}):**")
                # Phóng to ảnh crop bằng phương pháp Nearest Neighbor để giữ nguyên độ vỡ/pixelate
                cropped_resized = cv2.resize(cropped, (w_enh, h_enh), interpolation=cv2.INTER_NEAREST)
                st.image(bgr_to_rgb(cropped_resized), use_container_width=True)
            with col2:
                st.markdown(f"**✨ Sau khi làm nét ({enhance_method} — {w_enh}x{h_enh}):**")
                st.image(bgr_to_rgb(enhanced), use_container_width=True)

        # Chi tiết OCR
        if ocr_results:
            with st.expander(f"🔍 Chi tiết OCR biển số #{i + 1}", expanded=False):
                for j, ocr_res in enumerate(ocr_results):
                    st.markdown(
                        f"- Text: `{ocr_res['text']}` — "
                        f"Confidence: {get_confidence_badge(ocr_res['confidence'])}",
                        unsafe_allow_html=True
                    )

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Thêm vào bảng tổng kết
        results_summary.append({
            "STT": i + 1,
            "Biển số": plate_text,
            "YOLO Conf": f"{det['confidence']:.1%}",
            "OCR Conf": f"{ocr_confidence:.1%}",
        })

    post_detect_time = stop_timer(post_detect_start_time, sync_gpu=True)

    # === Hoàn tất ===
    progress_bar.progress(100, text="✅ Hoàn tất nhận diện!")

    # === Lưu kết quả (optional) ===
    st.markdown("### 📊 Tổng kết")
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-item">
            <div class="metric-value">{post_detect_time:.2f}s</div>
            <div class="metric-label">Sau detect đến kết quả</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    render_results_table(results_summary)

elif uploaded_file is not None:
    if input_mode == "Video":
        st.markdown("### 🎞️ Preview video")
        size_mb = uploaded_file.size / (1024 * 1024)
        current_video_key = f"{uploaded_file.name}:{uploaded_file.size}"
        if st.session_state.get("last_uploaded_video_key") != current_video_key:
            st.session_state["last_uploaded_video_key"] = current_video_key
            st.toast(
                "Đã tải video lên. Bấm Detect video để bắt đầu xử lý.",
                icon="🎞️",
            )
        st.markdown(f"**File đã chọn:** `{uploaded_file.name}` ({size_mb:.2f} MB)")
        st.info("Video đã sẵn sàng. Bấm **Detect video** để bắt đầu xử lý và tạo video xem trên web.")
    else:
        # Hiển thị preview ảnh khi chưa bấm nút
        try:
            image_bgr = load_image_from_upload(uploaded_file)
            image_bgr = resize_image(image_bgr)
            st.markdown("### 📷 Preview ảnh")
            st.image(bgr_to_rgb(image_bgr), use_container_width=True)
            st.info("👆 Bấm nút **Nhận diện biển số** để bắt đầu xử lý.")
        except Exception as e:
            st.error(f"❌ Không thể đọc ảnh: {str(e)}")

else:
    # Hướng dẫn khi chưa upload dữ liệu
    st.markdown("""
    <div class="info-box" style="text-align: center; padding: 2rem;">
        <h3 style="color: #667eea; margin-top: 0;">📤 Hãy upload ảnh hoặc video để bắt đầu</h3>
        <p style="color: rgba(255,255,255,0.6);">
            Ảnh: JPG, JPEG, PNG<br>
            Video: MP4, AVI, MOV, MKV<br>
            Có thể xử lý nhiều biển số trong ảnh hoặc từng frame video
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Tips
    st.markdown("### 💡 Mẹo sử dụng")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="result-card">
            <h4 style="color: #00b09b; margin-top: 0;">📸 Chất lượng ảnh</h4>
            <p style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">
                Ảnh càng rõ nét, kết quả OCR càng chính xác.
                Tránh ảnh quá mờ hoặc biển số bị che khuất.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="result-card">
            <h4 style="color: #f7971e; margin-top: 0;">🎯 Nhận diện tự động</h4>
            <p style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">
                Hệ thống tự động điều chỉnh ngưỡng confidence và áp dụng
                thuật toán tối ưu để phát hiện biển số chuẩn xác.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="result-card">
            <h4 style="color: #764ba2; margin-top: 0;">✨ Tự động làm nét</h4>
            <p style="color: rgba(255,255,255,0.7); font-size: 0.85rem;">
                Ảnh biển số luôn được tự động làm nét để tăng tối đa
                tốc độ và độ chính xác của quá trình đọc OCR.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ============================================
# FOOTER
# ============================================
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.3); font-size: 0.75rem; padding: 1rem 0;">
    Hệ thống nhận diện biển số xe Việt Nam • YOLOv8 + ESRGAN + EasyOCR<br>
    Developed with ❤️ using Streamlit
</div>
""", unsafe_allow_html=True)
