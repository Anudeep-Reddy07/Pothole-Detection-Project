import os
import tempfile
from pathlib import Path
from io import BytesIO
import base64

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ---------- CONFIG ----------
MODEL_PATH = "best.pt"  # your trained model file
CONF_THRESHOLD = 0.25   # confidence threshold for detections

# ---------- UTIL: LOAD MODEL ----------
@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    return model

# ---------- UTIL: DOWNLOAD FROM URL (FIXED FOR BASE64) ----------
def download_from_url(url: str, suffix: str | None = None) -> Path:
    """
    Download a file from URL into a temp file and return the path.
    Supports both http/https URLs and base64 data URLs (data:image/jpeg;base64,...).
    """
    
    # Case 1: Base64 data URL support
    if url.startswith("data:"):
        try:
            header, b64data = url.split(",", 1)
            # Infer suffix from header
            if suffix is None:
                if "image/jpeg" in header or "image/jpg" in header:
                    suffix = ".jpg"
                elif "image/png" in header:
                    suffix = ".png"
                elif "video/mp4" in header:
                    suffix = ".mp4"
                else:
                    suffix = ".bin"
            
            # Decode base64
            binary = base64.b64decode(b64data)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(binary)
            tmp.flush()
            tmp.close()
            return Path(tmp.name)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 data URL: {str(e)}")
    
    # Case 2: Normal http/https URL
    response = requests.get(url, stream=True)
    response.raise_for_status()

    if suffix is None:
        # Try to infer from URL path
        url_path = url.split("?")[0]  # strip query
        _, ext = os.path.splitext(url_path)
        suffix = ext if ext else ".bin"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            tmp.write(chunk)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)

# ---------- CHECK IF VIDEO IS READABLE ----------
def is_valid_video(path: Path) -> bool:
    """Return True if at least one frame can be read from the video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return False
    ret, frame = cap.read()
    cap.release()
    return ret and frame is not None

# ---------- RUN YOLO ON VIDEO MANUALLY ----------
def process_video_with_yolo(
    video_path: Path,
    model: YOLO,
    conf: float = 0.25,
    max_frames: int | None = None,
):
    """
    Read a video with OpenCV, run YOLO frame-by-frame, and save an annotated video.
    Returns:
        all_results: list of per-frame YOLO Results
        annotated_video_path: Path to the saved annotated video file
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video with OpenCV.")

    # Get properties (with safe fallbacks)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    # Prepare video writer for annotated output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    annotated_video_path = Path(tmp_out.name)
    out = cv2.VideoWriter(str(annotated_video_path), fourcc, fps, (width, height))

    all_results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if max_frames is not None and frame_idx > max_frames:
            break

        # Run YOLO on this frame
        results = model.predict(
            source=frame,
            conf=conf,
            save=False,
            verbose=False,
        )
        r = results[0]
        all_results.append(r)

        # Get annotated frame (BGR) and write to output video
        annotated_frame = r.plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

    return all_results, annotated_video_path

# ---------- UTIL: SEVERITY & REPORT ----------
def compute_severity_and_report(
    results,
    is_video: bool = False,
    max_frames_for_report: int = 30,
):
    """
    Build a severity report from YOLO results.
    For images: - Use the single results[0].
    For videos: - Use up to first `max_frames_for_report` frames to keep it fast.
    """
    rows = []

    if not is_video:
        iterable = [(results[0], 0)]
    else:
        iterable = []
        for idx, r in enumerate(results):
            if idx >= max_frames_for_report:
                break
            iterable.append((r, idx))

    for r, frame_idx in iterable:
        if r.boxes is None:
            continue

        # image shape: (H, W, C)
        if isinstance(r.orig_img, Image.Image):
            w, h = r.orig_img.size
        else:
            h, w = r.orig_img.shape[:2]

        img_area = w * h if w and h else 1

        names = r.names  # dict: {class_id: class_name}

        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = names.get(cls_id, str(cls_id))
            conf = float(box.conf[0].item())

            # xyxy format: [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            box_w = max(0.0, x2 - x1)
            box_h = max(0.0, y2 - y1)
            box_area = box_w * box_h
            box_area_pct = (box_area / img_area) * 100.0 if img_area > 0 else 0.0

            # Simple per-object severity based on area and class
            base_weight = 1.0
            if "pothole" in cls_name.lower():
                base_weight = 1.5
            elif "crack" in cls_name.lower():
                base_weight = 1.2

            object_score = base_weight * box_area_pct  # higher area -> higher severity

            if object_score < 1:
                sev_level = "Very Low"
            elif object_score < 3:
                sev_level = "Low"
            elif object_score < 6:
                sev_level = "Moderate"
            elif object_score < 12:
                sev_level = "High"
            else:
                sev_level = "Critical"

            rows.append({
                "Frame": frame_idx if is_video else 0,
                "Class": cls_name,
                "Confidence": round(conf, 3),
                "BoxArea%": round(box_area_pct, 2),
                "ObjectSeverityScore": round(object_score, 2),
                "SeverityLevel": sev_level,
            })

    if not rows:
        df = pd.DataFrame(
            columns=[
                "Frame",
                "Class",
                "Confidence",
                "BoxArea%",
                "ObjectSeverityScore",
                "SeverityLevel",
            ]
        )
        overall_score = 0.0
        overall_severity = "No Issues Detected"
        return df, overall_score, overall_severity

    df = pd.DataFrame(rows)

    # Compute an overall severity score.
    mean_obj_score = df["ObjectSeverityScore"].mean()
    overall_score = max(0.0, min(100.0, mean_obj_score * 2.0))

    if overall_score < 5:
        overall_severity = "Very Low"
    elif overall_score < 15:
        overall_severity = "Low"
    elif overall_score < 30:
        overall_severity = "Moderate"
    elif overall_score < 60:
        overall_severity = "High"
    else:
        overall_severity = "Critical"

    return df, round(overall_score, 2), overall_severity

# ---------- STREAMLIT UI ----------
def main():
    st.set_page_config(page_title="CleaRoute – YOLO Road Damage Analyzer", layout="wide")

    st.title("CleaRoute – YOLO Road Damage Analyzer")
    st.caption("Upload an image or video (or use a URL), detect road damage, and generate a severity report.")

    model = load_model()

    col1, col2 = st.columns(2)
    with col1:
        media_type = st.radio("Select media type:", ["Image", "Video"], horizontal=True)
    with col2:
        source_type = st.radio("Select source:", ["Upload from device", "URL"], horizontal=True)

    uploaded_file = None
    url_input = None
    local_path: Path | None = None

    # ------ INPUTS ------
    if source_type == "Upload from device":
        if media_type == "Image":
            file_types = ["jpg", "jpeg", "png"]
        else:
            file_types = ["mp4", "mov", "avi", "mkv"]
        uploaded_file = st.file_uploader(
            "Upload file", type=file_types, accept_multiple_files=False
        )
    else:
        url_input = st.text_input(
            "Enter direct URL to image or video",
            placeholder="https://example.com/road_image.jpg or data:image/jpeg;base64,...",
        )

    if st.button("Run Detection", type="primary"):
        # Basic validation
        if source_type == "Upload from device" and not uploaded_file:
            st.warning("Please upload a file first.")
            return
        if source_type == "URL" and not url_input:
            st.warning("Please enter a URL first.")
            return

        # ------ PREPARE LOCAL FILE PATH ------
        with st.spinner("Preparing file..."):
            if source_type == "Upload from device":
                orig_name = uploaded_file.name
                _, ext = os.path.splitext(orig_name)
                ext = ext.lower()

                if media_type == "Image" and ext not in [".jpg", ".jpeg", ".png"]:
                    st.error("You selected 'Image' but uploaded a non-image file.")
                    return
                if media_type == "Video" and ext not in [".mp4", ".mov", ".avi", ".mkv"]:
                    st.error("You selected 'Video' but uploaded a non-video file.")
                    return

                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                tmp.write(uploaded_file.read())
                tmp.flush()
                tmp.close()
                local_path = Path(tmp.name)
            else:
                # URL case (now supports base64 too!)
                if media_type == "Image":
                    suffix = ".jpg"
                else:
                    suffix = ".mp4"
                try:
                    local_path = download_from_url(url_input, suffix=suffix)
                except Exception as e:
                    st.error(f"Failed to download from URL: {e}")
                    return

        if local_path is None or not local_path.exists():
            st.error("Failed to prepare local file.")
            return

        file_size = os.path.getsize(local_path)
        if file_size == 0:
            st.error("Downloaded/saved file is empty. Please try another file.")
            return

        st.success(f"File ready: {local_path.name} ({file_size} bytes)")

        # ------ RUN YOLO INFERENCE ------
        st.info("Running YOLO inference... This may take a few seconds on CPU.")

        if media_type == "Image":
            # Use Ultralytics API for single image
            try:
                results = model.predict(
                    source=str(local_path),
                    conf=CONF_THRESHOLD,
                    save=False,
                    verbose=False,
                )
            except Exception as e:
                st.error(f"YOLO failed on this image: {e}")
                return

            if not results:
                st.error("YOLO did not return any results for this image.")
                return

            result = results[0]

            # Get annotated image as numpy array
            annotated = result.plot()
            annotated_image = Image.fromarray(annotated[..., ::-1])  # BGR -> RGB

            # Show original and annotated side by side
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Original")
                st.image(str(local_path), use_column_width=True)
            with c2:
                st.subheader("Detections")
                st.image(annotated_image, use_column_width=True)

            # Severity & report
            df_report, overall_score, overall_severity = compute_severity_and_report(results, is_video=False)
            st.subheader("Severity Summary")
            st.metric("Overall Damage Score (0–100)", overall_score, help="Higher score means more severe damage.")
            st.write(f"**Overall Severity Level:** {overall_severity}")

            st.subheader("Detailed Object Report")
            st.dataframe(df_report, use_container_width=True, hide_index=True)

            # Download annotated image
            img_buf = BytesIO()
            annotated_image.save(img_buf, format="JPEG")
            img_buf.seek(0)

            st.download_button(
                "Download annotated image",
                data=img_buf,
                file_name="annotated_image.jpg",
                mime="image/jpeg",
            )

            # Download report CSV
            csv_buf = df_report.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download severity report (CSV)",
                data=csv_buf,
                file_name="severity_report.csv",
                mime="text/csv",
            )

        else:  # Video
            # Check video readability
            if not is_valid_video(local_path):
                st.error(
                    "Could not read any frames from this video.\n\n"
                    "Possible reasons:\n"
                    "- Unsupported codec\n"
                    "- Corrupted or empty file\n"
                    "- Very unusual container format\n\n"
                    "Try converting it to a standard MP4 (H.264) and upload again."
                )
                return

            with st.spinner("Processing video frame-by-frame with YOLO..."):
                try:
                    # You can set max_frames to limit processing for long videos, e.g., 500
                    all_results, annotated_video_path = process_video_with_yolo(
                        local_path,
                        model,
                        conf=CONF_THRESHOLD,
                        max_frames=None,  # or some int to cap frames
                    )
                except Exception as e:
                    st.error(f"Failed to process video: {e}")
                    return

            st.subheader("Annotated Video Preview")
            st.video(str(annotated_video_path))

            # Severity & report (we use first N frames)
            df_report, overall_score, overall_severity = compute_severity_and_report(
                all_results, is_video=True, max_frames_for_report=30
            )

            st.subheader("Severity Summary")
            st.metric("Overall Damage Score (0–100)", overall_score, help="Higher score means more severe damage.")
            st.write(f"**Overall Severity Level:** {overall_severity}")

            st.subheader("Detailed Object Report (first 30 processed frames)")
            st.dataframe(df_report, use_container_width=True, hide_index=True)

            # Download annotated video
            with open(annotated_video_path, "rb") as f:
                video_bytes = f.read()
            st.download_button(
                "Download annotated video",
                data=video_bytes,
                file_name=annotated_video_path.name,
                mime="video/mp4",
            )

            # Download report CSV
            csv_buf = df_report.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download severity report (CSV)",
                data=csv_buf,
                file_name="severity_report_video.csv",
                mime="text/csv",
            )

# Run app
if __name__ == "__main__":
    main()
