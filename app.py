# app.py
"""
QR Code Generator & Scanner (Streamlit)
- Generate QR codes from text and download as PNG.
- Scan QR codes from camera capture (mobile-friendly) or uploaded images.
Requirements: see requirements.txt

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
from typing import List, Tuple, Optional

import numpy as np
import qrcode
from PIL import Image
import streamlit as st
import cv2

st.set_page_config(page_title="QR Generator & Scanner", page_icon="🔳", layout="centered")


def generate_qr_image(
    data: str,
    box_size: int = 10,
    border: int = 4,
    fill_color: str = "black",
    back_color: str = "white",
    version: Optional[int] = None,
) -> Image.Image:
    """
    Generate a QR code PIL Image from text.
    - version: int 1..40 or None to let library decide
    """
    qr = qrcode.QRCode(
        version=version,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=box_size,
        border=border,
    )
    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(fill_color=fill_color, back_color=back_color).convert("RGB")
    return img


def pil_image_to_bytes(pil_img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()


def decode_qr_from_bytes(image_bytes: bytes) -> List[str]:
    """
    Decode QR codes from image bytes using OpenCV QRCodeDetector.
    Returns list of decoded strings (empty list if none found).
    Tries detectAndDecodeMulti (if available) then falls back to detectAndDecode.
    """
    # Convert bytes to numpy array for OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    detector = cv2.QRCodeDetector()

    # Try multi decode (some OpenCV builds return different signatures)
    try:
        # many builds: retval, decoded_info, points, straight_qrcodes
        retval, decoded_info, points, _ = detector.detectAndDecodeMulti(img)
        if retval and decoded_info:
            # decoded_info is a list or tuple of strings; sometimes empty strings included
            return [s for s in decoded_info if s]
    except Exception:
        # Some OpenCV versions have a different return pattern or raise; ignore and fallback
        pass

    # Fallback: single decode
    try:
        data, points, _ = detector.detectAndDecode(img)
        if isinstance(data, str) and data:
            return [data]
    except Exception:
        pass

    return []


# UI
st.title("🔳 QR Code Generator & Scanner")
st.write("Generate QR codes and scan them using your camera or by uploading an image. Designed to work on Android mobile browsers (use the site in your phone browser).")

tab1, tab2 = st.tabs(["Generate QR", "Scan QR"])

with tab1:
    st.header("Generate QR Code")
    text = st.text_area("Enter text or URL to encode", value="https://example.com", height=120)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        box_size = st.number_input("Box size (pixels)", min_value=1, max_value=40, value=10)
    with col2:
        border = st.number_input("Border (modules)", min_value=1, max_value=10, value=4)
    with col3:
        version = st.selectbox("Version (auto fits)", options=["Auto (fit)"] + list(range(1, 41)), index=0)
        version = None if version == "Auto (fit)" else int(version)
    fill_color = st.text_input("QR color (CSS name or hex)", value="black")
    back_color = st.text_input("Background color (CSS name or hex)", value="white")

    if st.button("Generate"):
        if not text.strip():
            st.warning("Please enter some text or a URL to encode.")
        else:
            img = generate_qr_image(text, box_size=box_size, border=border, fill_color=fill_color, back_color=back_color, version=version)
            st.image(img, caption="Generated QR code", use_column_width=False)
            img_bytes = pil_image_to_bytes(img, fmt="PNG")
            st.download_button("Download PNG", data=img_bytes, file_name="qrcode.png", mime="image/png")
            st.success("QR code ready — you can download or scan it from another device.")

with tab2:
    st.header("Scan QR Code")
    st.write("Use your mobile camera or upload an image containing a QR code.")

    # Camera input (mobile browsers and modern desktop browsers)
    camera_file = st.camera_input("Open camera to capture QR (recommended on mobile)")

    uploaded_file = st.file_uploader("Or upload an image file", type=["png", "jpg", "jpeg", "bmp", "gif"])

    # Prefer camera capture if provided, else uploaded file
    target_file = None
    if camera_file is not None:
        target_file = camera_file
    elif uploaded_file is not None:
        target_file = uploaded_file

    if st.button("Scan"):
        if target_file is None:
            st.warning("Please capture a photo with the camera or upload an image first.")
        else:
            image_bytes = target_file.read()
            results = decode_qr_from_bytes(image_bytes)
            if results:
                st.success(f"Found {len(results)} QR code(s):")
                for i, r in enumerate(results, start=1):
                    st.markdown(f"**{i}.** {r}")
                    # If result looks like a URL, provide a clickable link
                    if r.startswith("http://") or r.startswith("https://"):
                        st.markdown(f"[Open link]({r})")
            else:
                st.error("No QR code detected. Try taking a clearer photo or using a different image.")

    # Provide immediate quick-scan when camera_file present (auto attempt)
    if camera_file is not None and st.button("Quick scan camera capture"):
        image_bytes = camera_file.read()
        results = decode_qr_from_bytes(image_bytes)
        if results:
            st.success(f"Found {len(results)} QR code(s):")
            for i, r in enumerate(results, start=1):
                st.markdown(f"**{i}.** {r}")
                if r.startswith("http://") or r.startswith("https://"):
                    st.markdown(f"[Open link]({r})")
        else:
            st.error("No QR code detected in the camera capture.")

st.markdown("---")
st.markdown("**Notes & tips:**")
st.markdown(
    "- For best scanning results, ensure the QR code fills most of the camera frame and is well-lit.\n"
    "- `st.camera_input` uses your browser's camera permission; allow camera access when prompted.\n"
    "- If scanning fails, try uploading a clear photo of the QR code instead.\n"
)
