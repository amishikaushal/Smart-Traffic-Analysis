import streamlit as st
import cv2
import tempfile
import time
from ultralytics import YOLO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Traffic Analysis",
    page_icon="🚦",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .title { font-size: 3rem; font-weight: 800; color: #00e5ff; text-align: center; }
    .subtitle { text-align: center; font-size: 1.2rem; color: #cfd8dc; margin-bottom: 30px; }
    .card { background: #161b22; padding: 20px; border-radius: 15px; box-shadow: 0 0 20px rgba(0,0,0,0.3); margin-bottom: 25px; }
    .counter { font-size: 2.5rem; color: #00e676; font-weight: 700; text-align: center; background: #1c2128; border-radius: 10px; padding: 10px; border: 1px solid #00e676; }
    footer { text-align: center; color: #8b949e; margin-top: 50px; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🚦 Smart Traffic Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-Time Car Detection & Counting using YOLOv8</div>', unsafe_allow_html=True)

# ---------------- DESCRIPTION CARD ----------------
st.markdown("""
<div class="card">
<h3>📌 Project Overview</h3>
<p>This application leverages <b>YOLOv8 deep learning models</b> to perform <b>real-time vehicle detection, tracking, and counting</b>. 
Each car is uniquely tracked using object IDs to ensure accurate counting.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    # Using yolov8n.pt (nano) is recommended for Streamlit Cloud to prevent memory crashes
    return YOLO("yolov8n.pt") 

model = load_model()

# ---------------- VIDEO UPLOAD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_video = st.file_uploader("📤 Upload a traffic video", type=["mp4", "avi", "mov"])
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PROCESS VIDEO ----------------
if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)

    if not cap.isOpened():
        st.error("❌ Video could not be loaded")
        st.stop()

    st.success("✅ Video loaded successfully")
    
    # Placeholders for UI updates
    count_placeholder = st.empty()
    video_placeholder = st.empty()

    start = st.button("▶ Start Detection")

    if start:
        car_count = 0
        tracked_ids = set()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv8 tracking
            # persist=True keeps IDs across frames
            results = model.track(frame, persist=True, verbose=False)[0]

            # CRITICAL FIX: Check if both boxes and IDs exist before looping
            if results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                ids = results.boxes.id.cpu().numpy().astype(int)
                clss = results.boxes.cls.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()

                for box, track_id, cls, conf in zip(boxes, ids, clss, confs):
                    label = model.names[cls]

                    # Detect only cars with confidence > 0.4
                    if label == "car" and conf > 0.4:
                        if track_id not in tracked_ids:
                            tracked_ids.add(track_id)
                            car_count += 1

                        # Draw Visuals
                        x1, y1, x2, y2 = box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Anchor point (bottom center)
                        cv2.circle(frame, ((x1 + x2) // 2, y2), 5, (0, 0, 255), -1)
                        
                        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Update UI
            count_placeholder.markdown(
                f'<div class="counter">🚘 Total Cars: {car_count}</div>', 
                unsafe_allow_html=True
            )

            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Small delay to prevent the browser from freezing
            time.sleep(0.01)

        cap.release()
        st.success("✅ Processing complete!")

st.markdown(f"""
<footer>
Built with ❤️ using <b>YOLOv8</b> & <b>Streamlit</b><br>
By <b>Amishi Kaushal</b> • <a href="https://github.com/amishikaushal" target="_blank">GitHub</a>
</footer>
""", unsafe_allow_html=True)