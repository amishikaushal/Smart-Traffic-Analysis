import streamlit as st
import cv2
import tempfile
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
    .main {
        background-color: #0e1117;
    }
    .title {
        font-size: 3rem;
        font-weight: 800;
        color: #00e5ff;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #cfd8dc;
        margin-bottom: 30px;
    }
    .card {
        background: #161b22;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 0 20px rgba(0,0,0,0.3);
        margin-bottom: 25px;
    }
    .counter {
        font-size: 2rem;
        color: #00e676;
        font-weight: 700;
        text-align: center;
    }
    footer {
        text-align: center;
        color: #8b949e;
        margin-top: 50px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🚦 Smart Traffic Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Real-Time Car Detection & Counting using YOLOv8</div>',
    unsafe_allow_html=True
)

# ---------------- DESCRIPTION CARD ----------------
st.markdown("""
<div class="card">
<h3>📌 Project Overview</h3>
<p>
This application leverages <b>YOLOv8 deep learning models</b> to perform 
<strong>real-time vehicle detection, tracking, and counting</strong> from traffic videos.
Each car is uniquely tracked using object IDs, ensuring accurate counting without duplicates.
</p>

<ul>
<li>🚗 Vehicle detection using YOLOv8</li>
<li>🎯 Unique object tracking</li>
<li>📍 Anchor-point based localization</li>
<li>📊 Live car count visualization</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")

model = load_model()

# ---------------- VIDEO UPLOAD ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_video = st.file_uploader(
    "📤 Upload a traffic video",
    type=["mp4", "avi", "mov"]
)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PROCESS VIDEO ----------------
if uploaded_video is not None:
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(uploaded_video.read())

    cap = cv2.VideoCapture(temp_video.name)

    if not cap.isOpened():
        st.error("❌ Video could not be loaded")
        st.stop()

    st.success("✅ Video loaded successfully")

    video_placeholder = st.empty()
    count_placeholder = st.empty()

    start = st.button("▶ Start Detection")

    if start:
        car_count = 0
        tracked_ids = set()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, persist=True, verbose=False)[0]

            if results.boxes is not None:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]

                    if label == "car" and box.conf[0] > 0.5:
                        track_id = int(box.id[0])

                        if track_id not in tracked_ids:
                            tracked_ids.add(track_id)
                            car_count += 1

                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2),
                                      (0, 255, 0), 2)

             
                        anchor_x = (x1 + x2) // 2
                        anchor_y = y2
                        cv2.circle(frame, (anchor_x, anchor_y),
                                   5, (0, 0, 255), -1)

                    
                        cv2.putText(frame, f"ID {track_id}",
                                    (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 255, 255), 2)

            count_placeholder.markdown(
                f'<div class="counter">🚘 Cars Detected: {car_count}</div>',
                unsafe_allow_html=True
            )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB")

        cap.release()
        st.success("✅ Processing complete!")


st.markdown("""
<footer>
Built with ❤️ using <b>YOLOv8</b> & <b>Streamlit</b><br>
By <b>Amishi Kaushal </b> • <a href="https://github.com/amishikaushal" target="_blank">GitHub</a>
</footer>
""", unsafe_allow_html=True)
