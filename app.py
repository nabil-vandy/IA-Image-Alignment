import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import av
import math
import threading

# Page Config
st.set_page_config(page_title="ImageAssist AI Alignment", page_icon="📸")

st.title("ImageAssist: Smart Alignment Proto")
st.markdown("""
**Instructions:**
1. Upload a "Reference Photo".
2. Start the camera.
3. Follow the text overlays (UP, DOWN, CLOSER, etc.).
4. Click **"Take Photo"** when aligned.
""")

# --- GLOBAL VARIABLES (Thread-Safe Data Transfer) ---
# We use a lock to ensure the background thread reads data safely
lock = threading.Lock()
ref_data = {
    "nose": None,
    "eye_dist": None,
    "image": None
}

# --- HELPER: DISTANCE CALC ---
def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# --- STEP 1: UPLOAD REFERENCE ---
uploaded_file = st.file_uploader("Upload Reference Photo (Face)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Process reference only once
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    ref_img = cv2.imdecode(file_bytes, 1)
    
    # Run MediaPipe on Reference
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5
    ) as face_mesh:
        rgb_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_ref)
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            
            # Update Global Reference Data safely
            with lock:
                ref_data["nose"] = (lm[1].x, lm[1].y)
                # Eye distance (Index 33 to 263)
                l_eye = (lm[33].x, lm[33].y)
                r_eye = (lm[263].x, lm[263].y)
                ref_data["eye_dist"] = calculate_distance(l_eye, r_eye)
                ref_data["image"] = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB) # Save for comparison
                
            st.success("✅ Reference Loaded! Scroll down to start.")
            st.image(ref_data["image"], caption="Reference (Goal)", width=200)
        else:
            st.error("❌ No face found in reference.")

# --- STEP 2: VIDEO PROCESSOR CLASS ---
class AlignmentProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.frame_count = 0
        self.last_instructions = []
        self.last_frame = None # Store the latest frame for capture

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Mirror immediately
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        # Keep a copy of the clean frame (or processed frame) for capturing
        # We'll save the processed one so they see the overlay in the capture? 
        # Usually better to save the clean one, but for demo let's save processed.
        
        # Access Global Reference Data
        with lock:
            r_nose = ref_data["nose"]
            r_dist = ref_data["eye_dist"]

        # If reference exists, run logic
        if r_nose is not None:
            self.frame_count += 1
            
            # Run AI every 3rd frame (balance speed/smoothness)
            if self.frame_count % 3 == 0:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_img)
                
                self.last_instructions = [] # Reset
                
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    
                    # Current Metrics
                    c_nose = (lm[1].x, lm[1].y)
                    c_l = (lm[33].x, lm[33].y)
                    c_r = (lm[263].x, lm[263].y)
                    c_dist = calculate_distance(c_l, c_r)
                    
                    # 1. Position Logic
                    thr_pos = 0.05
                    if c_nose[0] < r_nose[0] - thr_pos: self.last_instructions.append("MOVE RIGHT >>")
                    elif c_nose[0] > r_nose[0] + thr_pos: self.last_instructions.append("<< MOVE LEFT")
                    
                    if c_nose[1] < r_nose[1] - thr_pos: self.last_instructions.append("MOVE DOWN v")
                    elif c_nose[1] > r_nose[1] + thr_pos: self.last_instructions.append("MOVE UP ^")
                    
                    # 2. Depth Logic
                    thr_depth = 0.02
                    if c_dist > r_dist + thr_depth: self.last_instructions.append("MOVE BACK (-)")
                    elif c_dist < r_dist - thr_depth: self.last_instructions.append("MOVE CLOSER (+)")

            # Draw Instructions (Persistent)
            if not self.last_instructions:
                cv2.putText(img, "PERFECT SHOT!", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                # Draw green box
                cv2.rectangle(img, (20, 20), (w-20, h-20), (0, 255, 0), 5)
            else:
                for i, text in enumerate(self.last_instructions):
                    cv2.putText(img, text, (30, 80 + (i*50)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # Draw Ghost Target
            t_x, t_y = int(r_nose[0] * w), int(r_nose[1] * h)
            cv2.circle(img, (t_x, t_y), 8, (255, 255, 0), 2)
            
        # Update latest frame
        self.last_frame = img
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- STEP 3: STREAMER & UI ---
col1, col2 = st.columns([2, 1])

with col1:
    st.write("### 🎥 Live Guide")
    # Define RTC Config
    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    
    # We create the streamer context
    ctx = webrtc_streamer(
        key="alignment-stream",
        video_processor_factory=AlignmentProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": {"width": 480}, "audio": False}
    )

with col2:
    st.write("### Controls")
    # The Capture Button
    if st.button("📸 Take Photo"):
        if ctx.video_processor:
            # Grab the latest frame from the processor
            if ctx.video_processor.last_frame is not None:
                st.session_state['captured_img'] = ctx.video_processor.last_frame
                st.success("Photo Captured!")
            else:
                st.warning("Camera not ready yet.")

# --- STEP 4: RESULT COMPARISON ---
if 'captured_img' in st.session_state and ref_data["image"] is not None:
    st.write("---")
    st.subheader("📊 Comparison Result")
    
    # Prepare images for display
    # Captured image is BGR (from OpenCV), convert to RGB
    cap_rgb = cv2.cvtColor(st.session_state['captured_img'], cv2.COLOR_BGR2RGB)
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.image(ref_data["image"], caption="Original Reference", use_container_width=True)
    with res_col2:
        st.image(cap_rgb, caption="Your New Aligned Photo", use_container_width=True)
