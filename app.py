import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import av
import math

# --- PAGE CONFIG ---
st.set_page_config(page_title="ImageAssist AI Alignment", page_icon="📸")

# --- GLOBAL SHARED DATA (The Fix for "No Overlay") ---
# We use a standard global dictionary instead of st.session_state
# so the video thread can definitely see it.
REF_DATA = {
    "nose": None,
    "eye_dist": None,
    "image": None
}

# --- HELPER FUNCTIONS ---
def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# --- STEP 1: UI & SETUP ---
st.title("ImageAssist: Smart Alignment Proto")

# Initialize Session State for the Capture logic only
if 'capture_done' not in st.session_state:
    st.session_state['capture_done'] = False
if 'final_image' not in st.session_state:
    st.session_state['final_image'] = None

# --- STEP 2: REFERENCE UPLOAD ---
# Only show if we haven't captured the final shot yet
if not st.session_state['capture_done']:
    uploaded_file = st.file_uploader("1. Upload Reference Photo", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        # 1. Read Image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        ref_img = cv2.imdecode(file_bytes, 1)
        
        # 2. Process Reference (ONCE)
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
                
                # UPDATE GLOBAL VARIABLES DIRECTLY
                REF_DATA["nose"] = (lm[1].x, lm[1].y)
                l_eye = (lm[33].x, lm[33].y)
                r_eye = (lm[263].x, lm[263].y)
                REF_DATA["eye_dist"] = calculate_distance(l_eye, r_eye)
                REF_DATA["image"] = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                
                st.success("✅ Reference Loaded! Scroll down.")
            else:
                st.error("❌ No face found. Please use a clearer photo.")

# --- STEP 3: VIDEO PROCESSOR ---
class AlignmentProcessor(VideoProcessorBase):
    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.frame_count = 0
        self.last_instructions = []
        self.clean_frame = None 

    def recv(self, frame):
        # Convert to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Flip immediately (Mirror effect)
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        # Save clean copy for capture
        self.clean_frame = img.copy()

        # READ GLOBAL REFERENCE DATA
        # (This fixes the "No Overlay" issue)
        r_nose = REF_DATA["nose"]
        r_dist = REF_DATA["eye_dist"]

        if r_nose is not None:
            self.frame_count += 1
            
            # --- PERFORMANCE OPTIMIZATION ---
            # Run AI only every 3rd frame to prevent lag
            if self.frame_count % 3 == 0:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_img)
                
                self.last_instructions = []
                
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    
                    # Current Metrics
                    c_nose = (lm[1].x, lm[1].y)
                    c_l = (lm[33].x, lm[33].y)
                    c_r = (lm[263].x, lm[263].y)
                    c_dist = calculate_distance(c_l, c_r)
                    
                    # Logic: Position
                    thr_pos = 0.05
                    if c_nose[0] < r_nose[0] - thr_pos: self.last_instructions.append("MOVE RIGHT >>")
                    elif c_nose[0] > r_nose[0] + thr_pos: self.last_instructions.append("<< MOVE LEFT")
                    
                    if c_nose[1] < r_nose[1] - thr_pos: self.last_instructions.append("MOVE DOWN v")
                    elif c_nose[1] > r_nose[1] + thr_pos: self.last_instructions.append("MOVE UP ^")
                    
                    # Logic: Depth
                    thr_depth = 0.02
                    if c_dist > r_dist + thr_depth: self.last_instructions.append("MOVE BACK (-)")
                    elif c_dist < r_dist - thr_depth: self.last_instructions.append("MOVE CLOSER (+)")

            # --- DRAWING (Persistent) ---
            if not self.last_instructions:
                # Perfect Shot - Green Box
                cv2.rectangle(img, (20, 20), (w-20, h-20), (0, 255, 0), 4)
                cv2.putText(img, "PERFECT SHOT!", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # Instructions - Red Text
                for i, text in enumerate(self.last_instructions):
                    cv2.putText(img, text, (40, 60 + (i*40)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Draw Ghost Target (Yellow Circle)
            t_x, t_y = int(r_nose[0] * w), int(r_nose[1] * h)
            cv2.circle(img, (t_x, t_y), 6, (0, 255, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- STEP 4: MAIN LOGIC ---

# IF NOT CAPTURED YET -> SHOW CAMERA
if not st.session_state['capture_done']:
    if REF_DATA["nose"] is not None:
        st.write("### 2. Live Alignment Guide")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
            
            # STREAMER
            ctx = webrtc_streamer(
                key="alignment-stream",
                video_processor_factory=AlignmentProcessor,
                rtc_configuration=rtc_configuration,
                media_stream_constraints={
                    "video": {"width": 640, "height": 480}, # 640x480 (Good balance of speed vs blur)
                    "audio": False
                }
            )

        with col2:
            st.info("Align until the box turns green.")
            if st.button("📸 Take Photo", type="primary"):
                if ctx.video_processor and ctx.video_processor.clean_frame is not None:
                    # Capture the CLEAN frame
                    st.session_state['final_image'] = ctx.video_processor.clean_frame
                    st.session_state['capture_done'] = True
                    st.rerun()

# IF CAPTURED -> SHOW RESULT
else:
    st.write("### 3. Comparison Result")
    
    if st.button("🔄 Start Over"):
        st.session_state['capture_done'] = False
        st.session_state['final_image'] = None
        # Note: We keep REF_DATA so they don't have to re-upload the reference
        st
