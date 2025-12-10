import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import av
import math

# Page Config
st.set_page_config(page_title="ImageAssist AI Alignment", page_icon="📸")

st.title("ImageAssist: Smart Alignment Proto")
st.markdown("""
**Instructions:**
1. Upload a "Reference Photo" (the "Gold Standard").
2. Allow camera access.
3. The AI will guide you to match the reference angle and distance.
""")

# --- GLOBAL VARIABLES FOR REFERENCE DATA ---
if 'ref_landmarks' not in st.session_state:
    st.session_state['ref_landmarks'] = None
if 'ref_eye_dist' not in st.session_state:
    st.session_state['ref_eye_dist'] = None
if 'ref_nose_pos' not in st.session_state:
    st.session_state['ref_nose_pos'] = None

# --- HELPER FUNCTION ---
def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# --- STEP 1: UPLOAD REFERENCE ---
uploaded_file = st.file_uploader("Upload Reference Photo (Face)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Use a temporary FaceMesh just for the reference photo
    with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh_ref:
        
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        ref_img = cv2.imdecode(file_bytes, 1)
        
        h, w, _ = ref_img.shape
        ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        results = face_mesh_ref.process(ref_rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Store Reference Data
            st.session_state['ref_nose_pos'] = (landmarks[1].x, landmarks[1].y) 
            left_eye = (landmarks[33].x, landmarks[33].y)
            right_eye = (landmarks[263].x, landmarks[263].y)
            st.session_state['ref_eye_dist'] = calculate_distance(left_eye, right_eye)
            
            st.success("✅ Reference Processed! Scroll down to start camera.")
            st.image(ref_img, caption="Reference Photo", width=300)
        else:
            st.error("❌ No face detected in reference photo. Try another.")

# --- STEP 2: LIVE WEBRTC PROCESSOR (OPTIMIZED) ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        # Initialize MediaPipe INSIDE the class for thread safety
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Variables for Frame Skipping
        self.frame_count = 0
        self.skip_rate = 5 # Process only 1 out of every 5 frames
        self.last_instructions = []
        self.last_color = (0, 255, 0)
        
        # Cache the reference data so we don't access Session State every frame
        self.ref_nose = st.session_state.get('ref_nose_pos')
        self.ref_dist = st.session_state.get('ref_eye_dist')

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Mirror image immediately
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        # --- FRAME SKIPPING LOGIC ---
        self.frame_count += 1
        
        # Only run heavy AI inference if we are on the Nth frame
        if self.frame_count % self.skip_rate == 0 and self.ref_nose is not None:
            
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_img)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    lm = face_landmarks.landmark
                    
                    # Current Metrics
                    curr_nose = (lm[1].x, lm[1].y)
                    curr_left_eye = (lm[33].x, lm[33].y)
                    curr_right_eye = (lm[263].x, lm[263].y)
                    curr_dist = calculate_distance(curr_left_eye, curr_right_eye)
                    
                    # Reset instructions for this new frame
                    self.last_instructions = []
                    
                    # 1. Position Logic (X/Y)
                    threshold_pos = 0.05 
                    if curr_nose[0] < self.ref_nose[0] - threshold_pos:
                        self.last_instructions.append("MOVE RIGHT >>")
                    elif curr_nose[0] > self.ref_nose[0] + threshold_pos:
                        self.last_instructions.append("<< MOVE LEFT")
                    
                    if curr_nose[1] < self.ref_nose[1] - threshold_pos:
                        self.last_instructions.append("MOVE DOWN v")
                    elif curr_nose[1] > self.ref_nose[1] + threshold_pos:
                        self.last_instructions.append("MOVE UP ^")
                        
                    # 2. Depth Logic (Z)
                    threshold_depth = 0.02
                    if curr_dist > self.ref_dist + threshold_depth:
                        self.last_instructions.append("MOVE BACK (-)")
                    elif curr_dist < self.ref_dist - threshold_depth:
                        self.last_instructions.append("MOVE CLOSER (+)")
                    
                    # Set Color
                    self.last_color = (0, 0, 255) if self.last_instructions else (0, 255, 0)

        # --- DRAWING (Happens on EVERY frame using cached data) ---
        if self.ref_nose is not None:
            # Draw Ghost Target
            target_x = int(self.ref_nose[0] * w)
            target_y = int(self.ref_nose[1] * h)
            cv2.circle(img, (target_x, target_y), 10, (255, 255, 0), 2)

            # Draw Instructions (Cached)
            if not self.last_instructions:
                cv2.putText(img, "PERFECT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                for i, text in enumerate(self.last_instructions):
                    cv2.putText(img, text, (50, 100 + (i*60)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- START STREAM ---
if st.session_state['ref_nose_pos'] is not None:
    st.write("### 🎥 Live Alignment Guide")
    
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="alignment", 
        video_processor_factory=VideoProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={
            "video": {"width": 480, "height": 360}, # Keep low res for speed
            "audio": False
        }
    )
