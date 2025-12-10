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

# --- MEDIA PIPE SETUP ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# --- STEP 1: UPLOAD REFERENCE ---
uploaded_file = st.file_uploader("Upload Reference Photo (Face)", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Convert file to opencv image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    ref_img = cv2.imdecode(file_bytes, 1)
    
    # Process Reference Image ONCE
    h, w, _ = ref_img.shape
    ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(ref_rgb)
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Store Reference Data in Session State
        # Nose Tip (Index 1)
        st.session_state['ref_nose_pos'] = (landmarks[1].x, landmarks[1].y) # Normalized 0-1
        
        # Eye Distance (Index 33 to 263)
        left_eye = (landmarks[33].x, landmarks[33].y)
        right_eye = (landmarks[263].x, landmarks[263].y)
        st.session_state['ref_eye_dist'] = calculate_distance(left_eye, right_eye)
        
        st.success("✅ Reference Processed! Scroll down to start camera.")
        st.image(ref_img, caption="Reference Photo", width=300)
    else:
        st.error("❌ No face detected in reference photo. Try another.")

# --- STEP 2: LIVE WEBRTC PROCESSOR ---
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Mirror the image for user comfort
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(rgb_img)
        
        if results.multi_face_landmarks and st.session_state['ref_nose_pos'] is not None:
            for face_landmarks in results.multi_face_landmarks:
                lm = face_landmarks.landmark
                
                # Current Metrics
                curr_nose = (lm[1].x, lm[1].y)
                curr_left_eye = (lm[33].x, lm[33].y)
                curr_right_eye = (lm[263].x, lm[263].y)
                curr_dist = calculate_distance(curr_left_eye, curr_right_eye)
                
                # Retrieve Reference Metrics
                ref_nose = st.session_state['ref_nose_pos']
                ref_dist = st.session_state['ref_eye_dist']
                
                instructions = []
                
                # Logic: Compare Normalized Coordinates (0.0 - 1.0)
                # 1. Position (X/Y)
                threshold_pos = 0.05 # 5% tolerance
                if curr_nose[0] < ref_nose[0] - threshold_pos:
                    instructions.append("MOVE RIGHT >>")
                elif curr_nose[0] > ref_nose[0] + threshold_pos:
                    instructions.append("<< MOVE LEFT")
                
                if curr_nose[1] < ref_nose[1] - threshold_pos:
                    instructions.append("MOVE DOWN v")
                elif curr_nose[1] > ref_nose[1] + threshold_pos:
                    instructions.append("MOVE UP ^")
                    
                # 2. Depth (Z) - Using Eye Distance
                # If current eyes are wider apart (larger dist) -> Camera is too CLOSE
                threshold_depth = 0.02
                if curr_dist > ref_dist + threshold_depth:
                    instructions.append("MOVE BACK (-)")
                elif curr_dist < ref_dist - threshold_depth:
                    instructions.append("MOVE CLOSER (+)")
                
                # Draw Visuals
                color = (0, 0, 255) if instructions else (0, 255, 0)
                
                if not instructions:
                    cv2.putText(img, "PERFECT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                else:
                    for i, text in enumerate(instructions):
                        cv2.putText(img, text, (50, 100 + (i*60)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                # Draw Ghost Target (Reference Nose Position)
                target_x = int(ref_nose[0] * w)
                target_y = int(ref_nose[1] * h)
                cv2.circle(img, (target_x, target_y), 10, (255, 255, 0), 2)
        
        # Return the processed frame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- START STREAM ---
if st.session_state['ref_nose_pos'] is not None:
    st.write("### 🎥 Live Alignment Guide")
    
    # STUN servers help the webcam connect over the internet
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    webrtc_streamer(
        key="alignment", 
        video_processor_factory=VideoProcessor,  # Updated command
        rtc_configuration=rtc_configuration
    )
