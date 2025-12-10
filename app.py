import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import av
import math

# --- PAGE CONFIG ---
st.set_page_config(page_title="ImageAssist Proto", page_icon="📸", layout="centered")

# --- 1. SESSION STATE SETUP ---
if 'ref_data' not in st.session_state:
    st.session_state['ref_data'] = {"nose": None, "eye_dist": None, "image": None}
if 'capture_done' not in st.session_state:
    st.session_state['capture_done'] = False
if 'final_image' not in st.session_state:
    st.session_state['final_image'] = None

# --- 2. GLOBAL SYNC ---
GLOBAL_REF = st.session_state['ref_data']

# --- HELPER FUNCTIONS ---
def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def reset_app():
    st.session_state['ref_data'] = {"nose": None, "eye_dist": None, "image": None}
    st.session_state['capture_done'] = False
    st.session_state['final_image'] = None

# --- APP HEADER ---
st.title("ImageAssist Mobile Demo")

# Collapsible Instructions to save mobile screen space
with st.expander("ℹ️ About & Instructions", expanded=False):
    st.write("**Goal:** Standardize clinical photos using AI guidance.")
    st.write("1. Upload a reference.")
    st.write("2. Align your camera until the box turns GREEN.")
    st.write("3. Tap 'Take Photo' to capture.")

# --- STEP 1: UPLOAD PHASE ---
if st.session_state['ref_data']['image'] is None:
    st.header("Step 1: Upload Reference")
    
    uploaded_file = st.file_uploader("Select Reference Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        ref_img = cv2.imdecode(file_bytes, 1)
        
        # Process Reference
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
                
                # Save Data
                st.session_state['ref_data']["nose"] = (lm[1].x, lm[1].y)
                l_eye = (lm[33].x, lm[33].y)
                r_eye = (lm[263].x, lm[263].y)
                st.session_state['ref_data']["eye_dist"] = calculate_distance(l_eye, r_eye)
                st.session_state['ref_data']["image"] = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
