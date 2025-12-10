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
# This allows the background thread to see the reference data
GLOBAL_REF = st.session_state['ref_data']

# --- HELPER FUNCTIONS ---
def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def reset_app():
    st.session_state['ref_data'] = {"nose": None, "eye_dist": None, "image": None}
    st.session_state['capture_done'] = False
    st.session_state['final_image'] = None

# --- 3. PROCESSOR CLASS (DEFINED GLOBALLY TO PREVENT CRASHES) ---
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
        self.clean_frame = None 

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        self.clean_frame = img.copy()

        # Access Global Data
        r_nose = GLOBAL_REF["nose"]
        r_dist = GLOBAL_REF["eye_dist"]

        if r_nose is not None:
            self.frame_count += 1
            if self.frame_count % 3 == 0:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_img)
                self.last_instructions = []
                
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    c_nose = (lm[1].x, lm[1].y)
                    c_l = (lm[33].x, lm[33].y)
                    c_r = (lm[263].x, lm[263].y)
                    c_dist = calculate_distance(c_l, c_r)
                    
                    thr_pos = 0.05
                    if c_nose[0] < r_nose[0] - thr_pos: self.last_instructions.append("MOVE RIGHT >>")
                    elif c_nose[0] > r_nose[0] + thr_pos: self.last_instructions.append("<< MOVE LEFT")
                    
                    if c_nose[1] < r_nose[1] - thr_pos: self.last_instructions.append("MOVE DOWN v")
                    elif c_nose[1] > r_nose[1] + thr_pos: self.last_instructions.append("MOVE UP ^")
                    
                    thr_depth = 0.02
                    if c_dist > r_dist + thr_depth: self.last_instructions.append("MOVE BACK (-)")
                    elif c_dist < r_dist - thr_depth: self.last_instructions.append("MOVE CLOSER (+)")

            # --- DRAWING OVERLAYS ---
            # 1. Target Circle
            t_x, t_y = int(r_nose[0] * w), int(r_nose[1] * h)
            cv2.circle(img, (t_x, t_y), 10, (0, 255, 255), 2) 
            cv2.circle(img, (t_x, t_
