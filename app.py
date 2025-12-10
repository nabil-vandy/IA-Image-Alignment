import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import av
import math

# --- PAGE CONFIG ---
st.set_page_config(page_title="ImageAssist AI Alignment", page_icon="📸")

# --- 1. SESSION STATE SETUP (Persists across reloads) ---
if 'ref_data' not in st.session_state:
    st.session_state['ref_data'] = {"nose": None, "eye_dist": None, "image": None}
if 'capture_done' not in st.session_state:
    st.session_state['capture_done'] = False
if 'final_image' not in st.session_state:
    st.session_state['final_image'] = None

# --- 2. GLOBAL SYNC (The Fix for Video Threading) ---
# We create a global reference that the video thread can ALWAYS see.
# We populate it from Session State on every script re-run.
GLOBAL_REF = st.session_state['ref_data']

# --- HELPER FUNCTIONS ---
def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def reset_app():
    """Clears all state to truly start over"""
    st.session_state['ref_data'] = {"nose": None, "eye_dist": None, "image": None}
    st.session_state['capture_done'] = False
    st.session_state['final_image'] = None

# --- UI HEADER ---
st.title("ImageAssist: Smart Alignment Proto")

# --- STEP 3: LOGIC CONTROLLER ---

# SCENARIO A: START / UPLOAD PHASE
if st.session_state['ref_data']['image'] is None:
    uploaded_file = st.file_uploader("1. Upload Reference Photo", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        # Process Reference ONCE
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        ref_img = cv2.imdecode(file_bytes, 1)
        
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
                
                # Save to Session State (Persist)
                st.session_state['ref_data']["nose"] = (lm[1].x, lm[1].y)
                l_eye = (lm[33].x, lm[33].y)
                r_eye = (lm[263].x, lm[263].y)
                st.session_state['ref_data']["eye_dist"] = calculate_distance(l_eye, r_eye)
                st.session_state['ref_data']["image"] = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                
                st.rerun() # Reload to move to next step
            else:
                st.error("❌ No face found. Please use a clearer photo.")

# SCENARIO B: LIVE ALIGNMENT PHASE
elif not st.session_state['capture_done']:
    st.write("### 2. Live Alignment Guide")

    # Define the Processor Class
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
            img = cv2.flip(img, 1) # Mirror
            h, w, _ = img.shape
            
            # Save clean copy for capture
            self.clean_frame = img.copy()

            # READ FROM GLOBAL (Safe Access)
            r_nose = GLOBAL_REF["nose"]
            r_dist = GLOBAL_REF["eye_dist"]

            if r_nose is not None:
                self.frame_count += 1
                
                # Run AI every 3rd frame
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
                        
                        # Position Logic
                        thr_pos = 0.05
                        if c_nose[0] < r_nose[0] - thr_pos: self.last_instructions.append("MOVE RIGHT >>")
                        elif c_nose[0] > r_nose[0] + thr_pos: self.last_instructions.append("<< MOVE LEFT")
                        
                        if c_nose[1] < r_nose[1] - thr_pos: self.last_instructions.append("MOVE DOWN v")
                        elif c_nose[1] > r_nose[1] + thr_pos: self.last_instructions.append("MOVE UP ^")
                        
                        # Depth Logic
                        thr_depth = 0.02
                        if c_dist > r_dist + thr_depth: self.last_instructions.append("MOVE BACK (-)")
                        elif c_dist < r_dist - thr_depth: self.last_instructions.append("MOVE CLOSER (+)")

                # Draw Instructions
                if not self.last_instructions:
                    cv2.rectangle(img, (20, 20), (w-20, h-20), (0, 255, 0), 4)
                    cv2.putText(img, "PERFECT SHOT!", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    for i, text in enumerate(self.last_instructions):
                        cv2.putText(img, text, (40, 60 + (i*40)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Draw Ghost Target
                t_x, t_y = int(r_nose[0] * w), int(r_nose[1] * h)
                cv2.circle(img, (t_x, t_y), 6, (0, 255, 255), 2)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # UI Layout
    col1, col2 = st.columns([3, 1])
    with col1:
        rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        ctx = webrtc_streamer(
            key="alignment-stream",
            video_processor_factory=AlignmentProcessor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False}
        )
    with col2:
        st.info("Align until the box turns green.")
        if st.button("📸 Take Photo", type="primary"):
            if ctx.video_processor and ctx.video_processor.clean_frame is not None:
                st.session_state['final_image'] = ctx.video_processor.clean_frame
                st.session_state['capture_done'] = True
                st.rerun()

# SCENARIO C: RESULT PHASE
else:
    st.write("### 3. Comparison Result")
    
    # Reset Button with Callback
    st.button("🔄 Start Over", on_click=reset_app)

    # Display Images
    if st.session_state['ref_data']['image'] is not None and st.session_state['final_image'] is not None:
        final_rgb = cv2.cvtColor(st.session_state['final_image'], cv2.COLOR_BGR2RGB)
        
        c1, c2 = st.columns(2)
        with c1:
            st.image(st.session_state['ref_data']['image'], caption="Original Reference", use_container_width=True)
        with c2:
            st.image(final_rgb, caption="Your Aligned Photo", use_container_width=True)
