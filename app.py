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
            cv2.circle(img, (t_x, t_y), 2, (0, 255, 255), -1)
            
            # 2. Label
            cv2.putText(img, "NOSE HERE", (t_x + 15, t_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # 3. Instructions
            if not self.last_instructions:
                cv2.rectangle(img, (20, 20), (w-20, h-20), (0, 255, 0), 4)
                cv2.putText(img, "PERFECT SHOT!", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                for i, text in enumerate(self.last_instructions):
                    cv2.putText(img, text, (40, 60 + (i*40)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- APP START ---
st.title("ImageAssist Mobile Demo")

with st.expander("ℹ️ About & Instructions", expanded=False):
    st.write("1. Upload a reference.")
    st.write("2. Align nose to yellow target.")
    st.write("3. Tap Capture when GREEN.")

# --- STEP 1: UPLOAD PHASE ---
if st.session_state['ref_data']['image'] is None:
    st.header("Step 1: Upload Reference")
    uploaded_file = st.file_uploader("Select Reference Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        with st.spinner("Analyzing..."):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            ref_img = cv2.imdecode(file_bytes, 1)
            
            mp_face_mesh = mp.solutions.face_mesh
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            
            with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5) as face_mesh:
                rgb_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_ref)
                
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    
                    # Create Annotated Image (Blue Mesh)
                    annotated_image = ref_img.copy()
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=results.multi_face_landmarks[0],
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    
                    # Save Data
                    st.session_state['ref_data']["nose"] = (lm[1].x, lm[1].y)
                    l_eye = (lm[33].x, lm[33].y)
                    r_eye = (lm[263].x, lm[263].y)
                    st.session_state['ref_data']["eye_dist"] = calculate_distance(l_eye, r_eye)
                    st.session_state['ref_data']["image"] = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    
                    st.rerun()
                else:
                    st.error("❌ No face found.")

# --- STEP 2: LIVE ALIGNMENT PHASE ---
elif not st.session_state['capture_done']:
    st.header("Step 2: Alignment Guide")
    
    # 1. Create Placeholder at the TOP
    button_placeholder = st.empty()
    st.caption("Align your nose with the **Yellow Circle**.")

    # 2. Render Video Stream
    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    ctx = webrtc_streamer(
        key="alignment-stream",
        video_processor_factory=AlignmentProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False}
    )

    # 3. Fill Placeholder (Now that ctx exists)
    with button_placeholder:
        if st.button("📸 TAP HERE TO CAPTURE", type="primary", use_container_width=True):
            if ctx.state.playing and ctx.video_processor and ctx.video_processor.clean_frame is not None:
                st.session_state['final_image'] = ctx.video_processor.clean_frame
                st.session_state['capture_done'] = True
                st.rerun()
            else:
                st.warning("⚠️ Waiting for camera stream...")

# --- STEP 3: RESULT PHASE ---
else:
    st.header("Step 3: Results")
    st.button("🔄 Start Over", on_click=reset_app, use_container_width=True)

    if st.session_state['ref_data']['image'] is not None and st.session_state['final_image'] is not None:
        final_rgb = cv2.cvtColor(st.session_state['final_image'], cv2.COLOR_BGR2RGB)
        
        st.image(st.session_state['ref_data']['image'], caption="Reference (Analyzed)", use_container_width=True)
        st.image(final_rgb, caption="Your Aligned Photo", use_container_width=True)
