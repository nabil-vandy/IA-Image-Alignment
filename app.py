import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import av
import math

# --- PAGE CONFIG ---
st.set_page_config(page_title="ImageAssist Proto", page_icon="📸", layout="centered")

# --- CSS FOR GREEN BUTTON ---
st.markdown("""
    <style>
    div.stButton > button:first-child[kind="primary"] {
        background-color: #28a745;
        border-color: #28a745;
        color: white;
    }
    div.stButton > button:first-child[kind="primary"]:hover {
        background-color: #218838;
        border-color: #1e7e34;
    }
    </style>
""", unsafe_allow_html=True)

# --- 1. SESSION STATE SETUP ---
if 'ref_data' not in st.session_state:
    st.session_state['ref_data'] = {"nose": None, "eye_dist": None, "raw": None, "mesh": None}
if 'capture_done' not in st.session_state:
    st.session_state['capture_done'] = False
if 'final_captures' not in st.session_state:
    st.session_state['final_captures'] = {"clean": None, "overlay": None}

# --- 2. GLOBAL SYNC ---
GLOBAL_REF = st.session_state['ref_data']

# --- HELPER FUNCTIONS ---
def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def reset_app():
    st.session_state['ref_data'] = {"nose": None, "eye_dist": None, "raw": None, "mesh": None}
    st.session_state['capture_done'] = False
    st.session_state['final_captures'] = {"clean": None, "overlay": None}

def center_crop_and_resize(img, target_w, target_h):
    """
    Crops the center of the image to match target aspect ratio,
    then resizes to target dimensions. fixing distortion.
    """
    if img is None: return None
    
    h_orig, w_orig = img.shape[:2]
    
    # 1. Calculate Aspect Ratios
    target_aspect = target_w / target_h
    orig_aspect = w_orig / h_orig
    
    # 2. Determine Crop Box
    if orig_aspect > target_aspect:
        # Image is too wide: Crop sides
        new_w = int(h_orig * target_aspect)
        offset = (w_orig - new_w) // 2
        crop_img = img[:, offset:offset+new_w]
    else:
        # Image is too tall: Crop top/bottom
        new_h = int(w_orig / target_aspect)
        offset = (h_orig - new_h) // 2
        crop_img = img[offset:offset+new_h, :]
        
    # 3. Resize to exact target dims
    return cv2.resize(crop_img, (target_w, target_h))

def create_collage(img1, img2, img3, img4):
    """
    Stitches 4 images into a 2x2 Grid.
    Uses the resolution of the Captured Image (img4) as the master size.
    """
    # Use the captured image dimensions as the target (Full Res)
    h_target, w_target = img4.shape[:2]
    
    # Crop and Resize Reference images to match the Capture (No distortion)
    i1 = center_crop_and_resize(img1, w_target, h_target) # Ref Raw
    i2 = center_crop_and_resize(img2, w_target, h_target) # Ref Mesh
    i3 = center_crop_and_resize(img3, w_target, h_target) # Cap Overlay
    i4 = img4 # Cap Clean (Already correct size)

    # Add Labels
    def add_label(img, text):
        # Add a small black bar at the top for legibility
        overlay = img.copy()
        cv2.rectangle(overlay, (0,0), (w_target, 50), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        cv2.putText(img, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img

    i1 = add_label(i1, "1. Reference")
    i2 = add_label(i2, "2. AI Analysis")
    i3 = add_label(i3, "3. Guidance")
    i4 = add_label(i4, "4. Result")

    # Stitch
    top_row = np.hstack([i1, i2])
    bot_row = np.hstack([i3, i4])
    grid = np.vstack([top_row, bot_row])
    
    return grid

# --- 3. PROCESSOR CLASS ---
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
        self.processed_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        # Save Clean Frame (Full Resolution)
        self.clean_frame = img.copy()
        
        # Work on copy for display
        draw_img = img.copy()

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
                    
                    thr_pos = 0.05
                    if c_nose[0] < r_nose[0] - thr_pos: self.last_instructions.append("MOVE RIGHT >>")
                    elif c_nose[0] > r_nose[0] + thr_pos: self.last_instructions.append("<< MOVE LEFT")
                    
                    if c_nose[1] < r_nose[1] - thr_pos: self.last_instructions.append("MOVE DOWN v")
                    elif c_nose[1] > r_nose[1] + thr_pos: self.last_instructions.append("MOVE UP ^")
                    
                    thr_depth = 0.02
                    if c_dist > r_dist + thr_depth: self.last_instructions.append("MOVE BACK (-)")
                    elif c_dist < r_dist - thr_depth: self.last_instructions.append("MOVE CLOSER (+)")

            # --- OVERLAYS ---
            # 1. Target Circle (Yellow)
            t_x, t_y = int(r_nose[0] * w), int(r_nose[1] * h)
            cv2.circle(draw_img, (t_x, t_y), 10, (0, 255, 255), 2) 
            cv2.circle(draw_img, (t_x, t_y), 2, (0, 255, 255), -1)
            
            # Removed "NOSE HERE" text as requested

            # 2. Instructions / Green Box
            if not self.last_instructions:
                cv2.rectangle(draw_img, (20, 20), (w-20, h-20), (0, 255, 0), 4)
                cv2.putText(draw_img, "PERFECT SHOT!", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                for i, text in enumerate(self.last_instructions):
                    cv2.putText(draw_img, text, (40, 60 + (i*40)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        self.processed_frame = draw_img

        return av.VideoFrame.from_ndarray(draw_img, format="bgr24")

# --- APP START ---
st.title("ImageAssist Mobile Demo")

with st.expander("ℹ️ About & Instructions", expanded=False):
    st.write("1. Upload a reference.")
    st.write("2. Align nose to yellow target.")
    st.write("3. Tap Capture when GREEN.")

# --- STEP 1: UPLOAD PHASE ---
if st.session_state['ref_data']['raw'] is None:
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
                    
                    annotated_image = ref_img.copy()
                    
                    # FORCE BLUE COLOR for Mesh Connections
                    # MediaPipe uses (R,G,B) for DrawingSpec, but OpenCV draws in BGR
                    # We define a custom drawing spec to ensure it's visible Blue
                    connection_spec = mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=1) # Cyan/Blueish in BGR
                    landmark_spec = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)   # Yellow dots in BGR
                    
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=results.multi_face_landmarks[0],
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=landmark_spec,
                        connection_drawing_spec=connection_spec
                    )
                    
                    # Save Data
                    st.session_state['ref_data']["nose"] = (lm[1].x, lm[1].y)
                    l_eye = (lm[33].x, lm[33].y)
                    r_eye = (lm[263].x, lm[263].y)
                    st.session_state['ref_data']["eye_dist"] = calculate_distance(l_eye, r_eye)
                    st.session_state['ref_data']["raw"] = ref_img
                    st.session_state['ref_data']["mesh"] = annotated_image
                    
                    st.rerun()
                else:
                    st.error("❌ No face found.")

# --- STEP 2: LIVE ALIGNMENT PHASE ---
elif not st.session_state['capture_done']:
    st.header("Step 2: Alignment Guide")
    st.caption("Align your nose with the **Yellow Circle**.")

    # High Resolution Constraints (1280x720) for clearer output
    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    ctx = webrtc_streamer(
        key="alignment-stream",
        video_processor_factory=AlignmentProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": {"width": 1280, "height": 720}, "audio": False}
    )

    if st.button("📸 TAP HERE TO CAPTURE", type="primary", use_container_width=True):
        if ctx.state.playing and ctx.video_processor:
            clean = ctx.video_processor.clean_frame
            overlay = ctx.video_processor.processed_frame
            
            if clean is not None and overlay is not None:
                st.session_state['final_captures']['clean'] = clean
                st.session_state['final_captures']['overlay'] = overlay
                st.session_state['capture_done'] = True
                st.rerun()
        else:
            st.warning("⚠️ Waiting for camera stream...")

# --- STEP 3: RESULT PHASE ---
else:
    st.header("Step 3: Process Report")
    st.button("🔄 Start Over", on_click=reset_app, use_container_width=True)

    if st.session_state['ref_data']['raw'] is not None and st.session_state['final_captures']['clean'] is not None:
        
        # Retrieve Images
        img1 = st.session_state['ref_data']['raw']
        img2 = st.session_state['ref_data']['mesh']
        img3 = st.session_state['final_captures']['overlay']
        img4 = st.session_state['final_captures']['clean']

        # Generate Collage with Center-Crop Aspect Ratio Fix
        collage = create_collage(img1, img2, img3, img4)
        
        # Display
        collage_rgb = cv2.cvtColor(collage, cv2.COLOR_BGR2RGB)
        st.image(collage_rgb, caption="Standardization Report", use_container_width=True)
        st.success("✅ Audit Trail Generated")
