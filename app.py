import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import av
import math

# --- PAGE CONFIG ---
st.set_page_config(page_title="ImageAssist Proto", page_icon="📸", layout="centered")

# --- CSS: LAYOUT & BUTTON FIXES ---
st.markdown("""
    <style>
    /* 1. FORCE TOP ALIGNMENT (Remove huge white space at top) */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }

    /* 2. RESERVE VIDEO SPACE (Prevents button jumping) */
    /* This forces the camera box to be tall even before the video starts. */
    div[data-testid="stWebcStreamer"] {
        min-height: 520px; 
        background-color: #f0f2f6; /* Light gray placeholder background */
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    /* 3. HIDE NATIVE STOP BUTTON */
    div[data-testid="stWebcStreamer"] button[kind="secondary"] {
        display: none !important;
    }
    
    /* 4. STYLE CUSTOM 'CAPTURE' BUTTON */
    div.stButton > button:first-child[kind="primary"] {
        background-color: #28a745;
        color: white;
        font-weight: bold;
        padding-top: 12px;
        padding-bottom: 12px;
        font-size: 18px;
        margin-top: 10px; /* spacing from video */
