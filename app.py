import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.exposure import is_low_contrast
from PIL import Image
import tempfile
import threading
import time

def initialize_face_recognition():
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    known_face_encodings = []
    known_face_names = []

    path = "known_faces"
    if os.path.exists(path):
        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            name = os.path.splitext(filename)[0]
            img = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(img)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(name)
    
    return face_cascade, known_face_encodings, known_face_names

def detect_faces(img):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)
    return face_rects

def recognize_faces(img, face_rects, known_face_encodings, known_face_names):
    face_img = img.copy()
    for (x, y, w, h) in face_rects:
        roi = face_img[y:y+h, x:x+w]
        rgb_face = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_face)
        name = "Unknown"
        if encodings:
            matches = face_recognition.compare_faces(known_face_encodings, encodings[0])
            face_distances = face_recognition.face_distance(known_face_encodings, encodings[0])

            if True in matches:
                best_match_index = np.argmin(face_distances)
                name = known_face_names[best_match_index]
        
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(face_img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return face_img

def is_blur(img):
    threshold=150
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, variance

def classify_exposure(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    total_pixels = gray.size
    dark_pixels = np.sum(hist[:50])
    bright_pixels = np.sum(hist[205:])

    dark_ratio = dark_pixels / total_pixels
    bright_ratio = bright_pixels / total_pixels

    if dark_ratio > 0.3:
        return "Underexposed"
    elif bright_ratio > 0.3:
        return "Overexposed"
    else:
        return "Well-exposed"

def detect_noise(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    diff = cv2.absdiff(gray, blurred)

    mean, std = np.mean(diff), np.std(diff)

    if mean > 10 or std > 20:
        return True
    return False


class DrawingState:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.drawing = False
        self.start_x = -1
        self.start_y = -1
        self.img = None
        self.text_to_write = ""


draw_state = DrawingState()

def draw_rectangle(event, x, y, flags, param):
    global draw_state

    if event == cv2.EVENT_LBUTTONDOWN:
        draw_state.drawing = True
        draw_state.start_x, draw_state.start_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if draw_state.drawing and draw_state.img is not None:
            img_copy = draw_state.img.copy()
            cv2.rectangle(img_copy, (draw_state.start_x, draw_state.start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow("image", img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        draw_state.drawing = False
        if draw_state.img is not None:
            cv2.rectangle(draw_state.img, (draw_state.start_x, draw_state.start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow("image", draw_state.img)

def draw_circle(event, x, y, flags, param):
    global draw_state

    if event == cv2.EVENT_LBUTTONDOWN:
        draw_state.drawing = True
        draw_state.start_x, draw_state.start_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if draw_state.drawing and draw_state.img is not None:
            img_copy = draw_state.img.copy()
            center_x = (draw_state.start_x + x) // 2
            center_y = (draw_state.start_y + y) // 2
            radius = int(0.5 * np.sqrt((x - draw_state.start_x)**2 + (y - draw_state.start_y)**2))
            cv2.circle(img_copy, (center_x, center_y), radius, (0, 0, 255), 2)
            cv2.imshow("image", img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        draw_state.drawing = False
        if draw_state.img is not None:
            center_x = (draw_state.start_x + x) // 2
            center_y = (draw_state.start_y + y) // 2
            radius = int(0.5 * np.sqrt((x - draw_state.start_x)**2 + (y - draw_state.start_y)**2))
            cv2.circle(draw_state.img, (center_x, center_y), radius, (0, 0, 255), 2)
            cv2.imshow("image", draw_state.img)

def draw_line(event, x, y, flags, param):
    global draw_state

    if event == cv2.EVENT_LBUTTONDOWN:
        draw_state.drawing = True
        draw_state.start_x, draw_state.start_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if draw_state.drawing and draw_state.img is not None:
            img_copy = draw_state.img.copy()
            cv2.line(img_copy, (draw_state.start_x, draw_state.start_y), (x, y), (255, 0, 0), 2)
            cv2.imshow("image", img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        draw_state.drawing = False
        if draw_state.img is not None:
            cv2.line(draw_state.img, (draw_state.start_x, draw_state.start_y), (x, y), (255, 0, 0), 2)
            cv2.imshow("image", draw_state.img)

def write_text(event, x, y, flags, param):
    global draw_state
    if event == cv2.EVENT_LBUTTONDOWN and draw_state.img is not None:

        print(f"Click detected at position ({x}, {y})")
        print("Enter text (press Enter to confirm, ESC to cancel):")

        import sys
        if sys.stdin.isatty():
            text = input("Enter text: ")
            cv2.putText(draw_state.img, text, (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("image", draw_state.img)
            print(f"Text '{text}' added at position ({x}, {y})")


def safe_temp_file_cleanup(file_path):
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        st.warning(f"Could not clean up temporary file: {e}")

def create_temp_file_from_upload(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.read())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error creating temporary file: {e}")
        return None

# Streamlit GUI
st.set_page_config(page_title="CSCI435 Project", layout="wide")

st.title("CSCI435 Project - Computer Vision Application")

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose the app mode",
    ["Face Recognition (Live Camera)", "Face Recognition (Upload)", "Image Quality Assessment", "Image Drawing Tool"]
)

if app_mode == "Face Recognition (Live Camera)":
    st.header("👤 Face Recognition - Live Camera")
    
    if st.button("Start Live Face Recognition"):
        st.info("Starting camera... Press ESC in the OpenCV window to stop")
        face_cascade, known_face_encodings, known_face_names = initialize_face_recognition()
        cam = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            face_rects = detect_faces(frame)
            result = recognize_faces(frame, face_rects, known_face_encodings, known_face_names)
            cv2.imshow("Face Recognition", result)

            if cv2.waitKey(1) & 0xFF == 27:
                break
        cam.release()
        cv2.destroyAllWindows()
        st.success("Camera stopped!")
elif app_mode == "Face Recognition (Upload)":
    st.header("👤 Face Recognition - Upload Image")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        tmp_path = create_temp_file_from_upload(uploaded_file)
        
        if tmp_path:
            face_cascade, known_face_encodings, known_face_names = initialize_face_recognition()
            frame = cv2.imread(tmp_path)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(uploaded_file, use_container_width=True)
            
            with col2:
                st.subheader("Face Recognition Result")
                face_rects = detect_faces(frame)
                result = recognize_faces(frame, face_rects, known_face_encodings, known_face_names)
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                st.image(result_rgb, use_container_width=True)
elif app_mode == "Image Quality Assessment":
    st.header("🔍 Image Quality Assessment")
    
    assessment_type = st.selectbox(
        "Choose assessment type",
        ["Upload Image", "Process IQA Folder - Blur", "Process IQA Folder - Exposure", 
         "Process IQA Folder - Contrast", "Process IQA Folder - Noise"]
    )
    
    if assessment_type == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image for quality assessment...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            tmp_path = create_temp_file_from_upload(uploaded_file)
            if tmp_path:
                img = cv2.imread(tmp_path)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(uploaded_file, use_container_width=True)
                
                with col2:
                    st.subheader("Quality Assessment Results")
                    blurred, variance = is_blur(img)
                    blur_text = "Blurry" if blurred else "Not Blurry"
                    st.metric("Blur Detection", blur_text, f"Blur: {variance:.2f}")
                    
                    exposure = classify_exposure(img)
                    st.metric("Exposure", exposure)
                    
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if is_low_contrast(gray, 0.35):
                        contrast_text = "Low Contrast"
                    elif not is_low_contrast(gray, 0.85):
                        contrast_text = "High Contrast"
                    else:
                        contrast_text = "Normal Contrast"
                    st.metric("Contrast", contrast_text)
                    
                    noise_detected = detect_noise(img)
                    noise_text = "Noisy" if noise_detected else "Not Noisy"
                    st.metric("Noise Detection", noise_text)
    
    elif assessment_type.startswith("Process IQA Folder"):
        folder_type = assessment_type.split(" - ")[1].lower()
        folder_path = f"IQA/detect_{folder_type}"
        
        if st.button(f"Process {folder_type.title()} Detection"):
            st.info(f"Processing {folder_path}... Check OpenCV windows. Press any key to continue between images.")
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                if folder_type == "blur":
                    blurred, variance = is_blur(img)
                    text = "Blurry" if blurred else "Not Blurry"
                    cv2.putText(img, f"{text}: {variance:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Image Quality Assessment - Blur", img)
                
                elif folder_type == "exposure":
                    exposure = classify_exposure(img)
                    cv2.putText(img, f"{exposure}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Image Quality Assessment - Exposure", img)
                
                elif folder_type == "contrast":
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if is_low_contrast(gray, 0.35):
                        text = "Low Contrast"
                    elif not is_low_contrast(gray, 0.85):
                        text = "High Contrast"
                    else:
                        text = "Normal Contrast"
                    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Image Quality Assessment - Contrast", img)
                
                elif folder_type == "noise":
                    noise_detected = detect_noise(img)
                    text = "Noisy" if noise_detected else "Not Noisy"
                    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Image Quality Assessment - Noise", img)
                
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            st.success("Folder processing complete!")
elif app_mode == "Image Drawing Tool":
    st.header("Image Annotation Tool")
    use_default = st.checkbox("Use default image (known_faces/elon musk.jpg)", value=True)
    img_loaded = False
    if use_default:
        default_path = "known_faces/elon musk.jpg"
        draw_state.img = cv2.imread(default_path)
        img_loaded = True
    else:
        uploaded_file = st.file_uploader("Choose an image to draw on...", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            tmp_path = create_temp_file_from_upload(uploaded_file)
            draw_state.img = cv2.imread(tmp_path)
            img_loaded = True

    if img_loaded and draw_state.img is not None:
        shape_options = {1: "Circle", 2: "Rectangle", 3: "Line", 4: "Text"}
        shape = st.selectbox("Choose shape to draw:", list(shape_options.keys()), format_func=lambda x: shape_options[x])

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Drawing"):
                st.info("Drawing mode started! Check the OpenCV window. Press ESC to stop.")
                if shape == 4:
                    st.success("🎯 **Text Mode Active:** Click on the image and check your terminal/console for text input prompts!")
                draw_state.drawing = False
                draw_state.start_x, draw_state.start_y = -1, -1
                
                cv2.namedWindow("image")

                if shape == 1: 
                    cv2.setMouseCallback("image", draw_circle)
                elif shape == 2: 
                    cv2.setMouseCallback("image", draw_rectangle)
                elif shape == 3: 
                    cv2.setMouseCallback("image", draw_line)
                elif shape == 4: 
                    cv2.setMouseCallback("image", write_text)
                    print("\n" + "="*50)
                    print("TEXT MODE ACTIVATED")
                    print("Click anywhere on the image to add text")
                    print("You will be prompted to enter text for each click")
                    print("Press ESC in the image window to exit")
                    print("="*50 + "\n")
                
                while True:
                    cv2.imshow("image", draw_state.img)
                    if cv2.waitKey(1) & 0xFF == 27: 
                        break
                cv2.destroyAllWindows()
                st.success("Drawing completed!")