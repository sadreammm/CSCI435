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
import re
import sys

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

def initialize_landmark_detection():
    """Initialize SIFT detector and load landmark database"""
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    landmark_dir = 'content/museum_images'
    
    landmark_descriptors = []
    landmark_keypoints = []
    landmark_images = []
    landmark_names = []
    
    if os.path.exists(landmark_dir):
        for img_name in os.listdir(landmark_dir):
            img_path = os.path.join(landmark_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (600, 400))
            kp, des = sift.detectAndCompute(img, None)
            if des is not None:
                landmark_keypoints.append(kp)
                landmark_descriptors.append(des)
                landmark_images.append(img)
                landmark_names.append(os.path.splitext(img_name)[0])
    
    return sift, bf, landmark_descriptors, landmark_keypoints, landmark_images, landmark_names

def detect_landmark(query_img, sift, bf, landmark_descriptors, landmark_keypoints, landmark_images, landmark_names, match_threshold=10):
    """Detect landmarks in query image"""
    if query_img is None:
        return query_img, "No landmark detected", 0
    
    # Convert to grayscale if needed
    if len(query_img.shape) == 3:
        query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    else:
        query_gray = query_img
    
    query_gray = cv2.resize(query_gray, (600, 400))
    query_kp, query_des = sift.detectAndCompute(query_gray, None)
    
    if query_des is None:
        return query_img, "No features detected", 0

    best_match_count = 0
    best_good_matches = []
    best_index = -1
    best_landmark_name = "No landmark detected"

    for i, des in enumerate(landmark_descriptors):
        matches = bf.match(des, query_des)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]

        if len(good_matches) >= match_threshold:
            src_pts = np.float32([landmark_keypoints[i][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([query_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                matches_mask = mask.ravel().tolist()
                inliers = sum(matches_mask)

                if inliers > best_match_count:
                    best_match_count = inliers
                    best_good_matches = good_matches
                    best_index = i
                    best_landmark_name = landmark_names[i]

    if best_match_count >= match_threshold:
        query_color = cv2.resize(query_img if len(query_img.shape) == 3 else cv2.cvtColor(query_img, cv2.COLOR_GRAY2BGR), (600, 400))

        matched_img = cv2.drawMatches(
            landmark_images[best_index], landmark_keypoints[best_index],
            query_color, query_kp,
            best_good_matches[:20],
            None,
            matchColor=(0, 255, 0),
            singlePointColor=None,
            matchesMask=[1]*min(20, len(best_good_matches)),
            flags=2
        )
        return matched_img, best_landmark_name, best_match_count
    else:
        return query_img, "No landmark detected", 0
    

def compute_brightness_contrast_metrics(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    metrics = {
        'mean_gray': np.mean(gray),
        'mean_rgb': np.mean(img),
        'mean_hsv_v': np.mean(hsv[:, :, 2]),
        'mean_lab_l': np.mean(lab[:, :, 0]), 
    }
    
    return metrics

def classify_day_night(img, metric="mean_hsv_v"):
    thresholds = {
        'mean_hsv_v': 100,
        'mean_lab_l': 100,
        'mean_rgb': 100,
        'mean_gray': 100
    }
    
    metrics = compute_brightness_contrast_metrics(img)
    value = metrics[metric]
    threshold = thresholds[metric]
    
    label = "Day" if value > threshold else "Night"
    confidence = abs(value - threshold) / threshold
    
    return label, value, confidence

def wrap_text(text, font, font_scale, max_width):
    words = text.split(' ')
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        text_size = cv2.getTextSize(test_line, font, font_scale, 1)[0]
        
        if text_size[0] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
                current_line = word
            else:
                lines.append(word)
    
    if current_line:
        lines.append(current_line)
    
    return lines

def draw_wrapped_text(img, text, start_pos, font, font_scale, color, thickness, max_width):
    lines = wrap_text(text, font, font_scale, max_width)
    x, y = start_pos
    line_height = cv2.getTextSize("Ay", font, font_scale, thickness)[0][1] + 10
    
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + i * line_height), font, font_scale, color, thickness)
    
    return y + len(lines) * line_height

def highlight_blur_regions(img, gray, threshold=150):
    height, width = gray.shape
    window_size = min(height, width) // 8
    if window_size < 50:
        window_size = 50
    
    overlay = img.copy() 
    for y in range(0, height - window_size, window_size // 2):
        for x in range(0, width - window_size, window_size // 2):
            window = gray[y:y+window_size, x:x+window_size]
            laplacian_var = cv2.Laplacian(window, cv2.CV_64F).var()
            if laplacian_var < threshold:
                cv2.rectangle(overlay, (x, y), (x+window_size, y+window_size), (0, 0, 255), -1)
    return cv2.addWeighted(img, 0.8, overlay, 0.2, 0)

def highlight_exposure_regions(img, exposure_type):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    overlay = img.copy()
    if exposure_type == "Underexposed":
        mask = gray < 50
        overlay[mask] = [0, 255, 255]
    elif exposure_type == "Overexposed":
        mask = gray > 205
        overlay[mask] = [0, 165, 255]
    return cv2.addWeighted(img, 0.8, overlay, 0.2, 0)

def highlight_low_contrast_regions(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    window_size = min(height, width) // 6
    if window_size < 50:
        window_size = 50
    
    overlay = img.copy()
    for y in range(0, height - window_size, window_size // 2):
        for x in range(0, width - window_size, window_size // 2):
            window = gray[y:y+window_size, x:x+window_size]
            if is_low_contrast(window, 0.3):
                cv2.rectangle(overlay, (x, y), (x+window_size, y+window_size), (255, 0, 0), -1)
    return cv2.addWeighted(img, 0.8, overlay, 0.2, 0)

def highlight_noisy_regions(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    window_size = min(height, width) // 8
    if window_size < 50:
        window_size = 50
    
    overlay = img.copy()
    for y in range(0, height - window_size, window_size // 2):
        for x in range(0, width - window_size, window_size // 2):
            window = gray[y:y+window_size, x:x+window_size]
            blurred_window = cv2.GaussianBlur(window, (5, 5), 0)
            diff = cv2.absdiff(window, blurred_window)
            
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)
            if mean_diff > 8 or std_diff > 15:
                cv2.rectangle(overlay, (x, y), (x+window_size, y+window_size), (128, 0, 128), -1)
    return cv2.addWeighted(img, 0.8, overlay, 0.2, 0)

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

def get_comprehensive_assessment(img):
    results = {}
    highlighted_images = {}

    blurred, variance = is_blur(img)
    results['blur'] = {
        'status': "Blurry" if blurred else "Not Blurry",
        'variance': variance,
        'enhancement': "Apply sharpening filter (Laplacian) or recapture with stable camera." if blurred else None
    }
    
    if blurred:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        highlighted_images['blur'] = highlight_blur_regions(img, gray)

    exposure = classify_exposure(img)
    results['exposure'] = {
        'status': exposure,
        'enhancement': None
    }
    
    if exposure == "Underexposed":
        results['exposure']['enhancement'] = "Increase brightness and adjust levels/curves to recover shadow detail."
        highlighted_images['exposure'] = highlight_exposure_regions(img, exposure)
    elif exposure == "Overexposed":
        results['exposure']['enhancement'] = "Decrease brightness and adjust levels/curves to recover highlight detail."
        highlighted_images['exposure'] = highlight_exposure_regions(img, exposure)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if is_low_contrast(gray, 0.35):
        contrast_status = "Low Contrast"
        enhancement = "Increase contrast using curves, levels, or histogram equalization."
        highlighted_images['contrast'] = highlight_low_contrast_regions(img)
    elif not is_low_contrast(gray, 0.85):
        contrast_status = "High Contrast"
        enhancement = "Decrease contrast using curves or reducing highlights/shadows."
    else:
        contrast_status = "Normal Contrast"
        enhancement = None
    
    results['contrast'] = {
        'status': contrast_status,
        'enhancement': enhancement
    }

    noise_detected = detect_noise(img)
    results['noise'] = {
        'status': "Noisy" if noise_detected else "Not Noisy",
        'enhancement': "Apply noise reduction filter (Gaussian, median, or bilateral filter)." if noise_detected else None
    }
    if noise_detected:
        highlighted_images['noise'] = highlight_noisy_regions(img)
    
    return results, highlighted_images

def process_folder_with_highlighting(folder_path, assessment_type):
    if not os.path.exists(folder_path):
        return []
    
    results = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        max_text_width = width - 20
        
        annotated_img = img.copy()
        highlighted_img = img.copy()

        enhancement_text = None
        problem = False
        status = ""
        
        if assessment_type == "blur":
            blurred, variance = is_blur(img)
            status = "Blurry" if blurred else "Not Blurry"
            cv2.putText(annotated_img, f"{status}: {variance:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if blurred:
                enhancement_text = "Enhancement: Apply sharpening filter or recapture with stable camera."
                draw_wrapped_text(annotated_img, enhancement_text, (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, max_text_width)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                highlighted_img = highlight_blur_regions(img, gray)
                problem = True
        
        elif assessment_type == "exposure":
            exposure = classify_exposure(img)
            cv2.putText(annotated_img, f"{exposure}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if exposure in ["Underexposed", "Overexposed"]:
                if exposure == "Underexposed":
                    enhancement_text = "Enhancement: Increase brightness and adjust levels/curves to recover shadow detail."
                    problem = True
                else:
                    enhancement_text = "Enhancement: Decrease brightness and adjust levels/curves to recover highlight detail."
                    problem = True
                
                draw_wrapped_text(annotated_img, enhancement_text, (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, max_text_width)
                highlighted_img = highlight_exposure_regions(img, exposure)
        
        elif assessment_type == "contrast":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if is_low_contrast(gray, 0.4):
                status = "Low Contrast"
                enhancement_text = "Enhancement: Increase contrast using curves, levels, or histogram equalization."
                highlighted_img = highlight_low_contrast_regions(img)
                problem = True
            elif not is_low_contrast(gray, 0.85):
                status = "High Contrast"
                enhancement_text = "Enhancement: Decrease contrast using curves or reducing highlights/shadows."
            else:
                status = "Normal Contrast"
                enhancement_text = None
            
            cv2.putText(annotated_img, f"Contrast: {status}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if enhancement_text:
                draw_wrapped_text(annotated_img, enhancement_text, (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, max_text_width)
        
        elif assessment_type == "noise":
            noise_detected = detect_noise(img)
            status = "Noisy" if noise_detected else "Not Noisy"
            cv2.putText(annotated_img, f"Noise: {status}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if noise_detected:
                enhancement_text = "Enhancement: Apply noise reduction filter (Gaussian, median, or bilateral filter)."
                draw_wrapped_text(annotated_img, enhancement_text, (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, max_text_width)
                highlighted_img = highlight_noisy_regions(img)
                problem = True
        
        results.append({
            'filename': filename,
            'original': cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            'annotated': cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
            'highlighted': cv2.cvtColor(highlighted_img, cv2.COLOR_BGR2RGB),
            'enhancement': enhancement_text.split(": ")[1] if enhancement_text else None,
            'problem': problem,
            'status': status
        })
    
    return results

def initialize_similarity_search():
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    similarity_dir = 'content/museum_images'
    
    landmark_descriptors = []
    landmark_keypoints = []
    landmark_images = []
    landmark_names = []
    
    if os.path.exists(similarity_dir):
        for img_name in os.listdir(similarity_dir):
            img_path = os.path.join(similarity_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (600, 400))
            kp, des = sift.detectAndCompute(img, None)
            if des is not None:
                landmark_keypoints.append(kp)
                landmark_descriptors.append(des)
                landmark_images.append(img)
                landmark_names.append(img_name)
    
    return sift, bf, landmark_descriptors, landmark_keypoints, landmark_images, landmark_names

def retrieve_top_similar_images(query_img, sift, bf, landmark_descriptors, landmark_keypoints, landmark_images, landmark_names, top_k=3, match_threshold=10):
    if len(query_img.shape) == 3:
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    
    query_img = cv2.resize(query_img, (600, 400))
    query_kp, query_des = sift.detectAndCompute(query_img, None)
    
    if query_des is None:
        return []
    
    results = []
    
    for i, des in enumerate(landmark_descriptors):
        matches = bf.match(des, query_des)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]
        
        if len(good_matches) >= match_threshold:
            src_pts = np.float32([landmark_keypoints[i][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([query_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            try:
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if mask is not None:
                    inliers = int(np.sum(mask))
                    results.append((inliers, i, good_matches))
            except:
                continue
    
    results.sort(reverse=True, key=lambda x: x[0])
    top_images = []
    
    for inliers, idx, good_matches in results[:top_k]:
        matched_img = cv2.drawMatches(
            landmark_images[idx], landmark_keypoints[idx],
            query_img, query_kp, good_matches[:20], None,
            matchColor=(0, 255, 0), flags=2
        )
        top_images.append((landmark_names[idx], matched_img, inliers))
    
    return top_images

class DrawingState:
    def __init__(self):
        self.reset()
        self.current = 1  # 1: Circle, 2: Rectangle, 3: Line, 4: Text
        self.original_img = None
        #self.text_queue = queue.Queue()
        self.text_input_active = False
    
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

        if sys.stdin.isatty():
            text = input("Enter text: ")
            cv2.putText(draw_state.img, text, (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("image", draw_state.img)
            print(f"Text '{text}' added at position ({x}, {y})")

def mouse_callback(event, x, y, flags, param):
    global draw_state

    if draw_state.current == 1:
        draw_circle(event, x, y, flags, param)
    elif draw_state.current == 2:
        draw_rectangle(event, x, y, flags, param)
    elif draw_state.current == 3:
        draw_line(event, x, y, flags, param)
    elif draw_state.current == 4:
        write_text(event, x, y, flags, param)

def drawing_tool(image_path = None, uploaded_file = None):
    global draw_state
    if uploaded_file is not None:
        tmp_path = create_temp_file_from_upload(uploaded_file)
        draw_state.img = cv2.imread(tmp_path)
    elif image_path and os.path.exists(image_path):
        draw_state.img = cv2.imread(image_path)
    else:
        st.error("No valid image provided!")
        return
    
    draw_state.original_img = draw_state.img.copy()
    draw_state.current_mode = 1
    

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_callback)
    
    print(" KEYBOARD SHORTCUTS:")
    print("   1 - Circle mode (Red circles)")
    print("   2 - Rectangle mode (Green rectangles)")  
    print("   3 - Line mode (Blue lines)")
    print("   4 - Text mode (Green text) - Type text in console after clicking on image")
    print("   s - Save current image")
    print("   r - Reset to original image")
    print("   ESC - Exit drawing tool")

    while True:
        cv2.imshow("image", draw_state.img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('1'):
            draw_state.current = 1
            print("Switched to Circle mode (Red)")
        elif key == ord('2'):
            draw_state.current = 2
            print("Switched to Rectangle mode (Green)")
        elif key == ord('3'):
            draw_state.current = 3
            print("Switched to Line mode (Blue)")
        elif key == ord('4'):
            draw_state.current = 4
            print("Switched to Text mode (Green)")
        elif key == ord('s'):
            filename = f"drawn_image_{int(time.time())}.jpg"
            cv2.imwrite(filename, draw_state.img)
        elif key == ord('r'):
            draw_state.img = draw_state.original_img.copy()
            cv2.imshow("image", draw_state.img)
    
    cv2.destroyAllWindows()

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
    ["Face Recognition (Live Camera)", "Face Recognition (Upload)", "Landmark Detection (Live Camera)", "Landmark Detection (Upload)", "Live Landmark and Face Recognition", "Image Quality Assessment", "Image Drawing Tool", "Day/Night Classification", "Image Similarity Search"]
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
elif app_mode == "Landmark Detection (Live Camera)":
    st.header("🏛️ Landmark Detection - Live Camera")
    
    if st.button("Start Live Landmark Detection"):
        st.info("Starting camera... Press ESC in the OpenCV window to stop")
        sift, bf, landmark_descriptors, landmark_keypoints, landmark_images, landmark_names = initialize_landmark_detection()
        
        cam = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            
            result, landmark_name, match_count = detect_landmark(
                frame, sift, bf, landmark_descriptors, landmark_keypoints, 
                landmark_images, landmark_names
            )
            
            if landmark_name != "No landmark detected":
                cv2.putText(result, f"Landmark: {re.sub(r'\d+', '', landmark_name).strip()}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(result, f"Matches: {match_count}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(result, "No landmark detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Landmark Detection", result)

            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cam.release()
        cv2.destroyAllWindows()
        st.success("Camera stopped!")

elif app_mode == "Landmark Detection (Upload)":
    st.header("🏛️ Landmark Detection - Upload Image")
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        tmp_path = create_temp_file_from_upload(uploaded_file)
        if tmp_path:
            sift, bf, landmark_descriptors, landmark_keypoints, landmark_images, landmark_names = initialize_landmark_detection()

            frame = cv2.imread(tmp_path)
            result, landmark_name, match_count = detect_landmark(
                frame, sift, bf, landmark_descriptors, landmark_keypoints, 
                landmark_images, landmark_names
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Query Image")
                st.image(uploaded_file, use_container_width=True)
            
            with col2:
                st.subheader("Landmark Detection Result")
                if landmark_name != "No landmark detected":
                    cv2.putText(result, f"Landmark: {re.sub(r'\d+', '', landmark_name).strip()}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(result, f"Matches: {match_count}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(result, "No landmark detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                st.image(result_rgb, use_container_width=True)
elif app_mode == "Live Landmark and Face Recognition":
    st.header("🏛️ Live Landmark and Face Recognition")
    
    if st.button("Start Live Recognition"):
        st.info("Starting camera... Press ESC in the OpenCV window to stop")
        face_cascade, known_face_encodings, known_face_names = initialize_face_recognition()
        sift, bf, landmark_descriptors, landmark_keypoints, landmark_images, landmark_names = initialize_landmark_detection()
        
        cam = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cam.read()
            if not ret:
                break
            
            face_rects = detect_faces(frame)
            frame = recognize_faces(frame, face_rects, known_face_encodings, known_face_names)
            
            result, landmark_name, match_count = detect_landmark(
                frame, sift, bf, landmark_descriptors, landmark_keypoints, 
                landmark_images, landmark_names
            )
            
            if landmark_name != "No landmark detected":
                cv2.putText(result, f"Landmark: {re.sub(r'\d+', '', landmark_name).strip()}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(result, f"Matches: {match_count}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(result, "No landmark detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Live Recognition", result)

            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cam.release()
        cv2.destroyAllWindows()
        st.success("Camera stopped!")
elif app_mode == "Day/Night Classification":
    st.header("🌅 Day/Night Classification")
    
    option = st.selectbox(
        "Choose input method:",
        ["Upload Single Image", "Process Folder"]
    )
    
    if option == "Upload Single Image":
        uploaded_file = st.file_uploader("Choose an image for day/night classification...", type=['jpg', 'jpeg', 'png'])
        
        metric_options = {
            "mean_hsv_v": "HSV Value Channel (Recommended)",
            "mean_lab_l": "LAB Lightness Channel", 
            "mean_rgb": "RGB Mean",
            "mean_gray": "Grayscale Mean"
        }
        
        selected_metric = st.selectbox(
            "Choose classification metric:",
            list(metric_options.keys()),
            format_func=lambda x: metric_options[x]
        )
        
        if uploaded_file is not None:
            tmp_path = create_temp_file_from_upload(uploaded_file)
            if tmp_path:
                img = cv2.imread(tmp_path)
                label, value, confidence = classify_day_night(img, selected_metric)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Image")
                    st.image(uploaded_file, use_container_width=True)
                
                with col2:
                    st.subheader("Classification Result")

                    result_img = img.copy()
                    color = (0, 255, 0) if label == "Day" else (0, 0, 255)
                    cv2.putText(result_img, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

                    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    st.image(result_rgb, use_container_width=True)

                st.subheader("Classification Details")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Prediction", label)
                with col2:
                    st.metric("Metric Value", f"{value:.2f}")
                with col3:
                    st.metric("Confidence", f"{confidence:.2f}")
    elif option == "Process Folder":
        folder_path = st.text_input("Enter folder path:", value="Images")
        
        if st.button("Process Folder") and os.path.exists(folder_path):
            results = []
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(folder_path, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        label, value, confidence = classify_day_night(img)
                        color = (0, 255, 0) if label == "Day" else (0, 0, 255)
                        cv2.putText(img, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
                        results.append({
                            'filename': filename,
                            'image': cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                            'label': label,
                            'value': value,
                            'confidence': confidence
                        })
            
            if results:
                st.success(f"Processed {len(results)} images!")
                
                for i, result in enumerate(results):
                    st.subheader(f"{result['filename']}")
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.image(result['image'], use_container_width=True)
                    
                    with col2:
                        st.metric("Classification", result['label'])
                        st.metric("Brightness Value", f"{result['value']:.2f}")
                        st.metric("Confidence", f"{result['confidence']:.2f}")
                    
                    st.divider()
            else:
                st.error("No valid images found in the specified folder!")
elif app_mode == "Image Quality Assessment":
    st.header("🔍 Image Quality Assessment")
    assessment_type = st.selectbox(
        "Choose assessment type",
        ["Upload Image", "Process IQA Folder - Blur", "Process IQA Folder - Exposure", 
         "Process IQA Folder - Contrast", "Process IQA Folder - Noise"])   
    if assessment_type == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image for quality assessment...", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            tmp_path = create_temp_file_from_upload(uploaded_file)
            if tmp_path:
                img = cv2.imread(tmp_path)
                results, highlighted_images = get_comprehensive_assessment(img)
                st.subheader("Original Image")
                st.image(uploaded_file, use_container_width=True)
                st.subheader("Quality Assessment Results")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Blur Detection", results['blur']['status'], f"Blur: {results['blur']['variance']:.2f}")
                with col2:
                    st.metric("Exposure", results['exposure']['status'])
                with col3:
                    st.metric("Contrast", results['contrast']['status'])
                with col4:
                    st.metric("Noise Detection", results['noise']['status'])

                st.subheader("Enhancement Suggestions")
                enhancements = []
                for key, result in results.items():
                    if result.get('enhancement'):
                        enhancements.append(f"**{key.title()}**: {result['enhancement']}")
                
                if enhancements:
                    for enhancement in enhancements:
                        st.write(f"• {enhancement}")
                else:
                    st.success("✅ No enhancements needed - Image quality is good!")

                if highlighted_images:
                    st.subheader("Problem Area Highlights")
                    cols = st.columns(len(highlighted_images))
                    for i, (issue_type, highlighted_img) in enumerate(highlighted_images.items()):
                        with cols[i]:
                            st.write(f"**{issue_type.title()} Issues**")
                            highlighted_rgb = cv2.cvtColor(highlighted_img, cv2.COLOR_BGR2RGB)
                            st.image(highlighted_rgb, use_container_width=True)
    elif assessment_type.startswith("Process IQA Folder"):
        folder_type = assessment_type.split(" - ")[1].lower()
        folder_path = f"IQA/detect_{folder_type}"
        if st.button(f"Process {folder_type.title()} Detection"):
            results = process_folder_with_highlighting(folder_path, folder_type)
            if results:
                st.success(f"Processed {len(results)} images!")
                for i, result in enumerate(results):
                    st.subheader(f"Image {i+1}: {result['filename']}")
                    col1, col2 = st.columns(2)
                    if result['problem']:
                        col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Original**")
                        st.image(result['original'], use_container_width=True)
                    
                    with col2:
                        st.write("**With Annotations**")
                        st.image(result['annotated'], use_container_width=True)
                    
                    if result['problem']:
                        with col3:
                            st.write("**Problem Highlights**")
                            st.image(result['highlighted'], use_container_width=True)
                    
                    if result['status']:
                        st.subheader(f"{folder_type.capitalize()}: {result['status']}")
                    
                    if result['enhancement']:
                        st.subheader("Enhancement Suggestions")
                        st.write(f"**{result['enhancement']}**")
                    st.divider()
elif app_mode == "Image Similarity Search":
    st.header("🔍 Image Similarity Search")
    
    uploaded_file = st.file_uploader("Choose a query image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        tmp_path = create_temp_file_from_upload(uploaded_file)
        if tmp_path:
            if st.button("Find Similar Images"):
                with st.spinner("Initializing similarity search..."):
                    sift, bf, landmark_descriptors, landmark_keypoints, landmark_images, landmark_names = initialize_similarity_search()
                
                if not landmark_descriptors:
                    st.error("No images found in 'Similarity_images' folder! Please add reference images.")
                else:
                    with st.spinner("Searching for similar images..."):
                        query_img = cv2.imread(tmp_path)
                        top_matches = retrieve_top_similar_images(
                            query_img, sift, bf, landmark_descriptors, 
                            landmark_keypoints, landmark_images, landmark_names
                        )
                    
                    st.subheader("Query Image")
                    st.image(uploaded_file, use_container_width=True)
                    
                    if top_matches:
                        st.subheader(f"Top {len(top_matches)} Similar Images")
                        
                        for i, (name, matched_img, inliers) in enumerate(top_matches):
                            st.write(f"**Match {i+1}: {name}**")
                            st.write(f"Inliers: {inliers}")

                            matched_rgb = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)
                            st.image(matched_rgb, use_container_width=True)
                            st.divider()
                    else:
                        st.warning("No similar images found with the current threshold settings. Try lowering the minimum matches threshold.")
elif app_mode == "Image Drawing Tool":
    st.header("🎨 Image Annotation Tool")

    col1, col2 = st.columns(2)
    
    with col1:
        use_default = st.checkbox("Use default image (content/starry_night.jpg)", value=True)
    
    with col2:
        uploaded_file = st.file_uploader("Or upload your own image:", type=['jpg', 'jpeg', 'png'])
    
    img_available = False
    if use_default and os.path.exists("content/starry_night.jpg"):
        img_available = True
        image_source = "default"
    elif uploaded_file is not None:
        img_available = True
        image_source = "uploaded"
    
    if img_available:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Launch Drawing Tool", type="primary"):
                st.success("Drawing Tool launched! Check the OpenCV window.")
                if image_source == "default":
                    drawing_tool(image_path="content/starry_night.jpg")
                else:
                    drawing_tool(uploaded_file=uploaded_file)
                
                st.success("✅ Drawing session completed!")
        
        st.markdown("""
        **Keyboard Shortcuts:**
        ```
        1 = Circle (Red)    |  S = Save
        2 = Rectangle (Green) |  R = Reset  
        3 = Line (Blue)     |  ESC = Exit
        4 = Text (Green) - Enter text in terminal after clicking on the image
        ```
        """)
    else:
        st.warning("⚠️ Please select a default image or upload your own image to start drawing!")
        
        # Show preview of default image if available
        if os.path.exists("content/starry_night.jpg"):
            with st.expander("👀 Preview Default Image"):
                default_img = cv2.imread("content/starry_night.jpg")
                if default_img is not None:
                    default_rgb = cv2.cvtColor(default_img, cv2.COLOR_BGR2RGB)
                    st.image(default_rgb, caption="Default: Starry Night", use_container_width=True)