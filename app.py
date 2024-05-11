import streamlit as st
import torch
import numpy as np
import cv2
import scipy.ndimage as ndi
from PIL import Image

# Load the 2D pose detection TorchScript model
pose_model = torch.jit.load('C:/Users/varun/Downloads/3DHumanPoseEstimation/Streamlit/results.pt')

def extract_keypoints(heatmaps):
    keypoints = []
    for heatmap in heatmaps:
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        keypoints.append((x, y))
    return keypoints

def process_image(image):
    # Convert PIL image to numpy array and from RGB to BGR
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    tensor_image = torch.tensor(image).permute(2, 0, 1).float()
    tensor_image = tensor_image.unsqueeze(0)  # Add batch dimension
    
    model_output = pose_model(tensor_image)
    if isinstance(model_output, list):
        heatmaps = model_output[0]
    else:
        heatmaps = model_output

    if torch.is_tensor(heatmaps):
        heatmaps = heatmaps.detach().cpu().numpy()
    else:
        heatmaps = np.array(heatmaps)
    
    heatmaps = heatmaps[0]
    keypoints = extract_keypoints(heatmaps)

    return keypoints

def draw_poses(image, keypoints):
    for x, y in keypoints:
        cv2.circle(image, (x, y), 5, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
    return image

# Streamlit app interface
st.title('2D Pose Detection with Webcam')

frame = st.camera_input("Capture an image from your webcam")

if frame:
    # Convert the image data to a PIL image
    image = Image.open(frame)
    keypoints = process_image(image)  # Process the image with the model

    image_array = np.array(image.convert('RGB'))  # Convert image for drawing
    image_with_poses = draw_poses(image_array.copy(), keypoints)
    
    image_with_poses = cv2.cvtColor(image_with_poses, cv2.COLOR_BGR2RGB)
    st.image(image_with_poses, caption='Processed Image with Pose Estimations', use_column_width=True)
    st.write("Pose processing completed")



