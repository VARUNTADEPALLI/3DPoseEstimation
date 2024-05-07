import streamlit as st
import torch
import cv2
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Load the TorchScript model
model = torch.jit.load('results.pt')
model.eval()

class PoseEstimationProcessor(VideoProcessorBase):
    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Preprocess the image as required by your model
        input_tensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = cv2.resize(input_tensor, (256, 256))  # Adjust size based on your model's input
        input_tensor = torch.tensor(input_tensor).permute(2, 0, 1).unsqueeze(0).float()
        input_tensor = input_tensor / 255.0  # Normalize if your model expects normalized inputs

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)

        # Postprocessing and visualization
        # Example: assuming output is coordinates of keypoints
        output = output.squeeze(0).numpy()
        points = np.where(output > 0.5)  # Adjust threshold based on your output format and needs
        for point in zip(*points):
            x, y = point[1], point[0]  # Adjust based on how your model outputs coordinates
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

def main():
    st.title("Real-time Pose Estimation")

    # Start the webcam and process the video stream
    webrtc_streamer(key="pose-estimation", video_processor_factory=PoseEstimationProcessor)

if __name__ == "__main__":
    main()
