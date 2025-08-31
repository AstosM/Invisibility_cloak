import cv2
import numpy as np
import gradio as gr
import time

background = None

def invisibility(frame):
    global background

    # Initialize background once
    if background is None:
        background = frame.copy()

    # Flip frame for natural effect
    frame = cv2.flip(frame, 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define cloak color range (red here)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Mask
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    cloak_mask = mask1 + mask2

    # Clean mask
    cloak_mask = cv2.morphologyEx(cloak_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    cloak_mask = cv2.dilate(cloak_mask, np.ones((3, 3), np.uint8), iterations=1)

    # Inverse mask
    inverse_mask = cv2.bitwise_not(cloak_mask)

    cloak_area = cv2.bitwise_and(background, background, mask=cloak_mask)
    non_cloak_area = cv2.bitwise_and(frame, frame, mask=inverse_mask)

    final_output = cv2.addWeighted(cloak_area, 1, non_cloak_area, 1, 0)

    return final_output

# Gradio interface with webcam support
demo = gr.Interface(
    fn=invisibility,
    inputs=gr.Image(label="Webcam Input", type="numpy", webcam=True),
    outputs=gr.Image(label="Cloak Effect"),
    live=True,
    title="🧙 Magic Invisibility Cloak",
    description="Wear a red cloth and disappear!"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
