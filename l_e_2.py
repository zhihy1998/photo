import cv2
import numpy as np


def stack_video_frames(video_path, output_image_path, num_frames_to_average=100):
    """
    Extracts and averages a specified number of frames from a video
    to create a single composite image.

    Args:
        video_path (str): The path to the input_file video file.
        output_image_path (str): The path to save the resulting image.
        num_frames_to_average (int): The number of frames to average.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Read the first frame to get dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        cap.release()
        return

    # Initialize a blank image (float32 for averaging)
    averaged_image = np.zeros_like(frame, dtype=np.float32)
    frame_count = 0

    while frame_count < num_frames_to_average:
        ret, frame = cap.read()
        if not ret:
            break  # End of video or error

        averaged_image += frame.astype(np.float32)
        frame_count += 1

    cap.release()

    if frame_count > 0:
        averaged_image /= frame_count
        # Convert back to uint8 for saving
        final_image = averaged_image.astype(np.uint8)
        cv2.imwrite(output_image_path, final_image)
        print(f"Averaged image saved to {output_image_path}")
    else:
        print("No frames were processed.")


# Example usage:
stack_video_frames(r"D:\github\photo_long_exposure\input\firefly.mp4",
                   r"D:\github\photo_long_exposure\output\firefly.png",
                   num_frames_to_average=500000)
