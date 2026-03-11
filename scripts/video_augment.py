import cv2
import numpy as np

def interpolate_frames(frame1, frame2, ratio):
    """Linearly interpolates between two frames."""
    return cv2.addWeighted(frame1, 1 - ratio, frame2, ratio, 0)

def add_noise(frame, noise_ratio=0.01):
    """Adds white noise to a frame."""
    noise = (np.random.randn(*frame.shape) * (255 * noise_ratio)).astype(np.uint8)
    return cv2.add(frame, noise)

def video_augment(
    input_path: str,
    output_path: str,
    target_frame_rate: int,
    new_width: int,
    new_height: int,
    time_scale: float = 1.0,
    interpolation_method: str = "linear",
    white_noise_ratio: float = 0.005
):
    """
    Augments a base video by interpolating frames, adding noise, and upscaling.
    """
    cap = cv2.VideoCapture(input_path)
    base_fps = cap.get(cv2.CAP_PROP_FPS)
    base_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Adjust target frame rate based on time_scale
    final_fps = target_frame_rate
    final_duration = (base_frame_count / base_fps) * time_scale
    
    # The number of interpolated frames will determine the new length
    # To achieve time_scale, we need to adjust how many frames we generate.
    # Total frames needed = final_duration * final_fps
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, final_fps, (new_width, new_height))
    
    ret, prev_frame = cap.read()
    if not ret:
        print("Could not read video.")
        return
        
    prev_frame = cv2.resize(prev_frame, (new_width, new_height))
    out.write(add_noise(prev_frame, white_noise_ratio))

    # Calculate how many frames to insert between each original frame
    interpolation_factor = int(final_fps / base_fps * time_scale)
    
    while True:
        ret, current_frame = cap.read()
        if not ret:
            break
            
        current_frame = cv2.resize(current_frame, (new_width, new_height))
        
        # Interpolate between prev_frame and current_frame
        for i in range(1, interpolation_factor):
            ratio = i / interpolation_factor
            
            if interpolation_method == "linear":
                interp_frame = interpolate_frames(prev_frame, current_frame, ratio)
            else: # Placeholder for more methods
                interp_frame = interpolate_frames(prev_frame, current_frame, ratio)
            
            out.write(add_noise(interp_frame, white_noise_ratio))
        
        out.write(add_noise(current_frame, white_noise_ratio))
        prev_frame = current_frame

    cap.release()
    out.release()
    print(f"Augmented video saved to {output_path}")

if __name__ == "__main__":
    # Example usage: Double the video length
    video_augment(
        input_path=r'D:\OmissionAnalysis\microcircuit_schizophrenia.mp4',
        output_path=r'D:\OmissionAnalysis\augmented_video_2x_slow.mp4',
        target_frame_rate=100,
        new_width=1000,
        new_height=1000,
        time_scale=2.0,
        white_noise_ratio=0.005
    )
