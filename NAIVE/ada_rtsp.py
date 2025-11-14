import cv2
import time
import subprocess
from collections import deque
from ultralytics import YOLO

# sudo docker run --rm -it -p 8554:8554 bluenviron/mediamtx
# ffmpeg -re -stream_loop -1 -i trim.mp4 -c copy -f rtsp -rtsp_transport tcp rtsp://localhost:8554/mystream

class AdaptiveYOLOProcessor:
    def __init__(self, model_paths, output_fps, upgrade_interval=60):
        """
        Initialize the adaptive YOLO processor using ultralytics.

        Args:
            model_paths (dict): Dictionary mapping model names to their paths, ordered from smallest to largest.
            output_fps (float): Desired output fps.
            upgrade_interval (int): Seconds to wait before trying to upgrade to a better model.
        """
        self.model_paths = model_paths
        self.output_fps = output_fps
        self.target_frame_time = 1.0 / output_fps
        self.upgrade_interval = upgrade_interval

        # Load all models using ultralytics
        self.models = {name: YOLO(path) for name, path in model_paths.items()}
        print(f"Loaded models: {list(self.models.keys())}")

        if not self.models:
            raise ValueError("No models could be loaded")

        # Ordered model names from smallest to largest
        self.model_names = list(model_paths.keys())

        # Start with the smallest/fastest model for live streams
        self.current_model_idx = 0
        self.current_model_name = self.model_names[self.current_model_idx]
        print(f"Starting with initial model: {self.current_model_name}")

        self.processing_times = deque(maxlen=100)
        self.last_upgrade_time = 0

    def try_upgrade_model(self):
        """
        Check if we can upgrade to a better model based on current performance.
        """
        current_time = time.time()
        if current_time - self.last_upgrade_time < self.upgrade_interval:
            return False

        self.last_upgrade_time = current_time
        if self.current_model_idx >= len(self.model_names) - 1:
            return False # Already at the best model

        if not self.processing_times:
            return False

        avg_processing_time = sum(self.processing_times) / len(self.processing_times)

        # Upgrade if we have significant performance headroom (e.g., using less than 70% of target time)
        if avg_processing_time <= self.target_frame_time * 0.8:
            next_model_idx = self.current_model_idx + 1
            next_model_name = self.model_names[next_model_idx]
            print(f"\n--- Upgrading model from {self.current_model_name} to {next_model_name} ---\n")
            self.current_model_idx = next_model_idx
            self.current_model_name = next_model_name
            self.processing_times.clear()
            return True
        return False

    def check_downgrade_model(self):
        """
        Check if we need to downgrade to a less resource-intensive model.
        """
        # Calculate average time of the last 20 frames
        if len(self.processing_times) < 20:
             return False

        recent_times = list(self.processing_times)[-20:]
        recent_avg = sum(recent_times) / len(recent_times)

        # Downgrade if we are consistently failing to meet the target time (e.g., taking >105% of target time)
        if recent_avg > self.target_frame_time * 1.05 and self.current_model_idx > 0:
            new_model_idx = self.current_model_idx - 1
            new_model_name = self.model_names[new_model_idx]
            print(f"\n--- Downgrading model from {self.current_model_name} to {new_model_name} due to performance ---\n")
            self.current_model_idx = new_model_idx
            self.current_model_name = new_model_name
            self.processing_times.clear()
            return True
        return False

    def process_frame(self, frame, frame_count):
        """
        Process a single frame using the current model and handle adaptive logic.
        """
        model = self.models[self.current_model_name]

        start_time = time.time()
        results = model(frame, verbose=False) # Set verbose=False for cleaner logs
        end_time = time.time()

        processing_time = end_time - start_time
        self.processing_times.append(processing_time)

        # Adaptive logic
        self.check_downgrade_model()
        self.try_upgrade_model()

        # Render results and FPS info
        annotated_frame = results[0].plot()
        current_fps = 1.0 / processing_time if processing_time > 0 else 0
        
        cv2.putText(
            annotated_frame,
            f"Model: {self.current_model_name}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.putText(
            annotated_frame,
            f"FPS: {current_fps:.2f}",
            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        
        # Frame Count - For Testing
        cv2.putText(
            annotated_frame,
            f"Frame: {frame_count}",
            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        return annotated_frame

    def process_rtsp_stream(self, input_rtsp, output_rtsp):
        """
        Process an RTSP stream and output to another RTSP stream using TCP.
        """
        cap = cv2.VideoCapture(input_rtsp)
        if not cap.isOpened():
            raise ValueError(f"Could not open RTSP stream: {input_rtsp}")

        # Get video properties from the source
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Input Stream: {input_rtsp}")
        print(f"Resolution: {frame_width}x{frame_height} @ {source_fps:.2f} FPS")
        print(f"Target Processing FPS: {self.output_fps}")
        print(f"Output Stream: {output_rtsp}")

        if source_fps > 0:
            if self.output_fps > source_fps:
                print(f"\n[ERROR] Target output FPS ({self.output_fps}) cannot be greater than the input stream's FPS ({source_fps:.2f}).")
                print("Please set OUTPUT_FPS to a value less than or equal to the source FPS.")
                cap.release()
                return # Terminate the function
        else:
            print("\n[WARNING] Could not determine the FPS of the input stream. Proceeding without frame rate validation.")

        # FFmpeg command to create the output RTSP stream using TCP
        ffmpeg_command = [
            'ffmpeg',
            # Input options: receiving raw frames from this script's stdout
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{frame_width}x{frame_height}',
            '-r', str(self.output_fps),
            '-i', '-',
            # Output options: encoding and streaming to the RTSP server
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-rtsp_transport', 'tcp',
            '-f', 'rtsp',
            output_rtsp
        ]

        # Start the FFmpeg process
        print("\nStarting FFmpeg process...\n")
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

        frame_count = 0
        last_frame_time = time.time()
        total_start_time = time.time() # Added for overall FPS calculation

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from input stream. Attempting to reconnect...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(input_rtsp)
                continue
            
            frame_count += 1
            # Pass the frame_count to the processing method
            annotated_frame = self.process_frame(frame, frame_count)
            
            # Write the processed frame to FFmpeg's stdin
            try:
                ffmpeg_process.stdin.write(annotated_frame.tobytes())
            except (BrokenPipeError, IOError):
                print("FFmpeg process has closed. Exiting.")
                break

            # Display the processed frame locally (optional)
            # cv2.imshow('Adaptive YOLO RTSP Processor', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Loop timing to approximate the target output FPS
            wait_time = max(0, self.target_frame_time - (time.time() - last_frame_time))
            time.sleep(wait_time)
            last_frame_time = time.time()
            
            # Progress Indicator
            if frame_count % 100 == 0:
                elapsed_time = time.time() - total_start_time
                avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                print(
                    f"  Processed: {frame_count} frames | "
                    f"Current Model: {self.current_model_name} | "
                    # f"Overall Avg FPS: {avg_fps:.2f}"
                )

        # Cleanup
        print("Shutting down...")
        cap.release()
        if ffmpeg_process.stdin:
            ffmpeg_process.stdin.close()
        if ffmpeg_process:
            ffmpeg_process.wait()
        cv2.destroyAllWindows()
        print("Stream processing finished.")


if __name__ == "__main__":
    # Define model paths - from smallest/fastest to largest/most accurate
    # Ensure you have these .pt files in your directory
    model_paths = {
        "yolov5n": "yolov5n.pt",   # Nano
        "yolov5s": "yolov5s.pt",   # Small
        "yolov5m": "yolov5m.pt",   # Medium
        "yolov5l": "yolov5l.pt",   # Large
        "yolov5x": "yolov5x.pt",   # XLarge
    }

    # --- Configuration ---
    # Set desired output FPS
    OUTPUT_FPS = 15
    # The RTSP stream from your camera or simulated feed
    INPUT_RTSP_URL = "rtsp://localhost:8554/mystream"
    # The new RTSP stream that will contain the annotated video
    OUTPUT_RTSP_URL = "rtsp://localhost:8554/annotated"

    # Create and run the processor
    processor = AdaptiveYOLOProcessor(
        model_paths=model_paths,
        output_fps=OUTPUT_FPS,
        upgrade_interval=30  # Try upgrading model every 30 seconds
    )
    processor.process_rtsp_stream(INPUT_RTSP_URL, OUTPUT_RTSP_URL)
