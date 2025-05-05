import cv2
import time
import random
import numpy as np
from pathlib import Path
from collections import deque
from ultralytics import YOLO
import argparse
import os
import sys
from elasticsearch import Elasticsearch
import datetime

ES_HOST = "localhost"
ES_PORT = 9200
VIDEO_METRICS_INDEX = "video_processing_metrics"
ADAPTATION_LOG_INDEX = "video_adaptation_logs"

es_client = None
try:
    es_client = Elasticsearch([{'host': ES_HOST, 'port': ES_PORT}])
    if not es_client.ping():
        raise ValueError("Connection Error: Cannot connect to Elasticsearch!")
    print("VideoSwitch: Connected to Elasticsearch.")

    # Ensure indices exist (create if not) - Basic mapping example
    if not es_client.indices.exists(index=VIDEO_METRICS_INDEX):
        # Add more fields as needed
        mapping = {"mappings": {"properties": {"timestamp": {"type": "date"}}}}
        es_client.indices.create(index=VIDEO_METRICS_INDEX, body=mapping)
        print(f"VideoSwitch: Created index '{VIDEO_METRICS_INDEX}'")

    if not es_client.indices.exists(index=ADAPTATION_LOG_INDEX):
        # Add more fields as needed
        mapping = {"mappings": {"properties": {"timestamp": {"type": "date"}}}}
        es_client.indices.create(index=ADAPTATION_LOG_INDEX, body=mapping)
        print(f"VideoSwitch: Created index '{ADAPTATION_LOG_INDEX}'")

except Exception as e:
    print(
        f"VideoSwitch: WARNING - Failed to connect or setup Elasticsearch: {e}. Logging disabled.")
    es_client = None


def log_to_es(client, index_name, data_dict):
    if client:
        try:
            data_dict['timestamp'] = datetime.datetime.now(
            ).isoformat()  # Add timestamp
            client.index(index=index_name, document=data_dict)
        except Exception as e:
            print(
                f"VideoSwitch: WARNING - Failed to log to Elasticsearch index '{index_name}': {e}")


class AdaptiveYOLOProcessor:
    def __init__(self, model_paths, output_fps, test_frames=30, upgrade_interval=60):
        """
        Initialize the adaptive YOLO processor using ultralytics.

        Args:
            model_paths (dict): Dictionary mapping model names to their paths, ordered from smallest to largest
            output_fps (float): Desired output fps
            test_frames (int): Number of frames to test for performance evaluation
            upgrade_interval (int): Number of seconds to wait before trying to upgrade to a better model
        """
        self.model_paths = model_paths
        self.output_fps = output_fps
        self.test_frames = test_frames
        self.target_frame_time = 1.0 / output_fps
        self.upgrade_interval = upgrade_interval

        # Load all models using ultralytics
        self.models = {}
        for name, path in model_paths.items():
            try:
                self.models[name] = YOLO(path)
                print(f"Loaded model: {name}")
            except Exception as e:
                print(f"Failed to load model {name}: {e}")

        if not self.models:
            raise ValueError("No models could be loaded")

        # Ordered model names from smallest to largest
        self.model_names = list(model_paths.keys())

        # Current active model
        self.current_model_idx = 0
        self.current_model_name = self.model_names[self.current_model_idx]

        # Processing timestamps for FPS calculation
        self.processing_times = deque(maxlen=100)

        # Last upgrade attempt timestamp
        self.last_upgrade_time = 0

    def test_model_performance(self, frames):
        """
        Test how different models perform on sample frames.

        Args:
            frames (list): List of sample frames

        Returns:
            dict: Model names mapped to their average processing time per frame
        """
        performance = {}

        for name, model in self.models.items():
            print(f"Testing model: {name}")
            times = []

            # Test on random frames
            test_indices = random.sample(
                range(len(frames)), min(self.test_frames, len(frames)))
            test_frames = [frames[i] for i in test_indices]

            for frame in test_frames:
                start_time = time.time()
                _ = model(frame)
                end_time = time.time()
                times.append(end_time - start_time)

            avg_time = sum(times) / len(times)
            performance[name] = avg_time
            print(
                f"Model {name} average processing time: {avg_time:.4f}s ({1.0/avg_time:.2f} FPS)")

        return performance

    def select_optimal_model(self, performance):
        """
        Select the optimal model based on performance data.

        Args:
            performance (dict): Model performance data

        Returns:
            str: Name of the optimal model
        """
        # Start from the largest model and go down
        for name in reversed(self.model_names):
            # If this model can process frames at our target FPS with some buffer (90% of target)
            if performance[name] <= self.target_frame_time * 0.9:
                return name

        # If no model meets the criteria, use the smallest one
        return self.model_names[0]

    def try_upgrade_model(self):
        """
        Check if we can upgrade to a better model.

        Returns:
            bool: True if model was upgraded, False otherwise
        """
        current_time = time.time()

        # Only try upgrading if enough time has passed since last attempt
        if current_time - self.last_upgrade_time < self.upgrade_interval:
            return False

        self.last_upgrade_time = current_time

        # If we're already using the best model, no need to upgrade
        if self.current_model_idx == len(self.model_names) - 1:
            return False

        # Check if we can upgrade to the next model
        next_model_idx = self.current_model_idx + 1
        next_model_name = self.model_names[next_model_idx]

        # Calculate current average processing time
        if not self.processing_times:
            return False

        avg_processing_time = sum(
            self.processing_times) / len(self.processing_times)

        # If we have enough headroom (processing at 80% of target or better)
        if avg_processing_time <= self.target_frame_time * 0.8:
            old_model_name = self.current_model_name
            print(
                f"Upgrading model from {self.current_model_name} to {next_model_name}")
            self.current_model_idx = next_model_idx
            self.current_model_name = next_model_name

            self.processing_times.clear()  # Reset timing data

            # Log Metrics
            log_data = {
                "event_type": "model_upgrade",
                "old_model": old_model_name,
                "new_model": next_model_name,
                "reason": "performance_headroom_available",
                "avg_processing_time_before": avg_processing_time,
                "target_frame_time": self.target_frame_time
            }
            log_to_es(es_client, ADAPTATION_LOG_INDEX, log_data)
            return True

        return False

    def check_downgrade_model(self, last_processing_time):
        """
        Check if we need to downgrade to a less resource-intensive model.

        Args:
            last_processing_time (float): Time taken to process the last frame

        Returns:
            bool: True if model was downgraded, False otherwise
        """
        # If we're consistently missing our target FPS
        recent_avg = sum(self.processing_times) / len(
            self.processing_times) if self.processing_times else last_processing_time

        # If we're taking more than 110% of our target time per frame
        if recent_avg > self.target_frame_time * 1.1 and self.current_model_idx > 0:
            old_model_name = self.current_model_name
            new_model_idx = self.current_model_idx - 1
            new_model_name = self.model_names[new_model_idx]
            print(
                f"Downgrading model from {self.current_model_name} to {new_model_name} due to performance constraints")
            self.current_model_idx = new_model_idx
            self.current_model_name = new_model_name

            self.processing_times.clear()  # Reset timing data

            # Log Metrics
            log_data = {
                "event_type": "model_downgrade",
                "old_model": old_model_name,
                "new_model": new_model_name,
                "reason": "target_fps_missed",
                "avg_processing_time_before": recent_avg,
                "target_frame_time": self.target_frame_time
            }
            log_to_es(es_client, ADAPTATION_LOG_INDEX, log_data)

            return True

        # self.processing_times.clear()
        return False

    def process_frame(self, frame, frame_number):
        """
        Process a single frame using the current model.

        Args:
            frame (numpy.ndarray): Input frame
            frame_number (int): Current frame number for logging

        Returns:
            numpy.ndarray: Processed frame with detections
        """
        model = self.models[self.current_model_name]
        model_used_for_frame = self.current_model_name

        # Print current model being used for this inference
        print(
            f"\nFrame {frame_number}: Using model {self.current_model_name} for inference", end='')

        start_time = time.time()
        results = model(frame)
        end_time = time.time()

        processing_time = end_time - start_time
        self.processing_times.append(processing_time)

        # Print inference time
        current_fps = 1.0 / processing_time if processing_time > 0 else 0
        print(
            f"Frame {frame_number}: Inference time: {processing_time:.4f}s ({current_fps:.2f} FPS)")

        # Log Metrics
        confidences = results[0].boxes.conf.tolist()
        num_detections = len(confidences)
        avg_confidence = sum(confidences) / num_detections if num_detections > 0 else 0

        frame_metrics = {
            "frame_number": frame_number,
            "model_name": model_used_for_frame,
            "model_processing_time": processing_time,
            "current_fps": current_fps,
            "num_detections": num_detections,
            "avg_confidence": avg_confidence, # Example additional metric
            "target_frame_time": self.target_frame_time
            # Add other relevant metrics like CPU/memory if easily obtainable
        }
        log_to_es(es_client, VIDEO_METRICS_INDEX, frame_metrics)

        # Check if we need to downgrade the model
        if len(self.processing_times) >= 5:  # Wait for some data to accumulate
            self.check_downgrade_model(processing_time)

        # Try to upgrade model periodically
        self.try_upgrade_model()

        # Render results
        annotated_frame = results[0].plot()
        return annotated_frame

    def process_video(self, input_path, output_path=None):
        """
        Process an entire video file.

        Args:
            input_path (str): Path to input video
            output_path (str, optional): Path for output video. If None, no output video is saved.
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")

        # Get video properties
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Input video: {input_path}")
        print(f"Resolution: {frame_width}x{frame_height}")
        print(f"Input FPS: {input_fps}")
        print(f"Target output FPS: {self.output_fps}")
        print(f"Total frames: {total_frames}")

        # If target output fps is > input fps, then we break/return
        if (self.output_fps > input_fps):
            print(
                f"Input fps: {input_fps} of given video must be greater than target Output fps: {self.output_fps}")
            
            self.output_fps = input_fps
            self.target_frame_time = 1.0 / self.output_fps

        # Setup output video writer if needed
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc,
                                  self.output_fps, (frame_width, frame_height))

        # Extract frames for initial testing
        test_frames = []
        frame_indices = sorted(random.sample(
            range(total_frames), min(100, total_frames)))

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                test_frames.append(frame)

        # Reset to beginning of video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Test performance on all models
        performance = self.test_model_performance(test_frames)

        # Select initial model
        optimal_model = self.select_optimal_model(performance)
        self.current_model_idx = self.model_names.index(optimal_model)
        self.current_model_name = optimal_model
        print(f"Selected initial model: {self.current_model_name}")

        # Calculate frame skipping to achieve output FPS
        if input_fps > self.output_fps:
            frame_interval = max(1, round(input_fps / self.output_fps))
        else:
            frame_interval = 1

        frame_count = 0
        processed_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames to match output FPS
            if (frame_count - 1) % frame_interval != 0:
                continue

            processed_count += 1

            # Process the frame
            processed_frame = self.process_frame(frame, frame_count)

            # Display FPS info on frame
            current_fps = 1.0 / \
                self.processing_times[-1] if self.processing_times else 0
            cv2.putText(
                processed_frame,
                f"Model: {self.current_model_name} | FPS: {current_fps:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            # Write to output video if configured
            if out:
                out.write(processed_frame)

            # Display the frame
            cv2.imshow('Adaptive YOLO Processor', processed_frame)

            # Break on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Print progress every 100 frames
            if processed_count % 100 == 0:
                elapsed = time.time() - start_time
                avg_fps = processed_count / elapsed
                print(f"Processed {processed_count} frames ({frame_count} total) | "
                      f"Current model: {self.current_model_name} | "
                      f"Average FPS: {avg_fps:.2f}")

        end_process_time = time.time()
        total_duration = end_process_time - start_time
        overall_avg_fps = processed_count / total_duration if total_duration > 0 else 0
        summary_log = {
            "event_type": "processing_finished",
            "input_video": input_path,
            "output_video": output_path if output_path else "N/A",
            "total_frames_in_video": total_frames,
            "total_frames_processed": processed_count,
            "final_model_used": self.current_model_name,
            "target_output_fps": self.output_fps,
            "processing_duration_seconds": total_duration,
            "overall_average_fps": overall_avg_fps
        }
        log_to_es(es_client, ADAPTATION_LOG_INDEX, summary_log)

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        # print("\n")
        print(f"\nVideo processing complete!")
        print(
            f"Processed {processed_count} frames out of {frame_count} total frames")
        print(f"Final model used: {self.current_model_name}")


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Adaptive YOLO Video Processor")
    parser.add_argument("--input", required=True,
                        help="Path to the input video file.")
    parser.add_argument("--output", required=True,
                        help="Path for the output video file.")
    parser.add_argument("--fps", type=float, required=True,
                        help="Target output FPS.")
    parser.add_argument("--test_frames", type=int, default=30,
                        help="Number of frames for performance testing.")
    parser.add_argument("--upgrade_interval", type=int, default=30,
                        help="Interval in seconds to check for model upgrade.")

    args = parser.parse_args()

    print("--- Video Processor Script Started ---")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Target FPS: {args.fps}")

    # Find Video?
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
            raise ValueError(f"Could not open video file: {args.input}")

    # Download Models
    yolov5n = YOLO("yolov5n.pt")
    yolov5s = YOLO("yolov5s.pt")
    yolov5m = YOLO("yolov5m.pt")
    # yolov5l = YOLO("yolov5l.pt")
    # yolov5x = YOLO("yolov5x.pt")

    # Define model paths - from smallest/fastest to largest/most accurate
    model_paths = {
        "yolov5n": "yolov5nu.pt",  # Nano
        "yolov5s": "yolov5su.pt",  # Small
        "yolov5m": "yolov5mu.pt",  # Medium
        # "yolov5l": "yolov5lu.pt",  # Large
        # "yolov5x": "yolov5xu.pt"   # XLarge
    }

    # Set desired output FPS
    # output_fps = 16

    # Create processor
    processor = AdaptiveYOLOProcessor(
        model_paths=model_paths,
        output_fps=args.fps,
        test_frames=30,
        upgrade_interval=30  # Try upgrading model every 30 seconds
    )

    # Process video
    input_video = args.input
    output_video = args.output
    processor.process_video(input_video, output_video)