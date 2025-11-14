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
import requests
from elasticsearch import Elasticsearch, ElasticsearchException
import datetime
import logging

# --- Configuration ---
ES_HOST = "localhost"
ES_PORT = 9200
VIDEO_METRICS_INDEX = "video_processing_metrics"
ADAPTATION_LOG_INDEX = "video_adaptation_logs"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, # Change to logging.DEBUG for more verbose output
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout) # Log to standard output
logger = logging.getLogger("AdaptiveYOLO")

# --- Elasticsearch Mappings ---
VIDEO_METRICS_MAPPING = {
    "mappings": {
        "properties": {
            "timestamp": {"type": "date"},
            "frame_number": {"type": "integer"},
            "model_name": {"type": "keyword"},
            "model_processing_time": {"type": "float"},
            "current_fps": {"type": "float"},
            "num_detections": {"type": "integer"},
            "avg_confidence": {"type": "float"},
            "target_frame_time": {"type": "float"}
            # Add other relevant metrics like CPU/memory if easily obtainable
        }
    }
}

ADAPTATION_LOG_MAPPING = {
    "mappings": {
        "properties": {
            "timestamp": {"type": "date"},
            "event_type": {"type": "keyword"}, # e.g., "model_upgrade", "model_downgrade", "processing_finished", "initial_selection"
            "input_video": {"type": "keyword"},
            "output_video": {"type": "keyword"},
            "old_model": {"type": "keyword"},
            "new_model": {"type": "keyword"},
            "reason": {"type": "text"},
            "avg_processing_time_before": {"type": "float"},
            "target_frame_time": {"type": "float"},
            "total_frames_in_video": {"type": "integer"},
            "total_frames_processed": {"type": "integer"},
            "final_model_used": {"type": "keyword"},
            "target_output_fps": {"type": "float"},
            "processing_duration_seconds": {"type": "float"},
            "overall_average_fps": {"type": "float"}
            # Add other fields relevant to adaptation logs
        }
    }
}

# --- Elasticsearch Client Setup ---
def setup_elasticsearch(host, port, metrics_index, metrics_mapping, log_index, log_mapping):
    """ Attempts to connect to Elasticsearch and ensure indices exist. """
    es_client = None
    try:
        es_client = Elasticsearch(
            [{'host': host, 'port': port}],
            timeout=10, # Add a timeout
            max_retries=3,
            retry_on_timeout=True
        )
        if not es_client.ping():
            raise ElasticsearchException(f"Connection Error: Cannot ping Elasticsearch at {host}:{port}!")
        logger.info(f"Successfully connected to Elasticsearch at {host}:{port}")

        # Ensure indices exist with mappings
        for index_name, index_mapping in [(metrics_index, metrics_mapping), (log_index, log_mapping)]:
             if not es_client.indices.exists(index=index_name):
                try:
                    es_client.indices.create(index=index_name, body=index_mapping, ignore=400) # ignore=400 avoids error if index created between check and create
                    logger.info(f"Created Elasticsearch index '{index_name}'")
                except ElasticsearchException as create_err:
                    # Check if it was just a race condition (index already exists)
                    if not es_client.indices.exists(index=index_name):
                         logger.error(f"Failed to create index '{index_name}': {create_err}")
                         # Decide if you want to continue without this index or exit
                    else:
                         logger.warning(f"Index '{index_name}' already existed (likely race condition).")
             else:
                 logger.info(f"Elasticsearch index '{index_name}' already exists.")

        return es_client

    except ElasticsearchException as e:
        logger.error(f"Failed to connect or setup Elasticsearch: {e}. Logging will be disabled.")
        return None
    except Exception as e: # Catch other potential errors like config issues
        logger.error(f"An unexpected error occurred during Elasticsearch setup: {e}. Logging disabled.")
        return None

# --- Elasticsearch Logging Function ---
def log_to_es(client, index_name, data_dict):
    """ Logs a dictionary to the specified Elasticsearch index. """
    if client:
        try:
            data_dict['timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat() # Use UTC timestamp
            logger.debug(f"Attempting to log to ES index '{index_name}': {data_dict}") # Debug log data
            response = client.index(index=index_name, body=data_dict) # Use document= instead of body= for newer versions
            logger.debug(f"ES index response for '{index_name}': {response}")
            # Removed the successful log print here to reduce console noise, rely on DEBUG level if needed
        except ElasticsearchException as e:
            logger.warning(f"Failed to log to Elasticsearch index '{index_name}': {e}. Data: {data_dict}")
        except Exception as e:
            logger.error(f"Unexpected error during logging to ES index '{index_name}': {e}. Data: {data_dict}")
    # else: # Optionally log that logging is disabled if client is None
    #     logger.debug("Elasticsearch client is None, skipping logging.")

# --- Model Download Function ---
def download_model(url, save_path):
    """Downloads a file from a URL with a progress bar."""
    if os.path.exists(save_path):
        logger.info(f"Model already exists at {save_path}, skipping download.")
        return True
    
    logger.info(f"Downloading model from {url} to {save_path}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Download complete: {save_path}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download model from {url}: {e}")
        return False

# --- Adaptive Processor Class ---
class AdaptiveYOLOProcessor:
    # Constants for adaptation logic
    _TIMING_DEQUE_LEN = 100       # How many frames to average processing time over
    _DOWNGRADE_CHECK_WINDOW = 5   # Min number of frames processed before checking for downgrade
    _UPGRADE_HEADROOM_FACTOR = 0.8 # Upgrade if processing time is <= this fraction of target time
    _DOWNGRADE_THRESHOLD_FACTOR = 1.1 # Downgrade if avg processing time is >= this fraction of target time

    def __init__(self, model_paths, output_fps, test_frames=30, upgrade_interval=60, es_client=None):
        """
        Initialize the adaptive YOLO processor.

        Args:
            model_paths (dict): Dictionary mapping model names to their paths, ordered smallest to largest.
            output_fps (float): Desired output fps.
            test_frames (int): Number of frames for initial performance evaluation.
            upgrade_interval (int): Seconds to wait before trying to upgrade model.
            es_client (Elasticsearch, optional): Elasticsearch client instance. Defaults to None.
        """
        self.model_paths = model_paths
        self.output_fps = output_fps
        self.target_frame_time = 1.0 / output_fps if output_fps > 0 else float('inf')
        self.es_client = es_client # Store ES client
        self.test_frames = test_frames
        self.upgrade_interval = upgrade_interval
        self.last_upgrade_time = 0

        # Load all models using ultralytics after ensuring they exist
        self.models = {}
        for name, path in model_paths.items():
            if not os.path.exists(path):
                 logger.warning(f"Model file not found: {path}. Attempting to download.")
                 # Construct the URL from the model name for a public source, e.g., GitHub
                 # NOTE: This assumes ultralytics models are available at a predictable URL
                 url = f'https://github.com/ultralytics/yolov5/releases/download/v7.0/{name}.pt'
                 if not download_model(url, path):
                     continue # Skip to the next model if download fails

            try:
                self.models[name] = YOLO(path)
                logger.info(f"Successfully loaded model: {name} from {path}")
            except Exception as e:
                logger.error(f"Failed to load model '{name}' from {path}: {e}")

        if not self.models:
            raise ValueError("No YOLO models could be loaded. Please check paths and model files.")

        # Ordered model names from smallest to largest (based on loaded models)
        self.model_names = [name for name in model_paths if name in self.models] # Only use names of successfully loaded models
        if not self.model_names:
             raise ValueError("Model loading succeeded but no model names available. Check model_paths keys.")

        # Current active model (initialized later)
        self.current_model_idx = -1
        self.current_model_name = None

        # Processing timestamps for FPS calculation and adaptation
        self.processing_times = deque(maxlen=self._TIMING_DEQUE_LEN)

    def test_model_performance(self, frames):
        """ Test model performance on sample frames. """
        performance = {}
        if not frames:
            logger.warning("No frames provided for performance testing.")
            return performance

        for name in self.model_names: # Iterate based on the order defined in __init__
             model = self.models[name]
             logger.info(f"Testing performance of model: {name}")
             times = []

             # Test on random sample of the provided frames
             num_test = min(self.test_frames, len(frames))
             if num_test == 0:
                 logger.warning(f"Cannot test model {name}, not enough sample frames.")
                 continue
             test_indices = random.sample(range(len(frames)), num_test)
             current_test_frames = [frames[i] for i in test_indices]

             try:
                 for frame in current_test_frames:
                     start_time = time.monotonic() # Use monotonic clock for measuring intervals
                     _ = model(frame, verbose=False) # verbose=False reduces ultralytics console output
                     end_time = time.monotonic()
                     times.append(end_time - start_time)

                 if times:
                     avg_time = sum(times) / len(times)
                     performance[name] = avg_time
                     logger.info(f"Model '{name}' avg processing time: {avg_time:.4f}s ({1.0/avg_time:.2f} FPS)")
                 else:
                      logger.warning(f"No successful processing times recorded for model '{name}'.")

             except Exception as e:
                 logger.error(f"Error during performance testing for model '{name}': {e}")

        return performance

    def select_optimal_model(self, performance):
        """ Select the initial optimal model based on performance. """
        optimal_model_name = None
        # Start from the largest model and go down
        for name in reversed(self.model_names):
            if name not in performance:
                logger.warning(f"No performance data for model '{name}', skipping.")
                continue
            # Allow a small buffer (e.g., 5%) over the target time for selection
            if performance[name] <= self.target_frame_time * 1.05:
                optimal_model_name = name
                break # Found the best model that meets the criteria

        # If no model meets the criteria, use the smallest one available
        if optimal_model_name is None:
            if self.model_names:
                optimal_model_name = self.model_names[0]
                logger.warning(f"No model meets the target FPS ({self.output_fps:.2f}). Selecting the smallest model: {optimal_model_name}")
            else:
                 logger.error("Cannot select initial model: No models available.")
                 return None # Or raise error

        return optimal_model_name

    def _switch_model(self, new_model_idx, reason, avg_processing_time_before=None):
        """ Handles the logic of switching models and logging the event. """
        old_model_name = self.current_model_name
        new_model_name = self.model_names[new_model_idx]

        event_type = "model_upgrade" if new_model_idx > self.current_model_idx else "model_downgrade"

        logger.info(f"{event_type.replace('_', ' ').title()}: From '{old_model_name}' to '{new_model_name}'. Reason: {reason}")

        self.current_model_idx = new_model_idx
        self.current_model_name = new_model_name
        self.processing_times.clear() # Reset timing data after switch

        # Log adaptation event
        log_data = {
            "event_type": event_type,
            "old_model": old_model_name,
            "new_model": new_model_name,
            "reason": reason,
            "avg_processing_time_before": avg_processing_time_before,
            "target_frame_time": self.target_frame_time
        }
        log_to_es(self.es_client, ADAPTATION_LOG_INDEX, log_data)
        return True


    def try_upgrade_model(self):
        """ Check if we can upgrade to a better (larger/slower) model. """
        current_time = time.monotonic()

        # Check upgrade interval
        if current_time - self.last_upgrade_time < self.upgrade_interval:
            return False

        self.last_upgrade_time = current_time

        # Check if already at the best model
        if self.current_model_idx >= len(self.model_names) - 1:
            logger.debug("Already using the best model, cannot upgrade.")
            return False

        # Check if processing times deque is sufficiently populated
        if len(self.processing_times) < self._DOWNGRADE_CHECK_WINDOW:
             logger.debug("Not enough processing time data to consider upgrade.")
             return False

        avg_processing_time = sum(self.processing_times) / len(self.processing_times)

        # Check if we have enough performance headroom
        if avg_processing_time <= self.target_frame_time * self._UPGRADE_HEADROOM_FACTOR:
            next_model_idx = self.current_model_idx + 1
            return self._switch_model(
                new_model_idx=next_model_idx,
                reason="performance_headroom_available",
                avg_processing_time_before=avg_processing_time
            )
        else:
            logger.debug(f"No upgrade. Avg time: {avg_processing_time:.4f}s vs Target headroom: {self.target_frame_time * self._UPGRADE_HEADROOM_FACTOR:.4f}s")
            return False


    def check_downgrade_model(self, last_processing_time):
        """ Check if we need to downgrade to a smaller/faster model. """
         # Don't downgrade if already using the smallest model
        if self.current_model_idx <= 0:
            return False

        # Check if processing times deque is sufficiently populated
        if len(self.processing_times) < self._DOWNGRADE_CHECK_WINDOW:
            logger.debug("Not enough processing time data to consider downgrade.")
            return False

        recent_avg = sum(self.processing_times) / len(self.processing_times)

        # Check if consistently missing the target FPS (using the threshold factor)
        if recent_avg > self.target_frame_time * self._DOWNGRADE_THRESHOLD_FACTOR:
            new_model_idx = self.current_model_idx - 1
            return self._switch_model(
                new_model_idx=new_model_idx,
                reason="target_fps_missed",
                avg_processing_time_before=recent_avg
            )
        else:
             logger.debug(f"No downgrade. Avg time: {recent_avg:.4f}s vs Target threshold: {self.target_frame_time * self._DOWNGRADE_THRESHOLD_FACTOR:.4f}s")
             return False


    def process_frame(self, frame, frame_number):
        """ Process a single frame using the current model. """
        if self.current_model_name is None or self.current_model_name not in self.models:
             logger.error(f"Cannot process frame {frame_number}: Current model '{self.current_model_name}' is not loaded or selected.")
             return frame # Return original frame on error

        model = self.models[self.current_model_name]
        model_used_for_frame = self.current_model_name # Record model used before potential switches

        logger.debug(f"Frame {frame_number}: Using model '{model_used_for_frame}' for inference")

        start_time = time.monotonic()
        try:
            results = model(frame, verbose=False) # Suppress YOLO's own console logs per frame
            end_time = time.monotonic()
            processing_time = end_time - start_time
            self.processing_times.append(processing_time)

            current_fps = 1.0 / processing_time if processing_time > 0 else float('inf')
            logger.debug(f"Frame {frame_number}: Inference time: {processing_time:.4f}s ({current_fps:.2f} FPS)")

            # Extract metrics (handle potential errors if results format is unexpected)
            try:
                boxes = results[0].boxes
                confidences = boxes.conf.tolist() if hasattr(boxes, 'conf') and boxes.conf is not None else []
                num_detections = len(confidences)
                avg_confidence = sum(confidences) / num_detections if num_detections > 0 else 0.0
            except Exception as e:
                 logger.warning(f"Frame {frame_number}: Could not extract all metrics from results: {e}")
                 num_detections = -1 # Indicate an issue
                 avg_confidence = -1.0

            # Log Metrics to Elasticsearch
            frame_metrics = {
                "frame_number": frame_number,
                "model_name": model_used_for_frame,
                "model_processing_time": processing_time,
                "current_fps": current_fps if current_fps != float('inf') else 0.0, # Log 0 for infinite FPS
                "num_detections": num_detections,
                "avg_confidence": avg_confidence,
                "target_frame_time": self.target_frame_time
            }
            log_to_es(self.es_client, VIDEO_METRICS_INDEX, frame_metrics)

            # Check adaptation logic AFTER logging metrics for the frame
            self.check_downgrade_model(processing_time)
            self.try_upgrade_model() # Try upgrade periodically

            # Render results on the frame
            annotated_frame = results[0].plot() if results else frame

        except Exception as e:
            logger.error(f"Frame {frame_number}: Error during model inference or processing: {e}")
            # Attempt to log error?
            annotated_frame = frame # Return original frame on error

        return annotated_frame


    def process_video(self, input_path, output_path=None):
        """ Process an entire video file adaptively. """
        cap = None
        out = None
        start_time = time.monotonic()
        processed_frame_count = 0
        total_input_frames_read = 0

        try:
            logger.info(f"Starting video processing for: {input_path}")
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {input_path}")

            # Get video properties
            input_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            logger.info(f"Input video properties: Resolution={frame_width}x{frame_height}, Input FPS={input_fps:.2f}, Total Frames={total_frames}")
            logger.info(f"Target output FPS: {self.output_fps:.2f} (Target Frame Time: {self.target_frame_time:.4f}s)")

            # Adjust target FPS if it's higher than input FPS
            if self.output_fps > input_fps > 0:
                logger.warning(f"Target FPS ({self.output_fps:.2f}) is higher than input FPS ({input_fps:.2f}). Adjusting target to input FPS.")
                self.output_fps = input_fps
                self.target_frame_time = 1.0 / self.output_fps

            # --- Initial Model Selection ---
            # Extract sample frames for testing
            test_frames_list = []
            if total_frames > 0:
                 # Ensure we don't sample more frames than available or needed
                num_samples = min(100, total_frames, self.test_frames * len(self.models))
                if num_samples > 0:
                    frame_indices = sorted(random.sample(range(total_frames), num_samples))
                    logger.info(f"Extracting {len(frame_indices)} frames for initial performance testing...")
                    for idx in frame_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            test_frames_list.append(frame)
                        else:
                             logger.warning(f"Could not read frame at index {idx} for testing.")
                    # Reset video capture to the beginning
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                     logger.warning("Not enough frames in video to perform initial testing.")
            else:
                 logger.warning("Could not determine total frames or video is empty. Skipping initial performance testing.")


            # Test performance and select initial model
            performance_data = self.test_model_performance(test_frames_list)
            initial_model_name = self.select_optimal_model(performance_data)

            if initial_model_name is None:
                 logger.error("Failed to select an initial model. Aborting processing.")
                 return # Or raise specific error

            self.current_model_name = initial_model_name
            self.current_model_idx = self.model_names.index(initial_model_name)
            logger.info(f"Selected initial model: {self.current_model_name}")
            # Log initial selection
            log_to_es(self.es_client, ADAPTATION_LOG_INDEX, {
                "event_type": "initial_selection",
                "new_model": self.current_model_name,
                "target_frame_time": self.target_frame_time,
                "reason": "initial_performance_test"
            })


            # --- Setup Output Video Writer ---
            if output_path:
                # Ensure output directory exists
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    logger.info(f"Creating output directory: {output_dir}")
                    os.makedirs(output_dir)

                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or use 'XVID', 'MJPG' etc.
                out_fps = self.output_fps # Output video should have the target FPS
                out = cv2.VideoWriter(output_path, fourcc, out_fps, (frame_width, frame_height))
                if not out.isOpened():
                     logger.error(f"Could not open video writer for output file: {output_path}")
                     output_path = None # Disable writing if it failed
                     # Decide whether to continue without writing or abort


            # --- Frame Processing Loop ---
            # Calculate frame skipping interval based on *input* and *output* FPS
            frame_interval = 1
            if input_fps > 0 and self.output_fps > 0 and input_fps > self.output_fps:
                 frame_interval = max(1, round(input_fps / self.output_fps))
            logger.info(f"Processing approximately every {frame_interval} frame(s) from input.")


            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video file reached.")
                    break

                total_input_frames_read += 1

                # Frame skipping logic
                # Process frame 0, frame `frame_interval`, frame `2*frame_interval`, etc.
                if (total_input_frames_read - 1) % frame_interval != 0:
                    continue # Skip this frame

                processed_frame_count += 1

                # Process the selected frame
                processed_frame = self.process_frame(frame, total_input_frames_read) # Log with original frame number

                # Add informational text to the frame
                try:
                    display_fps = 1.0 / self.processing_times[-1] if self.processing_times else 0.0
                    info_text = f"Model: {self.current_model_name} | Frame: {total_input_frames_read} | Proc FPS: {display_fps:.1f}"
                    cv2.putText(processed_frame, info_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except Exception as text_e:
                     logger.warning(f"Could not draw text on frame: {text_e}")

                # Write to output video
                if out:
                    out.write(processed_frame)

                # Display the frame (optional, can be slow)
                cv2.imshow('Adaptive YOLO Processor', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Processing stopped by user ('q' key press).")
                    break

                # Log progress periodically
                if processed_frame_count % 100 == 0:
                    elapsed = time.monotonic() - start_time
                    avg_fps_so_far = processed_frame_count / elapsed if elapsed > 0 else 0
                    logger.info(f"Processed {processed_frame_count} frames ({total_input_frames_read} total read) | "
                                f"Current model: {self.current_model_name} | "
                                f"Overall Avg Proc FPS: {avg_fps_so_far:.2f}")

            # --- End of Processing ---
            end_process_time = time.monotonic()
            total_duration = end_process_time - start_time
            overall_avg_fps = processed_frame_count / total_duration if total_duration > 0 else 0

            logger.info(f"Video processing finished. Total duration: {total_duration:.2f} seconds.")
            logger.info(f"Processed {processed_frame_count} frames out of {total_input_frames_read} frames read from input.")
            logger.info(f"Final model used: {self.current_model_name}")
            logger.info(f"Overall average processing FPS: {overall_avg_fps:.2f}")

            # Log summary
            summary_log = {
                "event_type": "processing_finished",
                "input_video": input_path,
                "output_video": output_path if output_path else "N/A",
                "total_frames_in_video": total_frames, # Original total frames
                "total_frames_processed": processed_frame_count, # Actual number processed
                "final_model_used": self.current_model_name,
                "target_output_fps": self.output_fps, # The target FPS used
                "processing_duration_seconds": total_duration,
                "overall_average_fps": overall_avg_fps
            }
            log_to_es(self.es_client, ADAPTATION_LOG_INDEX, summary_log)

        except Exception as e:
            logger.exception(f"An error occurred during video processing: {e}") # Logs traceback
            # Log error event?
            log_to_es(self.es_client, ADAPTATION_LOG_INDEX, {
                "event_type": "error",
                "reason": str(e),
                "input_video": input_path,
                "current_frame_number": total_input_frames_read,
                "current_model": self.current_model_name
            })

        finally:
            # Ensure resources are released
            if cap:
                cap.release()
                logger.debug("Input video capture released.")
            if out:
                out.release()
                logger.info(f"Output video saved to: {output_path}")
            cv2.destroyAllWindows()
            logger.debug("OpenCV windows destroyed.")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Adaptive YOLO Video Processor with Elasticsearch Logging")
    parser.add_argument("--input", required=True, help="Path to the input video file.")
    parser.add_argument("--output", required=True, help="Path for the output video file.")
    parser.add_argument("--fps", type=float, required=True, help="Target output FPS.")
    parser.add_argument("--test_frames", type=int, default=30, help="Number of frames per model for initial performance testing.")
    parser.add_argument("--upgrade_interval", type=int, default=20, help="Interval in seconds to check for model upgrade.")
    parser.add_argument("--debug", action="store_true", help="Enable debug level logging.")

    args = parser.parse_args()

    # Adjust logging level if debug flag is set
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.info("Debug logging enabled.")

    logger.info("--- Adaptive Video Processor Script Started ---")
    logger.info(f"Input Video: {args.input}")
    logger.info(f"Output Video: {args.output}")
    logger.info(f"Target FPS: {args.fps}")
    logger.info(f"Initial Test Frames: {args.test_frames}")
    logger.info(f"Upgrade Check Interval: {args.upgrade_interval}s")

    # --- Input Validation ---
    if not os.path.exists(args.input):
        logger.error(f"Input video file not found: {args.input}")
        sys.exit(1) # Exit if input file doesn't exist

    if args.fps <= 0:
        logger.error("Target FPS must be a positive number.")
        sys.exit(1)

    # --- Setup Elasticsearch ---
    es_client_instance = setup_elasticsearch(
        ES_HOST, ES_PORT,
        VIDEO_METRICS_INDEX, VIDEO_METRICS_MAPPING,
        ADAPTATION_LOG_INDEX, ADAPTATION_LOG_MAPPING
    )
    # The script will continue even if es_client_instance is None, but logging will be disabled.

    # --- Define Models ---
    # NOTE: The paths are relative. The script will look for these files in its current directory.
    # The models will be downloaded if they don't exist.
    model_paths = {
        # Ordered from smallest/fastest to largest/slowest
        "yolov5nu": "yolov5nu.pt",  # Nano
        "yolov5su": "yolov5su.pt",  # Small
        "yolov5mu": "yolov5mu.pt",  # Medium
    }
    logger.info(f"Using models (check paths): {list(model_paths.keys())}")


    # --- Create and Run Processor ---\
    try:
        processor = AdaptiveYOLOProcessor(
            model_paths=model_paths,
            output_fps=args.fps,
            test_frames=args.test_frames,
            upgrade_interval=args.upgrade_interval,
            es_client=es_client_instance # Pass the client instance
        )

        processor.process_video(args.input, args.output)

        logger.info("--- Adaptive Video Processor Script Finished ---")

    except ValueError as ve:
         logger.error(f"Configuration or Initialization Error: {ve}")
         sys.exit(1)
    except Exception as main_e:
         logger.exception(f"An unexpected error occurred in the main execution block: {main_e}")
         sys.exit(1)
