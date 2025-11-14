import cv2
import time
import subprocess
import random
import numpy as np
import os
import sys
import requests
import datetime
import logging
from collections import deque
from ultralytics import YOLO
import argparse
from elasticsearch import Elasticsearch, ElasticsearchException

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
# These are identical to the VideoSwitch.py script
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
            timeout=10,
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
                    es_client.indices.create(index=index_name, body=index_mapping, ignore=400)
                    logger.info(f"Created Elasticsearch index '{index_name}'")
                except ElasticsearchException as create_err:
                    if not es_client.indices.exists(index=index_name):
                         logger.error(f"Failed to create index '{index_name}': {create_err}")
                    else:
                         logger.warning(f"Index '{index_name}' already existed (likely race condition).")
             else:
                 logger.info(f"Elasticsearch index '{index_name}' already exists.")

        return es_client

    except ElasticsearchException as e:
        logger.error(f"Failed to connect or setup Elasticsearch: {e}. Logging will be disabled.")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during Elasticsearch setup: {e}. Logging disabled.")
        return None

# --- Elasticsearch Logging Function ---
def log_to_es(client, index_name, data_dict):
    """ Logs a dictionary to the specified Elasticsearch index. """
    if client:
        try:
            data_dict['timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            response = client.index(index=index_name, body=data_dict)
        except ElasticsearchException as e:
            logger.warning(f"Failed to log to Elasticsearch index '{index_name}': {e}.")
        except Exception as e:
            logger.error(f"Unexpected error during logging to ES index '{index_name}': {e}.")

# --- Adaptive Processor Class ---
class AdaptiveYOLOProcessor:
    # Constants from VideoSwitch.py
    _TIMING_DEQUE_LEN = 100
    _DOWNGRADE_CHECK_WINDOW = 5
    _UPGRADE_HEADROOM_FACTOR = 0.8
    _DOWNGRADE_THRESHOLD_FACTOR = 1.1

    def __init__(self, model_paths, output_fps, upgrade_interval=60, es_client=None):
        self.model_paths = model_paths
        self.output_fps = output_fps
        self.target_frame_time = 1.0 / output_fps if output_fps > 0 else float('inf')
        self.upgrade_interval = upgrade_interval
        self.es_client = es_client # Store ES client
        self.last_upgrade_time = 0

        self.models = {}
        for name, path in model_paths.items():
            try:
                self.models[name] = YOLO(path)
                logger.info(f"Successfully loaded model: {name} from {path}")
            except Exception as e:
                logger.error(f"Failed to load model '{name}' from {path}: {e}")

        if not self.models:
            raise ValueError("No YOLO models could be loaded. Please check paths and model files.")

        self.model_names = [name for name in model_paths if name in self.models]
        if not self.model_names:
             raise ValueError("Model loading succeeded but no model names available. Check model_paths keys.")

        self.current_model_idx = 0
        self.current_model_name = self.model_names[self.current_model_idx]
        logger.info(f"Starting with initial model: {self.current_model_name}")

        self.processing_times = deque(maxlen=self._TIMING_DEQUE_LEN)

        # Log initial selection
        log_to_es(self.es_client, ADAPTATION_LOG_INDEX, {
            "event_type": "initial_selection",
            "new_model": self.current_model_name,
            "target_frame_time": self.target_frame_time,
            "reason": "live_stream_initial_selection"
        })

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
        current_time = time.monotonic()
        if current_time - self.last_upgrade_time < self.upgrade_interval:
            return False

        self.last_upgrade_time = current_time
        if self.current_model_idx >= len(self.model_names) - 1:
            logger.debug("Already using the best model, cannot upgrade.")
            return False

        if len(self.processing_times) < self._DOWNGRADE_CHECK_WINDOW:
             logger.debug("Not enough processing time data to consider upgrade.")
             return False

        avg_processing_time = sum(self.processing_times) / len(self.processing_times)

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

    def check_downgrade_model(self):
        if self.current_model_idx <= 0:
            return False

        if len(self.processing_times) < self._DOWNGRADE_CHECK_WINDOW:
            logger.debug("Not enough processing time data to consider downgrade.")
            return False

        recent_avg = sum(self.processing_times) / len(self.processing_times)

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
        if self.current_model_name is None or self.current_model_name not in self.models:
             logger.error(f"Cannot process frame {frame_number}: Current model '{self.current_model_name}' is not loaded or selected.")
             return frame

        model = self.models[self.current_model_name]
        model_used_for_frame = self.current_model_name

        start_time = time.monotonic()
        try:
            results = model(frame, verbose=False)
            end_time = time.monotonic()
            processing_time = end_time - start_time
            self.processing_times.append(processing_time)

            current_fps = 1.0 / processing_time if processing_time > 0 else float('inf')

            try:
                boxes = results[0].boxes
                confidences = boxes.conf.tolist() if hasattr(boxes, 'conf') and boxes.conf is not None else []
                num_detections = len(confidences)
                avg_confidence = sum(confidences) / num_detections if num_detections > 0 else 0.0
            except Exception as e:
                 logger.warning(f"Frame {frame_number}: Could not extract all metrics from results: {e}")
                 num_detections = -1
                 avg_confidence = -1.0

            # Log Metrics to Elasticsearch
            frame_metrics = {
                "frame_number": frame_number,
                "model_name": model_used_for_frame,
                "model_processing_time": processing_time,
                "current_fps": current_fps if current_fps != float('inf') else 0.0,
                "num_detections": num_detections,
                "avg_confidence": avg_confidence,
                "target_frame_time": self.target_frame_time
            }
            log_to_es(self.es_client, VIDEO_METRICS_INDEX, frame_metrics)

            self.check_downgrade_model()
            self.try_upgrade_model()

            annotated_frame = results[0].plot() if results else frame

            # Add informational text to the frame
            info_text = f"Model: {self.current_model_name} | Frame: {frame_number} | Proc FPS: {current_fps:.1f}"
            cv2.putText(annotated_frame, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception as e:
            logger.error(f"Frame {frame_number}: Error during model inference or processing: {e}")
            annotated_frame = frame

        return annotated_frame


    def process_rtsp_stream(self, input_rtsp, output_rtsp):
        cap = cv2.VideoCapture(input_rtsp)
        if not cap.isOpened():
            raise ValueError(f"Could not open RTSP stream: {input_rtsp}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"Input Stream: {input_rtsp}")
        logger.info(f"Resolution: {frame_width}x{frame_height} @ {source_fps:.2f} FPS")
        logger.info(f"Target Processing FPS: {self.output_fps}")
        logger.info(f"Output Stream: {output_rtsp}")

        if source_fps > 0 and self.output_fps > source_fps:
            logger.error(f"Target output FPS ({self.output_fps}) cannot be greater than the input stream's FPS ({source_fps:.2f}).")
            cap.release()
            return

        ffmpeg_command = [
            'ffmpeg', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{frame_width}x{frame_height}', '-r', str(self.output_fps), '-i', '-',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'ultrafast',
            '-tune', 'zerolatency', '-rtsp_transport', 'tcp', '-f', 'rtsp', output_rtsp
        ]
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        processed_frame_count = 0
        total_input_frames_read = 0
        start_time = time.monotonic()
        last_frame_time = time.monotonic()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to grab frame. Reconnecting...")
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(input_rtsp)
                    continue

                total_input_frames_read += 1
                processed_frame_count += 1

                annotated_frame = self.process_frame(frame, processed_frame_count)

                try:
                    ffmpeg_process.stdin.write(annotated_frame.tobytes())
                except (BrokenPipeError, IOError):
                    logger.error("FFmpeg process has closed. Exiting.")
                    break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Processing stopped by user ('q' key press).")
                    break

                wait_time = max(0, self.target_frame_time - (time.monotonic() - last_frame_time))
                time.sleep(wait_time)
                last_frame_time = time.monotonic()

                if processed_frame_count % 100 == 0:
                    elapsed = time.monotonic() - start_time
                    avg_fps_so_far = processed_frame_count / elapsed if elapsed > 0 else 0
                    logger.info(f"Processed {processed_frame_count} frames | Current model: {self.current_model_name} | Overall Avg Proc FPS: {avg_fps_so_far:.2f}")

        except Exception as e:
            logger.exception(f"An error occurred during RTSP stream processing: {e}")
            log_to_es(self.es_client, ADAPTATION_LOG_INDEX, {
                "event_type": "error",
                "reason": str(e),
                "input_video": input_rtsp,
                "current_frame_number": processed_frame_count,
                "current_model": self.current_model_name
            })
        finally:
            end_process_time = time.monotonic()
            total_duration = end_process_time - start_time
            overall_avg_fps = processed_frame_count / total_duration if total_duration > 0 else 0

            logger.info("RTSP stream processing finished.")
            logger.info(f"Processed {processed_frame_count} frames. Total duration: {total_duration:.2f}s.")
            logger.info(f"Final model used: {self.current_model_name}")
            logger.info(f"Overall average processing FPS: {overall_avg_fps:.2f}")

            summary_log = {
                "event_type": "processing_finished",
                "input_video": input_rtsp,
                "output_video": output_rtsp,
                "total_frames_in_video": -1, # Live stream, no known total
                "total_frames_processed": processed_frame_count,
                "final_model_used": self.current_model_name,
                "target_output_fps": self.output_fps,
                "processing_duration_seconds": total_duration,
                "overall_average_fps": overall_avg_fps
            }
            log_to_es(self.es_client, ADAPTATION_LOG_INDEX, summary_log)

            if cap: cap.release()
            if ffmpeg_process:
                if ffmpeg_process.stdin: ffmpeg_process.stdin.close()
                ffmpeg_process.wait(timeout=5)
            cv2.destroyAllWindows()
            logger.debug("Resources released.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Adaptive YOLO RTSP Processor with Elasticsearch Logging")
    parser.add_argument("--input", default="rtsp://localhost:8554/mystream", help="RTSP URL of the input video stream.")
    parser.add_argument("--output", default="rtsp://localhost:8554/annotated", help="RTSP URL for the output video stream.")
    parser.add_argument("--fps", type=float, default=15, help="Target output FPS.")
    parser.add_argument("--upgrade_interval", type=int, default=20, help="Interval in seconds to check for model upgrade.")
    parser.add_argument("--debug", action="store_true", help="Enable debug level logging.")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.info("Debug logging enabled.")

    logger.info("--- Adaptive RTSP Processor Script Started ---")
    logger.info(f"Input Stream: {args.input}")
    logger.info(f"Output Stream: {args.output}")
    logger.info(f"Target FPS: {args.fps}")
    logger.info(f"Upgrade Check Interval: {args.upgrade_interval}s")

    if args.fps <= 0:
        logger.error("Target FPS must be a positive number.")
        sys.exit(1)

    es_client_instance = setup_elasticsearch(
        ES_HOST, ES_PORT,
        VIDEO_METRICS_INDEX, VIDEO_METRICS_MAPPING,
        ADAPTATION_LOG_INDEX, ADAPTATION_LOG_MAPPING
    )

    model_paths = {
        "yolov5nu": "yolov5nu.pt",
        "yolov5su": "yolov5su.pt",
        "yolov5mu": "yolov5mu.pt",
    }
    logger.info(f"Using models (check paths): {list(model_paths.keys())}")


    try:
        processor = AdaptiveYOLOProcessor(
            model_paths=model_paths,
            output_fps=args.fps,
            upgrade_interval=args.upgrade_interval,
            es_client=es_client_instance
        )

        processor.process_rtsp_stream(args.input, args.output)

        logger.info("--- Adaptive RTSP Processor Script Finished ---")

    except ValueError as ve:
         logger.error(f"Configuration or Initialization Error: {ve}")
         sys.exit(1)
    except Exception as main_e:
         logger.exception(f"An unexpected error occurred in the main execution block: {main_e}")
         sys.exit(1)