#!/usr/bin/env python3
"""
Face Cropper: Crop a single image or video using MediaPipe Face Detection.

Usage examples:

# Crop a single image
python crop_face.py \
  --mode image \
  --image_path "/path/to/input.jpg" \
  --output_image_path "/path/to/output_cropped.jpg"

# Crop a single video
python crop_face.py \
  --mode video \
  --video_path "/path/to/input/video.mp4" \
  --output_video_path "/path/to/output/cropped_video.mp4" \
  --temp_dir "/path/to/temp"
"""

import os
import cv2
import math
import glob
import argparse
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ----------------------------- Frame extraction -----------------------------
def extract_frames_ffmpeg(video_path, output_dir, fps=24):
    os.makedirs(output_dir, exist_ok=True)
    output_pattern = os.path.join(output_dir, "frame_%010d.png")
    ffmpeg_cmd = [
        "ffmpeg", "-i", video_path, "-vf", f"fps={fps}",
        output_pattern, "-hide_banner", "-loglevel", "error"
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    frame_files = sorted(glob.glob(os.path.join(output_dir, "frame_*.png")))
    frame_data = [
        [path, i + 1, i / float(fps)] for i, path in enumerate(frame_files)
    ]
    return pd.DataFrame(frame_data, columns=["frame_path", "frame_index", "frame_timestamp"])


# ----------------------------- Face Detection Helpers -----------------------------
def _normalized_to_pixel_coordinates(x, y, image_width, image_height):
    if not (0 <= x <= 1 and 0 <= y <= 1):
        return None
    return min(int(x * image_width), image_width - 1), min(int(y * image_height), image_height - 1)


def get_bb_from_image(image, detection_result):
    height, width, _ = image.shape
    num_faces = len(detection_result.detections)
    if num_faces != 1:
        return None, None

    detection = detection_result.detections[0]
    bbox = detection.bounding_box
    bbox_dict = {
        "face_bb_origin_x": bbox.origin_x,
        "face_bb_origin_y": bbox.origin_y,
        "face_bb_width": bbox.width,
        "face_bb_height": bbox.height,
        "num_faces": num_faces
    }
    keypoints_list = [
        _normalized_to_pixel_coordinates(kp.x, kp.y, width, height)
        for kp in detection.keypoints
    ]
    return bbox_dict, keypoints_list


def get_bb_for_video(frames_df, model_path="detector.tflite"):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    bbox_data = {k: [] for k in ["x", "y", "w", "h", "faces"]}
    kp_save = []

    for _, row in tqdm(frames_df.iterrows(), total=len(frames_df)):
        image = mp.Image.create_from_file(row["frame_path"])
        detection_result = detector.detect(image)
        det_bbox, det_kp_list = get_bb_from_image(image.numpy_view(), detection_result)
        if det_bbox:
            bbox_data["x"].append(det_bbox["face_bb_origin_x"])
            bbox_data["y"].append(det_bbox["face_bb_origin_y"])
            bbox_data["w"].append(det_bbox["face_bb_width"])
            bbox_data["h"].append(det_bbox["face_bb_height"])
            bbox_data["faces"].append(det_bbox["num_faces"])
            kp_save.append({"frame_index": row["frame_index"], "keypoints": det_kp_list})
        else:
            for key in bbox_data:
                bbox_data[key].append(np.nan)
            kp_save.append(None)

    frames_df["face_bb_origin_x"] = bbox_data["x"]
    frames_df["face_bb_origin_y"] = bbox_data["y"]
    frames_df["face_bb_width"] = bbox_data["w"]
    frames_df["face_bb_height"] = bbox_data["h"]
    frames_df["num_detected_faces"] = bbox_data["faces"]

    return frames_df, kp_save


# ----------------------------- Cropping Helpers -----------------------------
def get_cropped_image_from_bb(image_path, output_image_path,
                              face_bb_width, face_bb_height,
                              face_bb_origin_x, face_bb_origin_y,
                              margin_pct=0, output_size=(256, 256)):
    if any(map(lambda v: v is None or (isinstance(v, float) and math.isnan(v)),
               [face_bb_width, face_bb_height, face_bb_origin_x, face_bb_origin_y])):
        return None

    image = cv2.imread(image_path)
    height, width, _ = image.shape

    MARGIN_X = int((margin_pct / 100.) * face_bb_width)
    MARGIN_Y = int((margin_pct / 100.) * face_bb_height)
    MARGIN_X = min(MARGIN_X, face_bb_origin_x, width - face_bb_origin_x - face_bb_width)
    MARGIN_Y = min(MARGIN_Y, face_bb_origin_y, height - face_bb_origin_y - face_bb_height)
    MARGIN = min(MARGIN_X, MARGIN_Y)

    avg_size = (face_bb_width + face_bb_height) / 2
    start_x = int(max(0, face_bb_origin_x - MARGIN))
    start_y = int(max(0, face_bb_origin_y - MARGIN))
    end_x = int(min(face_bb_origin_x + avg_size + MARGIN, width - 1))
    end_y = int(min(face_bb_origin_y + avg_size + MARGIN, height - 1))

    cropped = image[start_y:end_y, start_x:end_x]
    resized = cv2.resize(cropped, output_size)
    cv2.imwrite(output_image_path, resized)
    return output_image_path


# ----------------------------- Process Video -----------------------------
def process_video_frames(frames_df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    frames_df, kp_save_list = get_bb_for_video(frames_df)
    cropped_paths = []

    for _, row in tqdm(frames_df.iterrows(), total=len(frames_df)):
        out_path = os.path.join(save_dir, f"{row['frame_index']:010d}.png")
        cropped = get_cropped_image_from_bb(
            row["frame_path"], out_path,
            row["face_bb_width"], row["face_bb_height"],
            row["face_bb_origin_x"], row["face_bb_origin_y"]
        )
        cropped_paths.append(cropped)
    frames_df["cropped_frame_path"] = cropped_paths
    return frames_df, kp_save_list


def join_video_ffmpeg_wo_audio(frames_df, output_path):
    valid_paths = frames_df["cropped_frame_path"].dropna()
    if valid_paths.empty:
        print("❌ No valid cropped frames found.")
        return
    cropped_frame_par_path = "/".join(valid_paths.iloc[0].split("/")[:-1])
    ffmpeg_command = [
        "ffmpeg", "-framerate", "24", "-i", f"{cropped_frame_par_path}/%010d.png",
        "-c:v", "libx264", "-r", "24", "-pix_fmt", "yuv420p", output_path,
        "-hide_banner", "-loglevel", "error", "-y"
    ]
    subprocess.run(ffmpeg_command, check=True)


def crop_single_video(video_path, output_video_path, temp_dir):
    os.makedirs(temp_dir, exist_ok=True)

    print("🎬 Extracting frames...")
    frames_df = extract_frames_ffmpeg(video_path, os.path.join(temp_dir, "raw_frames"))

    print("🧠 Detecting and cropping faces...")
    cropped_frames_df, _ = process_video_frames(frames_df, os.path.join(temp_dir, "cropped_frames"))

    na_fraction = cropped_frames_df["cropped_frame_path"].isna().mean()

    if na_fraction > 0.3:
        print("❌ Too many failed crops. Skipping video.")
        return

    print("🎥 Joining frames into output video...")
    join_video_ffmpeg_wo_audio(cropped_frames_df, output_video_path)
    print(f"✅ Cropped video saved at: {output_video_path}")


# ----------------------------- Process Single Image -----------------------------
def crop_single_image(image_path, output_image_path, model_path="detector.tflite"):
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(image)
    det_bbox, _ = get_bb_from_image(image.numpy_view(), detection_result)
    if not det_bbox:
        print("❌ No face detected.")
        return
    cropped = get_cropped_image_from_bb(
        image_path, output_image_path,
        det_bbox["face_bb_width"], det_bbox["face_bb_height"],
        det_bbox["face_bb_origin_x"], det_bbox["face_bb_origin_y"]
    )
    if cropped:
        print(f"✅ Cropped image saved at: {output_image_path}")
    else:
        print("❌ Cropping failed.")


# ----------------------------- CLI Entry -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Crop a single image or video using Mediapipe face detection.")
    parser.add_argument("--mode", choices=["image", "video"], required=True, help="Choose between 'image' or 'video' mode.")

    # For image mode
    parser.add_argument("--image_path", help="Path to input image.")
    parser.add_argument("--output_image_path", help="Path to save cropped image.")

    # For video mode
    parser.add_argument("--video_path", help="Path to input video file.")
    parser.add_argument("--output_video_path", help="Path to save cropped output video.")
    parser.add_argument("--temp_dir", help="Temporary directory for intermediate frames.")

    args = parser.parse_args()

    if args.mode == "image":
        if not args.image_path or not args.output_image_path:
            parser.error("--image_path and --output_image_path are required for image mode.")
        crop_single_image(args.image_path, args.output_image_path)

    elif args.mode == "video":
        if not all([args.video_path, args.output_video_path, args.temp_dir]):
            parser.error("--video_path, --output_video_path, and --temp_dir are required for video mode.")
        crop_single_video(args.video_path, args.output_video_path, args.temp_dir)


if __name__ == "__main__":
    main()
