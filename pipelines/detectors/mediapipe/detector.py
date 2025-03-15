#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import warnings
import torchvision
import mediapipe as mp
import os
import cv2
import numpy as np


class LandmarksDetector:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.short_range_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0)
        self.full_range_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=1)

    def __call__(self, video_input):
        """
        Accepts either a video filename (str) or a list/array of frames.
        Returns the detected landmarks for each frame.
        """
        if isinstance(video_input, str):
            # Input is a video file: read frames from file.
            video_frames = torchvision.io.read_video(video_input, pts_unit='sec')[0].numpy()
        elif isinstance(video_input, list) or isinstance(video_input, np.ndarray):
            # Input is already a list or an array of frames.
            video_frames = video_input
        else:
            raise ValueError("Unsupported input type for video_input. Must be a filename or a list/array of frames.")

        landmarks = self.detect(video_frames, self.short_range_detector)

        if all(l is None for l in landmarks):
            # Skip inference by returning None.
            import logging
            logging.warning("No face detected in any frames. Skipping inference for this sequence.")
            return None
        
        return landmarks

    def detect(self, video_frames, detector):
        """
        Iterates over the provided frames and detects face landmarks using the specified detector.
        Returns a list of landmarks (or None for frames where detection failed).
        """
        landmarks = []
        for frame in video_frames:
            results = detector.process(frame)
            if not results.detections:
                landmarks.append(None)
                continue
            face_points = []
            max_id = 0
            max_size = 0
            for idx, detected_face in enumerate(results.detections):
                bboxC = detected_face.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                bbox = (
                    int(bboxC.xmin * iw),
                    int(bboxC.ymin * ih),
                    int(bboxC.width * iw),
                    int(bboxC.height * ih)
                )
                bbox_size = (bbox[2] - bbox[0]) + (bbox[3] - bbox[1])
                if bbox_size > max_size:
                    max_id, max_size = idx, bbox_size
                lmx = [
                    [int(detected_face.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(0).value].x * iw),
                     int(detected_face.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(0).value].y * ih)],
                    [int(detected_face.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(1).value].x * iw),
                     int(detected_face.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(1).value].y * ih)],
                    [int(detected_face.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(2).value].x * iw),
                     int(detected_face.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(2).value].y * ih)],
                    [int(detected_face.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(3).value].x * iw),
                     int(detected_face.location_data.relative_keypoints[self.mp_face_detection.FaceKeyPoint(3).value].y * ih)]
                ]
                face_points.append(lmx)
            landmarks.append(np.array(face_points[max_id]))
        return landmarks
