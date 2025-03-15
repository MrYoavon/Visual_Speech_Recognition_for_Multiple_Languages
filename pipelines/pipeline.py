#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import torch
import pickle
from configparser import ConfigParser

from mpc001.pipelines.model import AVSR
from mpc001.pipelines.data.data_module import AVSRDataLoader


class InferencePipeline(torch.nn.Module):
    def __init__(self, config_filename, detector="retinaface", face_track=False, device="cuda:0"):
        super(InferencePipeline, self).__init__()
        assert os.path.isfile(config_filename), f"config_filename: {config_filename} does not exist."

        config = ConfigParser()
        config.read(config_filename)

        # modality configuration
        modality = config.get("input", "modality")
        self.modality = modality

        # data configuration
        input_v_fps = config.getfloat("input", "v_fps")
        model_v_fps = config.getfloat("model", "v_fps")

        # model configuration
        model_path = config.get("model", "model_path")
        model_conf = config.get("model", "model_conf")

        # language model configuration
        rnnlm = config.get("model", "rnnlm")
        rnnlm_conf = config.get("model", "rnnlm_conf")
        penalty = config.getfloat("decode", "penalty")
        ctc_weight = config.getfloat("decode", "ctc_weight")
        lm_weight = config.getfloat("decode", "lm_weight")
        beam_size = config.getint("decode", "beam_size")

        self.dataloader = AVSRDataLoader(modality, speed_rate=input_v_fps/model_v_fps, detector=detector)
        self.model = AVSR(modality, model_path, model_conf, rnnlm, rnnlm_conf, penalty, ctc_weight, lm_weight, beam_size, device)

        if face_track and self.modality in ["video", "audiovisual"]:
            if detector == "mediapipe":
                from mpc001.pipelines.detectors.mediapipe.detector import LandmarksDetector
                self.landmarks_detector = LandmarksDetector()
            elif detector == "retinaface":
                from mpc001.pipelines.detectors.retinaface.detector import LandmarksDetector
                self.landmarks_detector = LandmarksDetector(device="cuda:0")
        else:
            self.landmarks_detector = None

    def process_landmarks(self, video_frames, landmarks_filename=None):
        """
        Process landmarks for a list of video frames. If a landmarks file is provided,
        it will be loaded; otherwise, the detector is used directly.
        """
        if self.modality in ["video", "audiovisual"]:
            if landmarks_filename is not None and os.path.isfile(landmarks_filename):
                with open(landmarks_filename, "rb") as f:
                    landmarks = pickle.load(f)
            else:
                # Assuming the detector can process a list of frames directly.
                landmarks = self.landmarks_detector(video_frames)
            return landmarks
        return None

    def forward_buffer(self, video_frames, landmarks_filename=None):
        """
        A custom forward method that accepts a list of video frames (frame buffer)
        instead of a filename.
        """
        if self.modality not in ["video", "audiovisual"]:
            raise ValueError("forward_buffer is only applicable for video or audiovisual modalities")

        landmarks = self.process_landmarks(video_frames, landmarks_filename)
        data = self.dataloader.load_data(video_frames=video_frames, landmarks=landmarks)
        if data is None:
            return None
        transcript = self.model.infer(data)
        return transcript
