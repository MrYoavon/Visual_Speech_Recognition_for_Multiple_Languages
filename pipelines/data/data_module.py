#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torchaudio
import numpy as np
import cv2
from .transforms import AudioTransform, VideoTransform

class AVSRDataLoader:
    def __init__(self, modality, speed_rate=1, transform=True, detector="retinaface", convert_gray=True):
        self.modality = modality
        self.transform = transform
        if self.modality in ["audio", "audiovisual"]:
            self.audio_transform = AudioTransform()
        if self.modality in ["video", "audiovisual"]:
            if detector == "mediapipe":
                from mpc001.pipelines.detectors.mediapipe.video_process import VideoProcess
                self.video_process = VideoProcess(convert_gray=convert_gray)
            elif detector == "retinaface":
                from mpc001.pipelines.detectors.retinaface.video_process import VideoProcess
                self.video_process = VideoProcess(convert_gray=convert_gray)
            self.video_transform = VideoTransform(speed_rate=speed_rate)

    def load_data(self, video_frames=None, audio_filename=None, landmarks=None, transform=True):
        """
        Load data based on the selected modality.

        For modality "video", video_frames should be a list of frames (e.g., numpy arrays)
        from the WebRTC track.
        For modality "audio", audio_filename (path to the audio file) is required.
        For modality "audiovisual", provide both video_frames and audio_filename.
        """
        if self.modality == "audio":
            if audio_filename is None:
                raise ValueError("audio_filename must be provided for audio modality")
            audio, sample_rate = self.load_audio(audio_filename)
            audio = self.audio_process(audio, sample_rate)
            return self.audio_transform(audio) if self.transform else audio

        elif self.modality == "video":
            if video_frames is None:
                raise ValueError("video_frames must be provided for video modality")
            video = self.load_video(video_frames)
            video = self.video_process(video, landmarks)
            if video is None:
                return None
            video = torch.tensor(video)
            return self.video_transform(video) if self.transform else video

        elif self.modality == "audiovisual":
            if video_frames is None or audio_filename is None:
                raise ValueError("Both video_frames and audio_filename must be provided for audiovisual modality")
            rate_ratio = 640
            audio, sample_rate = self.load_audio(audio_filename)
            audio = self.audio_process(audio, sample_rate)
            video = self.load_video(video_frames)
            video = self.video_process(video, landmarks)
            video = torch.tensor(video)
            # Align audio and video lengths
            min_t = min(len(video), audio.size(1) // rate_ratio)
            audio = audio[:, :min_t * rate_ratio]
            video = video[:min_t]
            if self.transform:
                audio = self.audio_transform(audio)
                video = self.video_transform(video)
            return video, audio

    def load_audio(self, data_filename):
        waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
        return waveform, sample_rate

    def load_video(self, frame_buffer):
        """
        Load video from a list of frames.

        Args:
            frame_buffer (list): A list of frames (e.g., numpy arrays) from the WebRTC track.

        Returns:
            numpy.ndarray: A stacked numpy array with shape (num_frames, H, W, C)
        """
        def resize_with_padding(frame, target_size=(128, 128)):
            target_w, target_h = target_size
            h, w = frame.shape[:2]
            # Compute scale factor to fit the frame inside target_size while preserving aspect ratio.
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            # Resize the frame.
            resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # Create a blank image of target_size.
            padded_frame = np.zeros((target_h, target_w, frame.shape[2]), dtype=frame.dtype)
            # Compute offsets to center the resized frame.
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            padded_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
            return padded_frame

        processed_frames = [resize_with_padding(frame, target_size=(192, 192)) for frame in frame_buffer]
        return np.stack(processed_frames, axis=0)

    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform
