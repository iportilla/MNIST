# coding: utf-8

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
FILE: sample_face_detection.py

DESCRIPTION:
    This sample demonstrates how to detect faces and analyze faces from an image or binary data.

USAGE:
    python sample_face_detection.py

    Set the environment variables with your own values before running this sample:
    1) AZURE_FACE_API_ENDPOINT - the endpoint to your Face resource.
    2) AZURE_FACE_API_ACCOUNT_KEY - your Face API key.
"""
import json
import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def beautify_json(obj):
    return json.dumps(obj, indent=4)


def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


IMAGE_PATH = Path(__file__).parent / "tomc1.jpeg"


class DetectFaces:
    def __init__(self):
        load_dotenv(find_dotenv())
        self.endpoint = os.getenv("AZURE_FACE_API_ENDPOINT")
        self.key = os.getenv("AZURE_FACE_API_ACCOUNT_KEY")
        self.logger = get_logger("sample_face_detection")

    def detect(self):
        from azure.core.credentials import AzureKeyCredential
        from azure.ai.vision.face import FaceClient
        from azure.ai.vision.face.models import (
            FaceDetectionModel,
            FaceRecognitionModel,
            FaceAttributeTypeDetection03,
            FaceAttributeTypeRecognition04,
        )

        with FaceClient(endpoint=self.endpoint, credential=AzureKeyCredential(self.key)) as face_client:
            sample_file_path = IMAGE_PATH
            result = face_client.detect(
                IMAGE_PATH.read_bytes(),
                detection_model=FaceDetectionModel.DETECTION03,
                recognition_model=FaceRecognitionModel.RECOGNITION04,
                return_face_id=False,
                return_face_attributes=[
                    FaceAttributeTypeDetection03.BLUR,
                    FaceAttributeTypeDetection03.HEAD_POSE,
                    FaceAttributeTypeDetection03.MASK,
                    FaceAttributeTypeRecognition04.QUALITY_FOR_RECOGNITION,
                ],
                return_face_landmarks=True,
                return_recognition_model=True,
                face_id_time_to_live=120,
            )

            self.logger.info(f"Detect faces from the file: {sample_file_path}")
            for idx, face in enumerate(result):
                self.logger.info(f"----- Detection result: #{idx+1} -----")
                self.logger.info(f"Face: {beautify_json(face.as_dict())}")

    def detect_from_url(self):
        from azure.core.credentials import AzureKeyCredential
        from azure.ai.vision.face import FaceClient
        from azure.ai.vision.face.models import (
            FaceDetectionModel,
            FaceRecognitionModel,
            FaceAttributeTypeDetection01,
        )

        with FaceClient(endpoint=self.endpoint, credential=AzureKeyCredential(self.key)) as face_client:
            sample_url = "https://learn.microsoft.com/azure/ai-services/computer-vision/media/quickstarts/presentation.png"
            result = face_client.detect_from_url(
                url=sample_url,
                detection_model=FaceDetectionModel.DETECTION01,
                recognition_model=FaceRecognitionModel.RECOGNITION04,
                return_face_id=False,
                return_face_attributes=[
                    FaceAttributeTypeDetection01.ACCESSORIES,
                    FaceAttributeTypeDetection01.EXPOSURE,
                    FaceAttributeTypeDetection01.GLASSES,
                    FaceAttributeTypeDetection01.NOISE,
                ],
            )

            self.logger.info(f"Detect faces from the url: {sample_url}")
            for idx, face in enumerate(result):
                self.logger.info(f"----- Detection result: #{idx+1} -----")
                self.logger.info(f"Face: {beautify_json(face.as_dict())}")


if __name__ == "__main__":
    sample = DetectFaces()
    sample.detect()
    sample.detect_from_url()
