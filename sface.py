# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021,
# Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.

import numpy as np
import cv2 as cv

class SFace:
    def __init__(self, modelPath, disType=0, backendId=0, targetId=0):
        self._modelPath = modelPath
        self._backendId = backendId
        self._targetId = targetId

        # Load SFace model
        self._model = cv.FaceRecognizerSF.create(
            model=self._modelPath,
            config="",
            backend_id=self._backendId,
            target_id=self._targetId
        )

        # 0 = cosine similarity
        # 1 = L2 distance
        self._disType = disType
        assert self._disType in [0, 1], "0: Cosine similarity, 1: L2 distance"

        # Thresholds from OpenCV Zoo
        self._threshold_cosine = 0.363
        self._threshold_norml2 = 1.128

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self._model = cv.FaceRecognizerSF.create(
            model=self._modelPath,
            config="",
            backend_id=self._backendId,
            target_id=self._targetId
        )

    # Preprocess image using YuNet bounding box
    def _preprocess(self, image, bbox):
        if bbox is None:
            return image
        else:
            return self._model.alignCrop(image, bbox)

    # Extract embedding vector
    def infer(self, image, bbox=None):
        aligned = self._preprocess(image, bbox)
        features = self._model.feature(aligned)
        return features

    # Compare two faces
    def match(self, image1, face1, image2, face2):
        feature1 = self.infer(image1, face1)
        feature2 = self.infer(image2, face2)

        if self._disType == 0:  # cosine
            score = self._model.match(feature1, feature2, self._disType)
            return score, 1 if score >= self._threshold_cosine else 0

        else:  # L2
            dist = self._model.match(feature1, feature2, self._disType)
            return dist, 1 if dist <= self._threshold_norml2 else 0
