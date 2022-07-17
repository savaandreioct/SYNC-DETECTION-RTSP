from random import random
import torch
import cv2
from threading import Thread
import time
import sys
from stream import Stream
from typing import Any, List
from structlog import get_logger
import numpy as np
from datetime import datetime
from util import load_file_as_list


class Detector:
    def __init__(self, stream: Stream, models: List[str]):
        self.stream = stream
        self.models = models
        self.loaded_models = self.load_models()
        self.classes = load_file_as_list("classes.txt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using Device: ", self.device)

    def _load_model(self, model_name: str) -> Any:
        """
        Loads Yolo5 model from pytorch hub
        """
        # TODO check if model is not here
        return torch.hub.load(
            "ultralytics/yolov5", "custom", path=model_name, force_reload=True
        )

    def load_models(self) -> List[Any]:
        """
        Loads every model from pytorch hub and return a list with them loaded.
        """

        loaded_models = []
        for model in self.models:
            loaded_models.append(self._load_model(model))

        return loaded_models

    def init_frame_dict(self) -> dict:
        """
        Init a dict used to save predicted frames for every model.
        Dict structure:
        {
            model_name: predicted_frame
        }
        """

        frame_dict = {}
        for name in self.models:
            frame_dict.update({name: None})

        return frame_dict

    def set_number_of_threads(self) -> int:
        """
        Set the number of frames to be number of models.
        """
        get_logger().info("Set number of threads", number_of_threads=len(self.models))
        return len(self.models)

    def class_to_label(self, x: int) -> str:
        """
        For a given label value, return corresponding string label.
        """
        return self.classes[int(x)]

    def detect(self, frame: np.ndarray, index: int):
        """
        Takes frame and model index number and calculate corresponding label and cord.
        """
        self.loaded_models[index].to(self.device)
        results = self.loaded_models[index]([frame])
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def plot_results(
        self, frame: np.ndarray, labels: torch.Tensor, cord: torch.Tensor
    ) -> np.ndarray:
        """
        Takes a frame, its label and cord and plots the bounding boxes and label on to the frame.
        """

        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(len(labels)):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = (
                    int(row[0] * x_shape),
                    int(row[1] * y_shape),
                    int(row[2] * x_shape),
                    int(row[3] * y_shape),
                )
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(
                    frame,
                    self.class_to_label(labels[i]),
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    bgr,
                    2,
                )

        return frame

    def predict(self, frame: np.ndarray, index: int, frame_dict: dict) -> None:
        """
        For every frame, call the detect and plot methods and update the internal
        dict used to save results.
        """

        labels, cord = self.detect(frame, index)
        frame = self.plot_results(frame, labels, cord)
        frame_dict.update({self.models[index]: frame})

    def __call__(self) -> None:
        """
        Read the video frame by frame, and display the output for every model.
        Called when class is executed.
        """

        stream = self.stream.get()
        self.stream.check_stream(stream=stream, retry=3, timeout=3.05)
        for name in self.models:
            get_logger().info("Creating window", model_name=name)
            cv2.namedWindow(f"YOLOv5 Detection {name}", cv2.WINDOW_NORMAL)

        num_of_threads = self.set_number_of_threads()

        while stream.isOpened():
            frame = cv2.resize(self.stream.read(stream=stream), (640, 640))
            threads = list()
            frame_dict = self.init_frame_dict()
            for i in range(num_of_threads):
                get_logger().info(
                    f"Creating thread", thread_number=i, model_name=self.models[i]
                )
                x = Thread(target=self.predict, args=(frame, i, frame_dict))
                threads.append(x)
                x.start()
            for i, thread in enumerate(threads):
                thread.join()

            for name in self.models:
                get_logger().info(
                    "Displaying results",
                    model_name=name,
                    datetime=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                )
                cv2.imshow(f"YOLOv5 Detection {name}", frame_dict.get(name))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                get_logger().info("Bye-Bye")
                break

        self.stream.stop(stream)
