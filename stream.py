import sys
from time import sleep
from typing import Union
import cv2
from structlog import get_logger
import numpy as np


class Stream:
    def __init__(self, source: str) -> None:
        self.source = source

    def get(self) -> Union[cv2.VideoCapture, None]:
        """
        Get Video Stream
        """
        video = None
        try:
            video = cv2.VideoCapture(self.source)
            get_logger().info("STREAM: Loaded video capture", source=self.source)
        except Exception:
            get_logger().error(
                "STREAM: video capture exception", source=self.source, exc_info=True
            )

        return video

    @staticmethod
    def check_stream(
        stream: cv2.VideoCapture, retry: int = 3, timeout: float = 5
    ) -> bool:
        if retry < 1:
            retry = 3
            get_logger().warning(
                "STREAM: retry parameter must be at least 1. Defaulting to 3...",
                retry=retry,
                timeout=timeout,
            )
        if timeout <= 0:
            timeout = 5
            get_logger().warning(
                "STREAM: timeout parameter must be at > 0. Defaulting to 5...",
                retry=retry,
                timeout=timeout,
            )
        for r in range(retry):
            if not stream.isOpened():
                r = r + 1
                get_logger().error(
                    "STREAM: Stream is not opened. Retrying...",
                    retry_count=r,
                    timeout=timeout,
                )
                sleep(timeout)
            else:
                get_logger().info(
                    "STREAM: Stream is opened.", retry_count=r + 1, timeout=timeout
                )
                return True

        get_logger().critical(
            "STREAM: Can not connect to stream after max attepts. Exiting...",
            retry=retry,
            timeout=timeout,
        )
        sys.exit()

    @staticmethod
    def read(stream: cv2.VideoCapture) -> np.ndarray:
        """
        Reading frames from Video
        """

        ret, frame = stream.read()
        if not ret:
            get_logger().critical(
                "STREAM: Issue reading from stream. Exiting...",
            )
            sys.exit()

        return frame

    @staticmethod
    def stop(stream: cv2.VideoCapture) -> None:
        """
        Stop Video Stream
        """
        stream.release()
