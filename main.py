"""
Main module of the project
"""


from argparse import ArgumentParser

from detector import Detector
from stream import Stream

if __name__ == "__main__":
    parser = ArgumentParser(description="Process script arguments")
    parser.add_argument(
        "--source",
        type=str,
        default="http://195.196.36.242/mjpg/video.mjpg",
        help="Source to the streaming video",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["yolov5s", "yolov5l"],
        help="List with models to be loaded",
    )
    args = parser.parse_args()
    detector = Detector(
        stream=Stream(args.source),
        models=args.models,
    )
    detector()
