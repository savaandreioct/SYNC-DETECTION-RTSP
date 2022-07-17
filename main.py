from detector import Detector
from stream import Stream

if __name__ == "__main__":

    detector = Detector(    
        stream=Stream("http://195.196.36.242/mjpg/video.mjpg"),
        models=["yolov5s", "yolov5l"],
    )
    detector()
