import cv2


class RSTPStream:
    def __init__(self, url: str) -> None:
        self.url = url

    def start(self):
        video = cv2.VideoCapture(self.url)
        while True:
            _, frame = video.read()
            cv2.imshow("RTSP", frame)
            k = cv2.waitKey(1)
            if k == ord("q"):
                break
        video.release()
        cv2.destroyAllWindows()


stream = RSTPStream("http://195.196.36.242/mjpg/video.mjpg")
stream.start()
