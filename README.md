<h1 align="center">SYNC-DETECTION-RTSP</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-1.0.0-blue.svg?cacheSeconds=2592000" />
</p>

> Load a video stream and apply YOLOV5 models to compare results.

## Install
Create a virtual env (optional) and run: 
```sh

pip install -r requirements.txt
```

To create a virual env:

```sh
pip install virtualenv
virtualenv {name}
```

To activate virtual env

```sh
\{name}\Scripts\activate 
```
## Usage

```sh

python main.py --source video_source_url --models model1 model2 model3

  --source SOURCE       Source to the streaming video
  --models [MODELS [MODELS ...]]
                        List with models to be loaded

```

## Run default

```sh
python main.py

```
Defatul values for arguments are:
```sh
--source http://195.196.36.242/mjpg/video.mjpg
--models yolov5s yolov5l

```


## Author

👤 **Octavian Andrei SAVA**

* Github: [@savaandreioct](https://github.com/savaandreioct)