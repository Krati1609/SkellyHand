# SkellyHand 🖐️✨

a little experiment using mediapipe to turn my hands into a glowing web. it tracks both hands, connects matching fingers between them (index to index, middle to middle, etc.), and the lines get thicker and brighter the closer your hands get to the camera. each finger has its own color too.

## demo

https://github.com/user-attachments/assets/bec697df-1e8e-4b8b-a4ed-2fac32ec7ac6


## getting it running

make sure you have python installed, then:

```
pip install opencv-python mediapipe numpy
```

grab the `hand_landmarker.task` model file and drop it in the same folder as the script:

```
curl -O https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

then run it:

```
python handLandmarker.py
```

hit `q` to quit.

## notes

- if it's lagging, try lowering `CAPTURE_WIDTH`/`CAPTURE_HEIGHT` near the top of the script
- colors and glow size are all tweakable in the config section at the top

## about

saw a post on linkedin about hand tracking with mediapipe and got curious, watched a few youtube videos to understand the basics, then built this. leaned on AI for the code, but the real learning was in debugging: noisy depth values, faking a glow in opencv, chasing down lag. still early days in ML/CV, just having fun learning by building.
