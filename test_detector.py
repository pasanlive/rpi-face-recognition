import cv2
import numpy as np
import os
import urllib.request

model_path = "./models/object_detection_nanodet_2022nov.onnx"
if not os.path.exists(model_path):
    url = "https://github.com/opencv/opencv_zoo/raw/main/models/object_detection_nanodet/object_detection_nanodet_2022nov.onnx"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as resp, open(model_path, 'wb') as f:
        f.write(resp.read())

net = cv2.dnn.readNetFromONNX(model_path)
print("NanoDet Model Loaded:", net)
