Model: YOLO11n
Version: target_v3

Task:
Single-class object detection

Class:
0 = target

Input:
images [1, 3, 640, 640]

Output:
output0 [1, 5, 8400]

Output meaning:
8400 candidate detections
5 values per candidate:
[cx, cy, w, h, confidence]

Training:
Base model: yolo11n.pt
Image size: 640 x 640
Batch: 8
Epochs: 150
Device: Tesla T4

Dataset:
Train: 32 images
Val: 8 images
No separate test split

Dataset notes:
- Real frames extracted from competition-scene video
- Includes positive target samples
- Includes negative/background samples
- Class 0 = target

Validation:
Fill in final P / R / mAP50 / mAP50-95 here if needed

Deployment files:
target_v3.pt
target_v3.onnx

openvino/
  target_v3.xml
  target_v3.bin
  metadata.yaml

Recommended deployment:
OpenVINO C++

OpenVINO files:
target_v3.xml
target_v3.bin

Important:
The XML and BIN files must be kept together.

Preprocessing:
- Input size: 640 x 640
- 3 channels
- RGB
- NCHW
- Float normalized input

Postprocessing:
- Raw output shape: [1, 5, 8400]
- Apply confidence threshold
- Convert cx,cy,w,h to bounding box
- Apply NMS
- Map coordinates back to original image

Notes:
- target_v3 performed well in independent video testing.
- Keep this model as the current deployment candidate.