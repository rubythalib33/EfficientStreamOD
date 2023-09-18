# EfficientStreamOD


Efficient Object Detection on video streams using OpenCV.

The `EfficientStreamOD` module provides an abstract class for efficiently processing video streams and performing object detection. The class uses a grid-based approach to process multiple frames in parallel and improve the overall processing speed.

## Installation

To install the package, run the following command:
```bash
pip install git+https://github.com/rubythalib33/EfficientStreamOD
```

## Benchmark
### Grid Method
#### Methodology
Grid method is as simply stack the frame into one image, for now it's only support for 1, 4 and 9 grid, here's the example
there's 4 frames: A, B, C, D
it will become like this:
![grid-image](./assets/grid_image.jpg)
#### Experimentation
This code is tested using Ryzen 7 5000 series Laptop CPU to run the object detection, for the architecture I choose [YOLOX-S](https://github.com/Megvii-BaseDetection/YOLOX)
 and run it on onnx environment, here is the benchmark

| Grid Type | Number of Frames Processed | Inference Speed |
|-----------|----------------------------|-----------------|
| 1(no grid)| 1                          | 0.07s          |
| 4         | 4                          | 0.08s          |
| 9         | 9                          | 0.1s           |

#### Conclusion
Process the object detection using more grid will more optimize the performance, but there's a cost of the optimization that more grid you use, the small object in a frame cannot be seen by the AI, and for my experimentation the best result I get is in 4 grid method, because the 4 grid one is still has a stable detection

## sample code to use the program from YOLOX-S.onnx
```python
class YOLOX(EfficientObjectDetection):
    def __init__(self, model_path, stream_url, input_shape=(640, 640), score_thr=0.3):
        super().__init__(stream_url)
        self.score_thr = score_thr
        self.input_shape = input_shape
        self.session = onnxruntime.InferenceSession(model_path)

    def inference(self, img):
        img, ratio = self.preprocess(img)
        output = self.session.run(None, {self.session.get_inputs()[0].name: img[None, :, :, :]})
        dets = self.postprocess(output, ratio)
        return dets

    def preprocess(self, img):
        img, ratio = preprocess(img, self.input_shape)
        return img, ratio

    def postprocess(self, output, ratio):
        predictions = demo_postprocess(output[0], self.input_shape)[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=self.score_thr)
        return dets

    def visualize(self, img, dets):
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            img = vis(img, final_boxes, final_scores, final_cls_inds,
                        conf=self.score_thr, class_names=COCO_CLASSES)
        return img

yolox = YOLOX(args.model, args.image_path, input_shape, args.score_thr)
yolox.start_stream(grid_type=4)
time.sleep(1)
while True:
    img,dets = yolox.get_result()
    dets = np.array(dets)
    if img is not None:
        img = yolox.visualize(img, dets)
        cv2.imshow('result', img)
        if cv2.waitKey(1) == ord('q'):
            break
    time.sleep(0.03)
```