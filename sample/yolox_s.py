#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os

import cv2
import numpy as np
import time

import onnxruntime

from utils import preproc as preprocess , COCO_CLASSES, demo_postprocess, vis, multiclass_nms

import sys
sys.path.append('../')
from efficientstreamod.module import EfficientObjectDetection


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


def make_parser():
    parser = argparse.ArgumentParser("onnxruntime inference sample")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="yolox_s.onnx",
        help="Input your onnx model.",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default='cars.mp4',
        help="Path to your input image.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='demo_output',
        help="Path to your output directory.",
    )
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    parser.add_argument(
        "--input_shape",
        type=str,
        default="640,640",
        help="Specify an input shape for inference.",
    )
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()

    input_shape = tuple(map(int, args.input_shape.split(',')))

    yolox = YOLOX(args.model, args.image_path, input_shape, args.score_thr)
    yolox.start_stream(grid_type=9)
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