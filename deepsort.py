import os
import cv2
import time
import argparse
import pickle as pkl

import numpy as np
import torch
import warnings

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config


class VideoTracker(object):
    def __init__(self, cfg, args, video_path, result_filename="results"):
        self.cfg = cfg
        self.args = args
        self.result_filename = result_filename
        self.video_path = video_path
        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        self.vdo = cv2.VideoCapture()
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = args.class_names

    def __enter__(self):
        assert os.path.isfile(self.video_path), "Path error"
        self.vdo.open(self.video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        assert self.vdo.isOpened()

        with open(self.args.box_file, 'rb') as f:
            self.bbox = pkl.load(f)

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.args.save_path, fourcc, 20, (self.im_width, self.im_height))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        results = []
        idx_frame = 0
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            bbox_tlbrc = np.concatenate([
                self.bbox[idx_frame - 1][1],
                self.bbox[idx_frame - 1][2]],
                axis=0
            )
            print(bbox_tlbrc[:10])
            if len(bbox_tlbrc) > 0:
                bbox_xywh = np.concatenate([
                    (bbox_tlbrc[:, :2] + bbox_tlbrc[:, 2:4]) / 2,
                    bbox_tlbrc[:, 2:4] - bbox_tlbrc[:, :2]
                ], axis=1)
                bbox_xywh[:, 2:] *= 1.2  # bbox dilation just in case bbox too small
                cls_conf = bbox_tlbrc[:, 4]

                # do tracking
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_tlwh = []
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

                    for bb_xyxy in bbox_xyxy:
                        bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                    results.append((idx_frame - 1, bbox_tlwh, identities))

            end = time.time()
            print("time: {:.03f}s, fps: {:.03f}".format(end - start, 1 / (end - start)))

            if self.args.save_path:
                self.writer.write(ori_im)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--box_file", type=str, default="./bboxes/cam_1.mp4.json")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--class_names", type=list, default=['car', 'truck'])
    parser.add_argument("--save_path", type=str, default="./demo/demo.avi")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()
