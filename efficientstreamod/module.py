import cv2
import threading
import time
from collections import deque
from abc import ABC, abstractmethod
import numpy as np

class EfficientObjectDetection(ABC):
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.frames = []
        self.results = {'frames': deque(maxlen=20), 'results': deque(maxlen=20)}

    def start_stream(self, grid_type):
        stream_thread = threading.Thread(target=self.run_frame, args=(grid_type,))
        stream_thread.start()

    def run_frame(self, grid_type):
        cap = cv2.VideoCapture(self.stream_url)
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
            if len(self.frames) > grid_type:
                t0 = time.time()
                grid_frames, result = self.grid_inference(grid_type)
                self.results['frames'].extend(grid_frames)
                self.results['results'].extend([result] if grid_type == 1 else result)
                t1 = time.time()
                print(f"processing {grid_type} frames took {t1-t0:.3f} seconds.")
            time.sleep(0.02)

    @abstractmethod
    def inference(self, x):
        raise NotImplementedError("Please implement the 'inference' method.")

    def grid_inference(self, grid_type):
        if grid_type not in [1, 4, 9]:
            raise ValueError("grid_type must be 1, 4, or 9.")
        
        n = grid_type
        grid_frames = self.frames[:n]
        del self.frames[:n]
        grid_image = self._create_grid(grid_frames, grid_type)
        grid_result = self.inference(grid_image)
        
        if grid_type == 1:
            return grid_frames, grid_result

        results = [[] for _ in range(n)]
        if grid_type == 4:
            rows = cols = 2
        elif grid_type == 9:
            rows = cols = 3

        frame_height, frame_width, _ = grid_frames[0].shape

        for result in grid_result:
            x1, y1, x2, y2, conf, class_id = result
            i = int(y1 // frame_height)
            j = int(x1 // frame_width)
            frame_idx = i * cols + j

            x1 -= j * frame_width
            x2 -= j * frame_width
            y1 -= i * frame_height
            y2 -= i * frame_height
            
            adjusted_result = (x1, y1, x2, y2, conf, class_id)
            results[frame_idx].append(adjusted_result)

        return grid_frames, results


    def _create_grid(self, grid_frames, grid_type):
        if grid_type == 1:
            return grid_frames[0]
        elif grid_type == 4:
            rows = 2
            cols = 2
        elif grid_type == 9:
            rows = 3
            cols = 3
        else:
            raise ValueError("grid_type must be 1, 4, or 9.")

        grid_image = None
        frame_height, frame_width, _ = grid_frames[0].shape

        for i in range(rows):
            row_frames = []
            for j in range(cols):
                index = i * cols + j
                if index < len(grid_frames):
                    frame = grid_frames[index]
                else:
                    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                row_frames.append(frame)
            row_image = np.hstack(row_frames)
            if grid_image is None:
                grid_image = row_image
            else:
                grid_image = np.vstack((grid_image, row_image))

        return grid_image

    def get_result(self):
        try:
            return self.results['frames'].popleft(), self.results['results'].popleft()
        except IndexError:
            return None, None

    def get_frame(self):
        try:
            return self.results['frames'].popleft()
        except IndexError:
            return None
