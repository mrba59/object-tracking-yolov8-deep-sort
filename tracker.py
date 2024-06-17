from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from deep_sort.tools import generate_detections as gdet
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
import numpy as np
from torchreid.utils import FeatureExtractor


class Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self):
        max_cosine_distance = 0.4
        nn_budget = None

        encoder_model_filename = '/home/reda/Documents/football_detection/deep-sort-tracking/mars-small128.pb'

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric)
        self.encoder = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='/home/reda/Documents/projets/deep-person-reid-master/log/resnet50/model/model.pth.tar-120',
            device='cuda'
        )
        #self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)

    def update(self, list_crop, detections, num_frame, init):
        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])
            self.update_tracks()
            return
        scores = [d[-1] for d in detections]
        features = self.encoder(list_crop)
        bboxes = np.asarray([d[1:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        self.tracker.predict()
        if init:
            self.tracker.initialization(dets)
        else:
            self.tracker.update(dets)
        self.update_tracks(num_frame)
    """def update(self, frame, detections, num_frame):

        if len(detections) == 0:
            self.tracker.predict()
            self.tracker.update([])
            self.update_tracks()
            return

        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-1] for d in detections]

        features = self.encoder(frame, bboxes)

        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        self.tracker.predict()
        if num_frame == 0:
            self.tracker.initialization(dets)
        else:
            self.tracker.update(dets)
        self.update_tracks(num_frame)"""

    def update_tracks(self, num_frame):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() and num_frame > 2:
                continue
            bbox = track.to_tlbr()

            id = track.track_id

            tracks.append(Track(id, bbox))

        self.tracks = tracks


class Track:
    track_id = None
    bbox = None

    def __init__(self, id, bbox):
        self.track_id = id
        self.bbox = bbox
