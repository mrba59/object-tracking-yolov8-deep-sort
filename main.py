import os
import random
from argparse import ArgumentParser
from football_detection.foot_utils.util import read_yolo_annot, get_frame_annotation, sorted_alphanumeric
import numpy as np
import cv2
from tracker import Tracker

parser = ArgumentParser()
parser.add_argument('annot_dir', help='path to annotation file coco (json) or dir to yolo', type=str)
parser.add_argument('data', help='path to video', type=str)
parser.add_argument('data_type', help='video or images', type=str)
parser.add_argument('output_type', help='video or images', type=str)
parser.add_argument('pattern_name', help='pattern label name', type=str)
parser.add_argument('output_dir', help='output_dir', type=str)

args = parser.parse_args()


def crop(frame, box):
    crop = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
    return crop


def get_cropped_detection(frame, annot_list):
    boxes = [annot[1:5] for annot in annot_list]
    return [crop(frame, box) for box in boxes]


def NMS(boxes, overlapThresh=0.4):
    # return an empty list, if no boxes given
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # We have a least a box of one pixel, therefore the +1
    indices = np.arange(len(x1))
    for i, box in enumerate(boxes):
        temp_indices = indices[indices != i]
        xx1 = np.maximum(box[0], boxes[temp_indices, 0])
        yy1 = np.maximum(box[1], boxes[temp_indices, 1])
        xx2 = np.minimum(box[2], boxes[temp_indices, 2])
        yy2 = np.minimum(box[3], boxes[temp_indices, 3])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        if np.any(overlap > overlapThresh):
            indices = indices[indices != i]
    return boxes[indices]


def load_yolo_labels(label_path):
    """Load YOLO format labels from a file."""
    with open(label_path, 'r') as f:
        labels = [line.strip().split() for line in f.readlines()]
    return labels


colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

if __name__ == "__main__":
    video_path = args.data
    annot_dir = args.annot_dir
    annot_txt_list, size_annot = read_yolo_annot(annot_dir)
    data_type = args.data_type
    pattern_name = args.pattern_name
    output_dir = args.output_dir
    output_type = args.output_type

    tracker = Tracker()

    vid = cv2.VideoCapture(video_path)
    ret, frame = vid.read()
    frame_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_dir, fourcc, fps, (frame_width, frame_height))

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    if output_type == 'video':
        cap_out = cv2.VideoWriter(os.path.join(output_dir, pattern_name + ".MP4"), cv2.VideoWriter_fourcc(*'MP4V'),
                                  vid.get(cv2.CAP_PROP_FPS),
                                  (frame.shape[1], frame.shape[0]))
    num_frame = 0
    for number_label in range(1, total_frames):
        annot_list, frame, number_label = get_frame_annotation(annot_dir, annot_txt_list, data_type, number_label, None,
                                                               pattern_name, vid, field=None,
                                                               tracking=False)
        list_crop = get_cropped_detection(frame, annot_list)
        tracker.update(frame, list_crop, annot_list)

        for track in tracker.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
            cv2.putText(frame, str(track_id), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2,
                        cv2.LINE_AA)
        if output_type == "video":
            cap_out.write(frame)
        else:
            cv2.imwrite(os.path.join(output_dir, "frame_" + str(num_frame) + ".jpg"), frame)
            num_frame += 1
    #
    # ret, frame = cap.read()
    # num_frame += 1

vid.release()
cap_out.release()
cv2.destroyAllWindows()
