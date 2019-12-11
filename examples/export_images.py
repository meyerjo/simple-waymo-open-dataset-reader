# Copyright (c) 2019, Grégoire Payen de La Garanderie, Durham University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import sys
import threading
import time
import errno
import os
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import dataset_pb2
from simple_waymo_open_dataset_reader import utils

EXPORT_FOLDER = '/home/meyerjo/dataset/waymo_export/'


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def convert_to_png(filename):
    assert(filename.endswith('numpy'))
    print('{}: Task started'.format(filename))
    path, ext = os.path.splitext(filename)
    im = np.load(filename, allow_pickle=True)
    pil = Image.fromarray(im)
    pil.save(path + '.png')
    os.remove(filename)
    print("{}: Task Executed {}".format(
        filename, threading.current_thread()))


class Exporter(object):
    def __init__(self, export_folder=None):
        if export_folder is None:
            export_folder = EXPORT_FOLDER
        self.export_folder = export_folder
        self.executor = ThreadPoolExecutor(max_workers=4)

    def export(self, camera_calibration, camera, labels, camera_labels, frame_id=0, sequence_id=None):
        assert(os.path.exists(self.export_folder))
        # Get the image transformation matrix
        vehicle_to_image = utils.get_image_transform(camera_calibration)

        # Decode the JPEG image
        img = utils.decode_image(camera)

        # Draw all the groundtruth labels
        box_3d_to_2d = []
        box_3d_class_labels = []
        for label in labels:
            obj = utils.get_3d_boxes_to_2d(img, vehicle_to_image, label)
            if obj is None:
                continue
            box_3d_to_2d.append(list(map(int, obj['box'])))
            box_3d_class_labels.append(obj['label'])

        box_2d = []
        box_2d_labels = []
        if camera_labels is not None:
            for label in camera_labels:
                obj = utils.get_2d_boxes(label)
                if obj is None:
                    continue
                box_2d.append(obj['box'])
                box_2d_labels.append(obj['label'])

        boxes = {
            '2d_boxes': {
                'boxes': box_2d,
                'label': box_2d_labels
            },
            '3d_boxes': {
                'boxes': box_3d_to_2d,
                'label': box_3d_class_labels
            }
        }

        velocities = {x[0].name: x[1] for x in camera.velocity.ListFields()}
        pose_data = {x[0].name: list(x[1]) for x in camera.pose.ListFields()}
        camera_info = {
            'id': '{}_{:05d}'.format(sequence_id, frame_id),
            'name': camera.name,
            'velocities': velocities,
            'pose': pose_data,
            'shutter': camera.shutter,
            'pose_timestamp': camera.pose_timestamp,
            'camera_trigger_time': camera.camera_trigger_time,
            'camera_readout_done_time': camera.camera_readout_done_time,
            'width': img.shape[1],
            'height': img.shape[0],
            'depth': img.shape[2]
        }

        if not os.path.exists(os.path.join(EXPORT_FOLDER, 'rgb', sequence_id)):
            mkdir_p(os.path.join(EXPORT_FOLDER, 'rgb', sequence_id))
        if not os.path.exists(os.path.join(EXPORT_FOLDER, 'label', sequence_id)):
            mkdir_p(os.path.join(EXPORT_FOLDER, 'label', sequence_id))
        if not os.path.exists(os.path.join(EXPORT_FOLDER, 'img_info', sequence_id)):
            mkdir_p(os.path.join(EXPORT_FOLDER, 'img_info', sequence_id))
        with open(
                os.path.join(
                    EXPORT_FOLDER,
                    'label', sequence_id,
                    'frame_{:05d}.json'.format(frame_id)), 'w') as f:
            json.dump(boxes, f)
        with open(
                os.path.join(
                    EXPORT_FOLDER,
                    'img_info', sequence_id,
                    'frame_{:05d}.json'.format(frame_id)), 'w') as f:
            json.dump(camera_info, f)

        _filename = os.path.join(EXPORT_FOLDER, 'rgb', sequence_id,
                              'frame_{:05d}.numpy'.format(frame_id))
        img.dump(_filename)

        self.executor.submit(convert_to_png, _filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)

    args = parser.parse_args()

    if args.input_dir is None:
        print("""Usage: python visualise_labels.py <datafile>
    Display the groundtruth 3D bounding boxes on the front camera video stream.""")
        sys.exit(0)

    # Open a .tfrecord
    import re
    input_path = args.input_dir
    export_path = args.output_dir
    if not os.path.exists(input_path):
        raise BaseException('input_path')

    exporter = Exporter(export_path)

    # get directories
    file_dirs = os.listdir(input_path)
    full_file_paths = [os.path.join(input_path, f) for f in file_dirs]
    full_file_paths = [f for f in full_file_paths if os.path.isfile(f)]

    for filename in full_file_paths:
        # get the sequence files
        print('Porting: {}'.format(filename))
        head, tail = os.path.split(filename)
        m = re.search('(.*)(_with_camera_labels)?\.tfrecord$', tail)
        if m is None:
            continue
        filename_without_tfrecord = m.group(1)

        # waymo data file reader
        datafile = WaymoDataFileReader(filename)

        # Generate a table of the offset of all frame records in the file.
        table = datafile.get_record_table()

        print("There are %d frames in this file." % len(table))

        # Loop through the whole file
        ## and display 3D labels.
        _start_time = time.time()
        frame_id = 0
        for frame in datafile:
            camera_name = dataset_pb2.CameraName.FRONT
            camera_calibration = utils.get(frame.context.camera_calibrations, camera_name)
            camera = utils.get(frame.images, camera_name)

            camera_labels = None
            if len(frame.camera_labels) != 0:
                camera_labels = utils.get(frame.camera_labels, camera_name)
                camera_labels = camera_labels.labels

            exporter.export(
                camera_calibration, camera,
                frame.laser_labels, camera_labels,
                frame_id=frame_id, sequence_id=filename_without_tfrecord
            )
            frame_id += 1
        print('Processed file in {}'.format(time.time() - _start_time))



