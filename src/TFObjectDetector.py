from py_pipe.pipe import Pipe
from threading import Thread
from inference import Inference
from inferedDetections import InferedDetections
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as viz_utils

import numpy as np
import tensorflow as tf


class TFObjectDetector:

    class Inference(Inference):

        def __init__(self, input, label_map, return_pipe=None, meta_dict=None):
            super().__init__(input, return_pipe, meta_dict)
            self.__label_map = label_map

        def get_label_map(self):
            return self.__label_map

    @staticmethod
    def __fetch_category_indices(label_map):

        class_count = len(label_map_util.get_label_map_dict(label_map).keys())
        label_map = label_map_util.load_labelmap(label_map)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=class_count,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        category_dict = {}
        for item in category_index.values():
            category_dict[item['id']] = item['name']
            category_dict[item['name']] = item['id']

        return category_index, category_dict

    def __init__(self, model, label_map,
                 image_shape=None,
                 flush_pipe_on_read=False):

        self.__category_index, self.__category_dict = self.__fetch_category_indices(label_map)
        self.__path_to_saved_model = model
        self.__model = tf.saved_model.load(self.__path_to_saved_model)
        self.__flush_pipe_on_read = flush_pipe_on_read
        self.__image_shape = image_shape
        self.__thread = None
        self.__in_pipe = Pipe(self.__in_pipe_process)
        self.__out_pipe = Pipe(self.__out_pipe_process)

    def load_model(self):
        self.__model = tf.saved_model.load(self.__path_to_saved_model)

    def model(self):
        return self.__model

    def __in_pipe_process(self, inference):
        images = inference.get_input()
        input_tensor = tf.convert_to_tensor(images)
        if len(images.shape) == 3:
            data = input_tensor[tf.newaxis, ...]
            if inference.get_meta("expand_dims") == None:
                inference.set_meta('expand_dims', True)
        else:
            print("shape ", images.shape)
            data = input_tensor
            inference.set_meta('expand_dims', False)

        inference.set_data(data)
        return inference

    def __out_pipe_process(self, result):
        result, inference = result

        if inference.get_meta('expand_dims'):

            num_detections = result['num_detections']
            detection_classes = result['detection_classes'][:num_detections].astype(np.uint8)
            detection_boxes = result['detection_boxes'][:num_detections]
            detection_scores = result['detection_scores'][:num_detections]
            if 'detection_masks' in result:
                detection_masks = result['detection_masks'][:num_detections]
            else:
                detection_masks = None

            results = InferedDetections(inference.get_input(), inference.get_label_map(), num_detections,
                                        detection_boxes, detection_classes,
                                        detection_scores,
                                        masks=detection_masks, is_normalized=True, get_category_fnc=self.get_category,
                                        annotator=self.annotate)

        else:
            results = []
            for i in range(len(result['num_detections'])):
                num_detections = int(result[i]['num_detections'])
                detection_classes = result[i]['detection_classes'][:num_detections].astype(np.uint8)
                detection_boxes = result[i]['detection_boxes'][:num_detections]
                detection_scores = result[i]['detection_scores'][:num_detections]
                if 'detection_masks' in result[i]:
                    detection_masks = result[i]['detection_masks'][:num_detections]
                else:
                    detection_masks = None

                results.append(
                    InferedDetections(inference.get_input()[i], inference.get_label_map(), num_detections,
                                      detection_boxes, detection_classes,
                                      detection_scores,
                                      masks=detection_masks, is_normalized=True,
                                      get_category_fnc=self.get_category,
                                      annotator=self.annotate))

        inference.set_result(results)

        if inference.get_return_pipe():
            return '\0'

        return inference

    def get_in_pipe(self):
        return self.__in_pipe

    def get_out_pipe(self):
        return self.__out_pipe

    def __detector(self, images):

        detections = self.__model(images)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections
        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        if 'detection_masks' in detections.keys():
            # The following processing is only for single image
            detection_boxes = tf.squeeze(detections['detection_boxes'], [0])
            detection_masks = tf.squeeze(detections['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(detections['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])

            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, self.__image_shape[0], self.__image_shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            detections['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)

        return detections

    def run(self):
        if self.__thread is None:
            self.__thread = Thread(target=self.__run)
            self.__thread.start()

    def __run(self):
        while self.__thread:

            if self.__in_pipe.is_closed():
                self.__out_pipe.close()
                return

            self.__in_pipe.pull_wait()
            ret, inference = self.__in_pipe.pull(self.__flush_pipe_on_read)
            if ret:
                self.__job(inference)

    def __job(self, inference):
        self.__out_pipe.push( (self.__detector(inference.get_data()), inference))

    def get_category(self, category):
        return self.__category_dict[category]

    @staticmethod
    def annotate(inferred_detections):
        annotated = inferred_detections.image.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
            annotated,
            inferred_detections.get_boxes_tlbr(),
            inferred_detections.get_classes().astype(np.int32),
            inferred_detections.get_scores(),
            TFObjectDetector.__fetch_category_indices(inferred_detections.get_label_map())[0],
            instance_masks=inferred_detections.get_masks(),
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=0.45,
            line_thickness=1)

        return annotated
