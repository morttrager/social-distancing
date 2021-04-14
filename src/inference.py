class Inference:
    def __init__(self, input, return_pipe=None, meta_dict=None):
        self.__input = input
        self.__meta_dict = meta_dict
        if not self.__meta_dict:
            self.__meta_dict = {}

        self.__return_pipe = return_pipe
        self.__data = None
        self.__result = None

    def get_input(self):
        return self.__input

    def get_meta_dict(self):
        return self.__meta_dict

    def get_return_pipe(self):
        return self.__return_pipe

    def set_result(self, result):
        self.__result = result
        if self.__return_pipe:
            self.__return_pipe.push(self)

    def get_result(self):
        return self.__result

    def set_data(self, data):
        self.__data = data

    def get_data(self):
        return self.__data

    def set_meta(self, key, val):
        print("key = ", key)
        print("val = ", val)
        print("self dict = ", self.__meta_dict)

        if key not in self.__meta_dict.keys():
            self.__meta_dict[key] = val
        else:
            raise Exception

    def set_meta_force(self, key, val):
        self.__meta_dict[key] = val

    def get_meta(self, key):
        if key in self.__meta_dict.keys():
            return self.__meta_dict[key]
        return None

    def get_meta_or_default(self, key, val):
        if key in self.__meta_dict.keys():
            return self.__meta_dict[key]
        return val



# class Inference():
#
# from threading import Thread
# from py_pipe.pipe import Pipe
# from transformation import check_distance, get_perspective_transform
# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as viz_utils
#
# import cv2
# import numpy as np
# import tensorflow as tf
# import config
#

#     def __init__(self,image, model, label_map):
#         self.__image = image
#         self.__model = model
#         self.__label_map = label_map
#
#     def set_image(self, image):
#         self.__image==image
#
#     def get_image(self):
#         return self.__image
#
#     def set_model(self, model):
#         self.__image == model
#
#     def get_model(self):
#         return self.model
#
#     def set_label_map(self, label_map):
#         self.__label_map == label_map
#
#     def get_image(self):
#         return self.__label_map
#
#
#     def inference(image, detect_fn, category_index):
#
#         input_tensor = tf.convert_to_tensor(image)
#         # The model expects a batch of images, so add an axis with `tf.newaxis`.
#         input_tensor = input_tensor[tf.newaxis, ...]
#
#         # input_tensor = np.expand_dims(image, 0)
#         detections = detect_fn(input_tensor)
#
#         # All outputs are batches tensors.
#         # Convert to numpy arrays, and take index [0] to remove the batch dimension.
#         # We're only interested in the first num_detections.
#         num_detections = int(detections.pop('num_detections'))
#         detections = {key: value[0, :num_detections].numpy()
#                       for key, value in detections.items()}
#         detections['num_detections'] = num_detections
#         # detection_classes should be ints.
#         detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
#
#         image_np_with_detections = image.copy()
#
#         check_social_distance = True
#         if check_social_distance:
#             image_np_with_detections = check_distance(image_np_with_detections, detections["detection_scores"], detections["detection_boxes"])
#         else:
#             viz_utils.visualize_boxes_and_labels_on_image_array(
#                 image_np_with_detections,
#                 detections['detection_boxes'],
#                 detections['detection_classes'],
#                 detections['detection_scores'],
#                 category_index,
#                 use_normalized_coordinates=True,
#                 max_boxes_to_draw=200,
#                 min_score_thresh=.30,
#                 agnostic_mode=False)
#
#         return image_np_with_detections
#
# #
# # def generator():
# #     cap = cv2.VideoCapture(config.INPUT_VIDEO_PATH)
# #     while (cap.isOpened()):
# #
# #         ret, frame = cap.read()
# #         if ret:
# #             pipe.push_wait()
# #             pipe.push(frame)
# #         else:
# #             break
# #
# #
# # def cumulative_adder():
# #     detections_on_perspective = False
# #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
# #     flag = True
# #     while True:
# #         pipe.pull_wait()
# #         ret, frame = pipe.pull()
# #
# #         if ret:
# #             if detections_on_perspective:
# #                 frame = get_perspective_transform(frame)
# #             frame = inference(frame)
# #             cv2.imshow('frame', frame)
# #             if config.SAVE_VIDEO:
# #                 if flag:
# #                     if "/" in config.INPUT_VIDEO_PATH:
# #                         out = cv2.VideoWriter("output/videos/"+config.INPUT_VIDEO_PATH.split("/")[-1].split(".")[0]+".avi", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
# #                     else:
# #                         print("in else")
# #                         out = cv2.VideoWriter("output/videos/"+config.INPUT_VIDEO_PATH.split(".")[0]+".avi", fourcc, 20.0, (frame.shape[1], frame.shape[0]))
# #
# #                     flag = False
# #                 out.write(frame)
# #             if cv2.waitKey(1) & 0xFF == ord('q'):
# #                 return
# #         else:
# #             break
# #
# #
# # pipe = Pipe()
# # print("video loaded : ",config.INPUT_VIDEO_PATH)
# # if not config.CHECK_PERSPECTIVE:
# #     category_index = label_map_util.create_category_index_from_labelmap(config.LABEL_MAP_PATH, use_display_name=True)
# #     detect_fn = tf.saved_model.load(config.SAVED_MODEL_PATH)
# #
# #     Thread(target=generator).start()
# #     Thread(target=cumulative_adder).start()
# #
# # else:
# #     cap = cv2.VideoCapture(config.INPUT_VIDEO_PATH)
# #     save_image = False
# #     while (cap.isOpened()):
# #         ret, frame = cap.read()
# #         if config.SAVE_FRAME:
# #             print("size = ", frame.shape)
# #             cv2.imwrite("output/images/"+config.INPUT_VIDEO_PATH.split("/")[-1].split(".")[0]+".jpg", frame)
# #             break
# #         if ret:
# #             frame = get_perspective_transform(frame)
# #             cv2.imshow("img", frame)
# #
# #             if cv2.waitKey(1) & 0xFF == ord('q'):
# #                 break
# #         else:
# #             break
