import cv2
import numpy as np
import math

class Transformation:

	def __init__(self):
		pass

	def get_bounding_box(self, bbox):

		xmin = bbox[1]
		ymin = bbox[0]
		xmax = bbox[3]
		ymax = bbox[2]

		return np.array([(xmin, ymin), (xmax, ymin), (xmin, ymax), (xmax, ymax)], np.int32)

	def get_centroid(self, bbox):

		return [(abs(bbox[3][0] + bbox[0][0]) // 2, abs(bbox[3][1] + bbox[0][1]) // 2)]

	def get_transformation_matrix(self):

		left_top = (260, 60)
		right_top = (625, 30)
		right_bottom = (455,360)
		left_bottom = (20, 160)

		# left_top = (710, 20)
		# right_top = (1230, 40)
		# right_bottom = (1100, 710)
		# left_bottom = (65, 400)

		widthA = np.sqrt(((right_bottom[0] - left_bottom[0]) ** 2) + ((right_bottom[1] - left_bottom[1]) ** 2))
		widthB = np.sqrt(((right_top[0] - left_top[0]) ** 2) + ((right_top[1] - left_top[1]) ** 2))
		maxWidth = max(int(widthA), int(widthB))

		heightA = np.sqrt(((right_top[0] - right_bottom[0]) ** 2) + ((right_top[1] - right_bottom[1]) ** 2))
		heightB = np.sqrt(((left_top[0] - left_bottom[0]) ** 2) + ((left_top[1] - left_bottom[1]) ** 2))
		maxHeight = max(int(heightA), int(heightB))

		img_rect = np.array([left_top, right_top, right_bottom, left_bottom], np.float32)
		dst = np.array([
			[0 + 250, 0],
			[maxWidth, 0],
			[maxWidth, maxHeight - 1],
			[0 + 250, maxHeight - 1]], dtype="float32")

		Matrix = cv2.getPerspectiveTransform(img_rect, dst)

		return Matrix, maxHeight, maxWidth

	def get_perspective_transform(self, image):
		matrix, maxHeight, maxWidth = self.get_transformation_matrix()
		image = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))
		return image

	def get_bbox_perfective_transoform(self, bboxes, Matrix):

		warped_bboxes = []
		for bbox in bboxes:
			cbbox = self.get_centroid(bbox)
			warped_bboxes.append(cv2.perspectiveTransform(np.array([cbbox], np.float32), Matrix))
		return warped_bboxes

	def get_distance(self, cbox1, cbox2):

		return math.sqrt((cbox2[0] - cbox1[0]) ** 2 + (cbox2[1] - cbox1[1]) ** 2)

	def check_distance(self, img, scores, detection_bboxes):
		bboxes = []
		for x, y in zip(scores, detection_bboxes):
			if x >= 0.4:
				bboxes.append(self.get_bounding_box(y))
		Matrix, _, _ = self.get_transformation_matrix()
		warped_bboxes = np.array(self.get_bbox_perfective_transoform(bboxes, Matrix), np.int32)

		centers = []
		for idx in range(0, len(bboxes)):
			center = self.get_centroid(bboxes[idx])
			centers.append(center)
		centers_drawn = []
		violations = 0
		for i in range(0, len(warped_bboxes)):
			for j in range(i + 1, len(warped_bboxes)):
				dist = self.get_distance(warped_bboxes[i][0][0], warped_bboxes[j][0][0])

				if dist <= 85:
					violations += 1
					cv2.line(img, centers[i][0], centers[j][0], color=(0, 0, 255), thickness=2)
					cv2.circle(img, tuple(centers[i][0]), radius=20, color=(0, 0, 255), thickness=2)
					cv2.circle(img, tuple(centers[j][0]), radius=20, color=(0, 0, 255), thickness=2)
					centers_drawn.append(i)
					centers_drawn.append(j)

		for i in range(0, len(centers)):
			if i not in centers_drawn:
				cv2.circle(img, tuple(centers[i][0]), radius=20, color=(0, 255, 0), thickness=2)

		# cv2.putText(img, "Violation = " + str(len(centers_drawn)), (img.shape[1] - 300, 50), cv2.FONT_HERSHEY_SIMPLEX,
		# 			1, (255, 0, 0), thickness=2)

		return img, violations, len(centers)


