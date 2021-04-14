import cv2


cap = cv2.VideoCapture('data/videos/vid_short.mp4')
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         print(frame.shape[0])
#         shape = frame.shape
#         print("shape = ",shape[0])
#         cv2.imshow('frame',gray)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

#cap.release()

while(True):

    frame = cv2.imread("/home/stgat/workspace/social-distancing/frame2.jpg")
    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cv2.destroyAllWindows()
cv2.destroyAllWindows()

#sample image
# left_bottom = (0, image_shape[0])
# 	right_bottom = (image_shape[1], image_shape[0])
# 	left_top = ((image_shape[1] - 0) // 2 - left_buffer, 0)
# 	right_top = ((image_shape[1] - 0) // 2 + right_buffer, 0)


#vid_short.mp4
# left_top = (710, 20)
# right_top = (1230, 40)
# right_bottom = (1100, 710)
# left_bottom = (65, 400)