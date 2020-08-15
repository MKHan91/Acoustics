from glob import glob
import cv2
import os

image_folder = 'D:\\Onepredict_MK\\LG CNS\\By_datasetMK\\201907\\20190717_generated_data\\fault1'
video_name = 'D:\\Onepredict_MK\\LG CNS\\By_datasetMK\\201907\\20190717_generated_data\\generated_fault1_video.mp4'

# data = []
# for img in sorted(os.listdir(image_folder)):
#     if "_concat" in img:
#         images += [img]
images = [img for img in sorted(os.listdir(image_folder))]

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 13.0, (width,height), True)

for num, image in enumerate(images):
    video.write(cv2.imread(os.path.join(image_folder, image)))
    print('current processing:{}'.format(num + 1))

# cv2.destroyAllWindows()
video.release()