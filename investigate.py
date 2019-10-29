import glob
from keras.preprocessing.image import load_img
folder_path = 'data/dataset\\'


import os
all_plants = [x[0].replace(folder_path,'') for x in os.walk(folder_path)]
format_error = 0
mode_error = 0
oserrors = 0
for plant in all_plants[1:]:
    for img_path in glob.glob('data/dataset/{}/*.jpg'.format(plant)):
        try:
            img =  load_img(img_path)
        except OSError:
            oserrors += 1
            os.remove(img_path)

oserrors
        # if img.format != 'JPEG':
        #     # print("Format ERROR")
        #     # print(img_path)
        #     # print()
        #     format_error += 1
        #
        # if img.mode != 'RGB':
        #     # print("Mode ERROR")
        #     # print(img_path)
        #     # print()
        #     print(img.mode)
        #     mode_error += 1



mode_error
format_error
