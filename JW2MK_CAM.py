from JW_will.MK_version.JW2MK_model.JW2MK_model_v2_window import *
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications.resnet50 import preprocess_input, decode_predictions
from scipy import ndimage
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser('CAM code of Myungkyu')
parser.add_argument('--main_name',      type=str,       help='current main file name', default='20190611_JW2MK_mainV2_M3_2')
parser.add_argument('--data_path',      type=str,       help='path containing LG_CNS files',
                    default='D:\\Onepredict_MK\\LG CNS\\20190417_LG_CNS_test_mode\\20190416_test_3sec_image')
parser.add_argument('--test_result_path',    type=str,       help='path containing LG_CNS files',
                    default='D:\\Onepredict_MK\\LG CNS\\20190417_LG_CNS_test_mode\\Test_result\\201906\\20190613_JW2MK_mainV3-9_M3_2_ep80')
parser.add_argument('--checkpoint_path',    type=str,       help='path containing LG_CNS files',
                    default='D:\\Onepredict_MK\\LG CNS\\Training_result\\TrainedModel')

args = parser.parse_args()

def Example():
    # image_path = args.data_path + '\\20190301074833_Wave1_2019Y03M01D_07h48m33s_3sec_nonhighpass.png'
    image_path = 'D:\\Onepredict_MK\\LG CNS\\20190417_LG_CNS_test_mode\\puppy.jpg'

    resnet = ResNet50(input_shape=(224, 224, 3), weights='imagenet', include_top=True)
    print(resnet.summary())

    activation_layer = resnet.get_layer('activation_48')
    model = Model(inputs=resnet.input, outputs=activation_layer.output)
    final_dense = resnet.get_layer('fc1000')
    weight = final_dense.get_weights()[0]

    img = image.load_img(image_path, target_size=(224, 224))
    img_input = preprocess_input(np.expand_dims(img, 0))
    fmaps = model.predict(img_input)[0]                     # fmaps.shape = (7, 7, 2048)

    probs = resnet.predict(img_input)                       # probs.shape = (1, 1000)
    class_names = decode_predictions(probs)
    class_name = class_names[0][0]
    pred = np.argmax(probs[0])                              # pred = 207

    w = weight[:, pred]                                     # w.shape = (2048,)
    cam = fmaps.dot(w)                                      # cam.shape = (7, 7)
    camp = ndimage.zoom(cam, (32, 32), order=1)

    plt.subplot(1, 2, 1)
    plt.imshow(img, alpha=0.8)
    plt.imshow(camp, cmap='jet', alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.title(class_name)
    plt.show()

def make_CAM(ckpt_path, main_name):
    test_images = sorted(glob(args.data_path + '/*.png'))

    resnet = resnet_windows()
    best_model_path = os.path.join(ckpt_path, '201906_Model', main_name + '_model', 'model-0014.ckpt')
    resnet.load_weights(best_model_path)

    for num, img in enumerate(test_images):
        img_pil = image.load_img(img, target_size=(256, 256))
        img_input = np.asarray(img_pil, dtype=np.float32)
        img_input = np.expand_dims(img_input, axis=0)
        img_input = img_input / 255.0

        activation_layer = resnet.layers[0].get_layer('activation_48')
        model = Model(inputs=resnet.layers[0].input, outputs=activation_layer.output)
        final_dense = resnet.get_layer('dense')
        weight = final_dense.get_weights()[0]  # weights.shape = (2048, 7)

        fmaps = model.predict(img_input)[0]  # fmaps.shape = (8, 8, 2048)
        # fmaps_v1 = fmaps[:,:,0:512]
        # fmaps_v2 = fmaps[:, :, 512:1024]
        # fmaps_v3 = fmaps[:, :, 1024:2048]
        probs = resnet.predict(img_input)
        pred = np.argmax(probs[0])  # matrix index

        # if os.path.exists(os.path.join(args.test_result_path, 'CAM_test_class{}'.format(str(pred)))) != True:
        #     os.makedirs(os.path.join(args.test_result_path, 'CAM_test_class{}'.format(str(pred))))

        """Extracting weight corresponding to specific class (pred)"""
        w = weight[:, pred]
        cam = fmaps.dot(w)
        # cam_v2 = fmaps_v2.dot(w)
        # cam_v3 = fmaps_v3.dot(w)
        camp = ndimage.zoom(cam, (32, 32), order=1)
        # camp_v2 = ndimage.zoom(cam_v2, (32, 32), order=1)
        # camp_v3 = ndimage.zoom(cam_v3, (32, 32), order=1)

        plt.imshow(img_pil, alpha=1.0, aspect='auto', interpolation='nearest')
        plt.imshow(camp, cmap='jet', alpha=0.6, interpolation='nearest')
        plt.axis('off')

        # plt.savefig(os.path.join(args.test_result_path, 'CAM_test_class{}'.format(str(pred)), img.split('\\')[-1][:-21] + '_CAM.png'))
        print('current processing: {}/{}'.format(num + 1, len(test_images)))
        plt.show()

if __name__ == '__main__':
    # read_CAM(args.checkpoint_path, args.main_name)
    # Example()
    make_CAM(args.checkpoint_path, args.main_name)