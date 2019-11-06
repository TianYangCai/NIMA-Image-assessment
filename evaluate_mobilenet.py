import numpy as np
import argparse
from path import Path

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import pandas as pd
import xlwt

from utils.score_utils import mean_score, std_score

parser = argparse.ArgumentParser(description='Evaluate NIMA(Inception ResNet v2)')
parser.add_argument('-dir', type=str, default='C:\\Users\\Admin\\Desktop\\neural-image-assessment\\images',
                    help='Pass a directory to evaluate the images in it')

parser.add_argument('-img', type=str, default=[None], nargs='+',
                    help='Pass one or more image paths to evaluate them')

parser.add_argument('-resize', type=str, default='true',
                    help='Resize images to 224x224 before scoring')

parser.add_argument('-rank', type=str, default='true',
                    help='Whether to tank the images after they have been scored')

args = parser.parse_args()
resize_image = args.resize.lower() in ("true", "yes", "t", "1")
target_size = (224, 224) if resize_image else None
rank_images = args.rank.lower() in ("true", "yes", "t", "1")

# give priority to directory
if args.dir is not None:
    print("Loading images from directory : ", args.dir)
    imgs = Path(args.dir).files('*.png')
    imgs += Path(args.dir).files('*.jpg')
    imgs += Path(args.dir).files('*.jpeg')

elif args.img[0] is not None:
    print("Loading images from path(s) : ", args.img)
    imgs = args.img

else:
    raise RuntimeError('Either -dir or -img arguments must be passed as argument')

with tf.device('/CPU:0'):
    base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights('weights/mobilenet_weights.h5')

    score_list = []


    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('mysheet', cell_overwrite_ok=True)
    head = ['image', 'mean', 'std']
    for h in range(len(head)):
        sheet.write(0,h,head[h])

    i = 1
    for img_path in imgs:
        img = load_img(img_path, target_size=target_size)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        x = preprocess_input(x)

        scores = model.predict(x, batch_size=1, verbose=0)[0]

        mean = mean_score(scores)
        std = std_score(scores)

        file_name = Path(img_path).name.lower()
        score_list.append((file_name, mean))

        print("Evaluating : ", img_path)
        print("NIMA Score : %0.3f +- (%0.3f)" % (mean, std))
        print()
        result = [img_path.split('\\')[6],mean,std]
        for j in range(3):
            sheet.write(i,j,result[j])
        i += 1

    book.save('C:\\Users\\Admin\\Desktop\\result.xls')


    if rank_images:
        print("*" * 40, "Ranking Images", "*" * 40)
        score_list = sorted(score_list, key=lambda x: x[1], reverse=True)

        for i, (name, score) in enumerate(score_list):
            print("%d)" % (i + 1), "%s : Score = %0.5f" % (name, score))


