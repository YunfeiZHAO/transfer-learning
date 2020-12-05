import os
import os.path
from os import path
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import colors as mcolors
import numpy as np
import argparse
import random
from shutil import copyfile
from skimage.morphology import skeletonize
from skimage import data
from skimage import img_as_bool, io, color, morphology
import matplotlib.pyplot as plt
from skimage.util import invert
import cv2
from scipy.signal import convolve2d


# python.exe D:/270219_STAGE/Utils/make_skel_from_bin.py -i //mph-p6004882.mph.morpho.com/Stage2019/Data/1117/Sea/bin_mkkp1 -o //mph-p6004882.mph.morpho.com/Stage2019/Data/1117/Sea/skel_mkkp1
# python.exe D:/270219_STAGE/Utils/make_skel_from_bin.py -i //mph-p6004882.mph.morpho.com/Stage2019/Data/NIST27/Sea/bin_mkkp1 -o //mph-p6004882.mph.morpho.com/Stage2019/Data/NIST27/Sea/skel_mkkp1
# python.exe D:/270219_STAGE/Utils/make_skel_from_bin.py -i //mph-p6004882.mph.morpho.com/Stage2019/Data/NIST27/Ref/seg_bin_kp1 -o //mph-p6004882.mph.morpho.com/Stage2019/Data/NIST27/Ref/seg_skel_kp1
# python.exe D:/270219_STAGE/Utils/make_skel_from_bin.py -i //mph-p6004882.mph.morpho.com/Stage2019/Data/NIST14/Sea/seg_bin_kp1 -o //mph-p6004882.mph.morpho.com/Stage2019/Data/NIST14/Sea/seg_skel_kp1
# python.exe D:/270219_STAGE/Utils/make_skel_from_bin.py -i //mph-p6004882.mph.morpho.com/Stage2019/Data/NIST14/Ref/seg_bin_kp1 -o //mph-p6004882.mph.morpho.com/Stage2019/Data/NIST14/Ref/seg_skel_kp1
# python.exe D:/270219_STAGE/Utils/make_skel_from_bin.py -i //mph-p6004882.mph.morpho.com/Stage2019/Data/Base_synthetique_Gan/v2_with_background_2000/synth_lt_bin_mkkp1 -o //mph-p6004882.mph.morpho.com/Stage2019/Data/Base_synthetique_Gan/v2_with_background_2000/synth_lt_skel_mkkp1
# python.exe D:/270219_STAGE/Utils/make_skel_from_bin.py -i //mph-p6004882.mph.morpho.com/Stage2019/Data/Base_synthetique_Gan/v2_2000/synth_lt_bin_mkkp1 -o //mph-p6004882.mph.morpho.com/Stage2019/Data/Base_synthetique_Gan/v2_2000/synth_lt_skel_mkkp1

argparser = argparse.ArgumentParser(description='compute skeletons from binary images')
argparser.add_argument( '-i',  '--indir',  help='path to input directory ')
argparser.add_argument( '-o',  '--outdir',  help='path to output directory ')


def repare_skel(skel_filled):
    convolved_1 = convolve_and_identify(skel_filled, np.array([[0, 0, 0],
                                                               [0, -1, 1],
                                                               [0, 1, -1]]))

    convolved_2 = convolve_and_identify(skel_filled, np.array([[0, 0, 0],
                                                               [1, -1, 0],
                                                               [-1, 1, 0]]))
    skel_filled = skel_filled + convolved_1 + convolved_2
    skel_filled[skel_filled > 255] = 255

    skel_filled_v2 = skel_filled.copy()
    skel_filled_v2 = skel_filled_v2 / 255

    for i in range(5):
        square_1 = convolve_and_identify(skel_filled_v2, np.array([[1, 1, 0],
                                                                   [1, 1, 0],
                                                                   [0, 0, 0]]), value=4)

        square_2 = convolve_and_identify(skel_filled_v2, np.array([[0, 1, 1],
                                                                   [0, 1, 1],
                                                                   [0, 0, 0]]), value=4)

        square_3 = convolve_and_identify(skel_filled_v2, np.array([[0, 0, 0],
                                                                   [1, 1, 0],
                                                                   [1, 1, 0]]), value=4)

        square_4 = convolve_and_identify(skel_filled_v2, np.array([[0, 0, 0],
                                                                   [0, 1, 1],
                                                                   [0, 1, 1]]), value=4)

        squares = (square_1 + square_2 + square_3 + square_4)

        neigh = convolve_and_identify(skel_filled_v2, np.array([[0, 1, 0],
                                                                [1, 0, 1],
                                                                [0, 1, 0]]), value=2)
        neigh[skel_filled_v2 == 0] = 0
        neigh = neigh / 255

        delete = np.logical_and(squares, neigh)

        skel_filled_v2[delete] = 0

    delete_ponctus = convolve_and_identify(skel_filled_v2, np.array([[-10, -10, -10],
                                                                     [1, 1, 1],
                                                                     [-10, 1, -10],
                                                                     [-10, -10, -10],
                                                                     [0, 0, 0]]), value=4)

    skel_filled_v2[delete_ponctus == 255] = 0

    delete_solo = convolve_and_identify(skel_filled_v2, np.array([[-10, -10, -10],
                                                                  [-10, 1, -10],
                                                                  [-10, -10, -10]]), value=1)

    skel_filled_v2[delete_solo == 255] = 0

    fill_hole = convolve_and_identify(skel_filled_v2, np.array([[-10, 1, -10],
                                                                [-10, -10, -10],
                                                                [-10, 1, -10]]), value=2)

    skel_filled_v2[fill_hole == 255] = 1

    skel_filled_v2 = skel_filled_v2 * 255

    return skel_filled_v2


def convolve_and_identify(image_skel, kernel_conv, value=510):
    convolved = convolve2d(image_skel, kernel_conv, mode="same")
    convolved[convolved != value] = 0
    convolved[convolved == value] = 255
    return convolved


def _main_(args):
    if not os.path.isdir(args.indir):
        print('ERROR : invalid input directory.\n\n' + args.indir + ' is not a valid directory!\n\n')
        exit(1)
    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)

    files = os.listdir(args.indir)

    n = len(files)
    if n != 0:
        for i in range(n):
            print('iteration ' + str(i) + '/' + str(n-1) + '\n')
            binary_img = cv2.imread(args.indir + '/' + files[i], cv2.IMREAD_GRAYSCALE)
            #bool_img = img_as_bool(binary_img)
            # vec = np.unique(binary_img)
            mask=np.where(binary_img==1) # fond
            binary_img[mask]=-1

            mask = np.where(binary_img == 0) # ridges (vallées = 255)
            binary_img[mask] = 1

            mask = np.where(binary_img == 255)  # ridges (vallées = 255)
            binary_img[mask] = 0

            mask = np.where(binary_img == -1)  # ridges (vallées = 255)
            binary_img[mask] = 0

            skeleton_img = skeletonize(binary_img)

            #skeleton_img_2 = morphology.medial_axis(bool_img)

            skeleton_img_4connexity = repare_skel(skeleton_img*255)

            file_out = files[i].replace('_bin','_skel')
            cv2.imwrite(args.outdir + '/' + file_out, skeleton_img_4connexity)

    else:
        print('ERROR : no file in directory.\n\n' + args.indir + ' is not a valid directory!\n\n')
        exit(1)

    return


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
