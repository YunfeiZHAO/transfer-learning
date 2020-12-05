import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from math import *
from skimage.morphology import skeletonize
from make_skel_from_bin import repare_skel
from matplotlib.patches import Circle
import copy
import math
import json
from orb import (orb_code, orb_match)
from skimage.filters import threshold_multiotsu
from skimage.feature import (match_descriptors, corner_harris, corner_peaks, ORB, plot_matches)
from skimage.feature import local_binary_pattern, blob, orb, corner, brief, haar, censure, peak, texture, hog
from sys import exit

def FP_binarize_simple(img, threshold=128):
    retval, bin_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    return bin_img

def remove_squares(skeleton):
    square_filter = np.array([[1, 1],
                              [1, 1]], dtype=np.uint8)

    initial_ret = cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=4)[0]

    squares = cv2.filter2D(skeleton / 255, -1, square_filter, borderType=cv2.BORDER_CONSTANT)
    squares_right_bottom_coordinates = np.where(squares == 4)
    squares_right_bottom_coordinates = list(
        zip(squares_right_bottom_coordinates[0], squares_right_bottom_coordinates[1]))

    if squares_right_bottom_coordinates == []:
        return skeleton

    for square in squares_right_bottom_coordinates:
        cells = [square, (square[0] - 1, square[1] - 1), (square[0] - 1, square[1]), (square[0], square[1] - 1)]
        for cell in cells:
            diagonal_neighbors = [(cell[0] + 1, cell[1] + 1), (cell[0] - 1, cell[1] - 1), (cell[0] + 1, cell[1] - 1),
                                  (cell[0] - 1, cell[1] + 1)]
            for diagonal_neighbor in diagonal_neighbors:
                if skeleton[diagonal_neighbor] == 0:
                    creates_new_square = False
                    skeleton[diagonal_neighbor] = 255
                    skeleton[cell] = 0
                    if cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=4)[0] != initial_ret:
                        skeleton[diagonal_neighbor] = 0
                        skeleton[cell] = 255
                        continue  # we cannot use this neighbor as it breaks the connexity
                    array_centered_on_neighbour = skeleton[
                                                  diagonal_neighbor[0] - 1:diagonal_neighbor[0] + 2,
                                                  diagonal_neighbor[1] - 1:diagonal_neighbor[1] + 2]

                    if np.sum(cv2.filter2D(array_centered_on_neighbour / 255, -1, square_filter, borderType=cv2.BORDER_CONSTANT)>=4) > 0:
                        creates_new_square = True
                        break
                    if creates_new_square:
                        skeleton[diagonal_neighbor] = 0
                        skeleton[cell] = 255
                        continue  # we cannot use this neighbor as it creates a new square
                    else:  # if for this cell, the neighbor does not beak connexity and does not create any new square, we can keep this change
                        return skeleton

def akaze_code(img):
    akaze = cv2.AKAZE_create()
    kpts, desc = akaze.detectAndCompute(img, None)

    return kpts, desc

def akaze_match_only(matcher, desc1, desc2):
    nn_matches = matcher.knnMatch(desc1, desc2, 2)

    return nn_matches

def clean_matched_desc(kpts1, kpts2, nn_matches):
    nn_match_ratio = 0.90  # Nearest neighbor matching ratio
    inlier_threshold = 2.5  # Distance threshold to identify inliers with homography check

    matched1 = []
    matched2 = []
    good = []
    notes = []
    for m, n in nn_matches:
        if m.distance < nn_match_ratio * n.distance:
            matched1.append(kpts1[m.queryIdx])
            matched2.append(kpts2[m.trainIdx])
            good.append([m])
            notes.append(1.00 * n.distance / (0.10 + m.distance))

    if len(good) > 5:
        src_pts = np.float32([kpts1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpts2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # else:
    #     print("Not enough matches are found - %d/%d" % (len(good), 10))

    M_scale_est = sqrt((M[0, 0] * M[0, 0]) + (M[1, 1] * M[1, 1]) / 2)
    homography = M
    inliers1 = []
    inliers2 = []
    good_matches = []
    total_note = 0.0
    for i, m in enumerate(matched1):
        col = np.ones((3, 1), dtype=np.float64)
        col[0:2, 0] = m.pt
        col = np.dot(homography, col)
        col /= col[2, 0]
        dist = sqrt(pow(col[0, 0] - matched2[i].pt[0], 2) + \
                    pow(col[1, 0] - matched2[i].pt[1], 2))
        if dist < inlier_threshold:
            good_matches.append(cv2.DMatch(len(inliers1), len(inliers2), 0))
            inliers1.append(matched1[i])
            inliers2.append(matched2[i])
            total_note += notes[i]

    inlier_ratio = len(inliers1) / float(len(matched1))

    return len(matched1), len(inliers1), inlier_ratio, total_note

def akaze_match(kpts1, desc1, kpts2, desc2):
    nn_match_ratio = 0.90  # Nearest neighbor matching ratio
    inlier_threshold = 2.5  # Distance threshold to identify inliers with homography check

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_matches = matcher.knnMatch(desc1, desc2, 2)

    matched1 = []
    matched2 = []
    good = []
    notes = []
    for m, n in nn_matches:
        if m.distance < nn_match_ratio * n.distance:
            matched1.append(kpts1[m.queryIdx])
            matched2.append(kpts2[m.trainIdx])
            good.append([m])
            notes.append(1.00 * n.distance / (0.10 + m.distance))

    if len(good) > 5:
        src_pts = np.float32([kpts1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpts2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # else:
    #     print("Not enough matches are found - %d/%d" % (len(good), 10))

    M_scale_est = sqrt((M[0, 0] * M[0, 0]) + (M[1, 1] * M[1, 1]) / 2)
    homography = M
    inliers1 = []
    inliers2 = []
    good_matches = []
    total_note = 0.0
    for i, m in enumerate(matched1):
        col = np.ones((3, 1), dtype=np.float64)
        col[0:2, 0] = m.pt
        col = np.dot(homography, col)
        col /= col[2, 0]
        dist = sqrt(pow(col[0, 0] - matched2[i].pt[0], 2) + \
                    pow(col[1, 0] - matched2[i].pt[1], 2))
        if dist < inlier_threshold:
            good_matches.append(cv2.DMatch(len(inliers1), len(inliers2), 0))
            inliers1.append(matched1[i])
            inliers2.append(matched2[i])
            total_note += notes[i]

    inlier_ratio = len(inliers1) / float(len(matched1))

    return len(desc1), len(desc2), len(matched1), len(inliers1), inlier_ratio, total_note

def akaze_code_and_match(img1, img2):
    nn_match_ratio = 0.90  # Nearest neighbor matching ratio
    inlier_threshold = 2.5  # Distance threshold to identify inliers with homography check

    akaze = cv2.AKAZE_create()
    kpts1, desc1 = akaze.detectAndCompute(img1, None)
    kpts2, desc2 = akaze.detectAndCompute(img2, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_matches = matcher.knnMatch(desc1, desc2, 2)

    matched1 = []
    matched2 = []
    good = []
    notes = []
    for m, n in nn_matches:
        if m.distance < nn_match_ratio * n.distance:
            matched1.append(kpts1[m.queryIdx])
            matched2.append(kpts2[m.trainIdx])
            good.append([m])
            notes.append(1.00 * n.distance / (0.10 + m.distance))

    if len(good) > 5:
        src_pts = np.float32([kpts1[m[0].queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpts2[m[0].trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    else:
        print("Not enough matches are found - %d/%d" % (len(good), 10))

    M_scale_est = sqrt((M[0, 0] * M[0, 0]) + (M[1, 1] * M[1, 1]) / 2)
    print('Homography scale estimation: ', M_scale_est)
    homography = M
    inliers1 = []
    inliers2 = []
    good_matches = []
    total_note = 0.0
    for i, m in enumerate(matched1):
        col = np.ones((3, 1), dtype=np.float64)
        col[0:2, 0] = m.pt
        col = np.dot(homography, col)
        col /= col[2, 0]
        dist = sqrt(pow(col[0, 0] - matched2[i].pt[0], 2) + \
                    pow(col[1, 0] - matched2[i].pt[1], 2))
        if dist < inlier_threshold:
            good_matches.append(cv2.DMatch(len(inliers1), len(inliers2), 0))
            inliers1.append(matched1[i])
            inliers2.append(matched2[i])
            total_note += notes[i]

    inlier_ratio = len(inliers1) / float(len(matched1))

    return len(desc1), len(desc2), len(matched1), len(inliers1), inlier_ratio, total_note

def loop_on_skel(local_used_map, nb_loop, cur_length, xx, yy):

    if nb_loop == 0:
        return xx, yy, cur_length

    aa = xx
    bb = yy
    local_used_map[xx, yy] = 0
    nb_loop = nb_loop - 1
    cur_length = cur_length + 1

    if local_used_map[xx - 1, yy] > 0:
        aa, bb, cur_length = loop_on_skel(local_used_map, nb_loop, cur_length, xx-1, yy)
    if local_used_map[xx + 1, yy] > 0:
        aa, bb, cur_length = loop_on_skel(local_used_map, nb_loop, cur_length, xx + 1, yy)
    if local_used_map[xx, yy - 1] > 0:
        aa, bb, cur_length = loop_on_skel(local_used_map, nb_loop, cur_length, xx, yy - 1)
    if local_used_map[xx, yy + 1] > 0:
        aa, bb, cur_length = loop_on_skel(local_used_map, nb_loop, cur_length, xx, yy + 1)

    return aa, bb, cur_length

def orientation_modulo(orient):
    if orient < -math.pi:
        orient = orient + 2 * math.pi
    if orient > math.pi:
        orient = orient - 2 * math.pi

    return orient

def orientation_middle(or1, or2):
    or_out = math.atan2( sin(or1) + sin(or2), cos(or1) + cos(or2) )
    return or_out

def estimate_bifurcation_orientation(or1, or2, or3):
    diff12 = fabs(orientation_modulo(or1 - or2))
    diff13 = fabs(orientation_modulo(or1 - or3))
    diff23 = fabs(orientation_modulo(or2 - or3))
    orientation = 'nan'
    if diff12 <= diff13:
        if diff12 <= diff23:
            orientation = orientation_middle(or1, or2)
    if diff13 <= diff12:
        if diff13 <= diff23:
            orientation = orientation_middle(or1, or3)
    if diff23 <= diff12:
        if diff23 <= diff13:
            orientation = orientation_middle(or2, or3)

    return orientation

def compute_mask(img):

    h, w = img.shape
    nb_kernel_iterations = 1
    kernel_erode_size = 15
    kernel_dilate_size = 40
    larger_offset = 100
    larger_offset2 = int(larger_offset / 2)

    # Copy in larger image
    bin_img_larger = np.ones((h + larger_offset, w + larger_offset), np.uint8) * 255
    bin_img_larger[larger_offset2:h + larger_offset2, larger_offset2:w + larger_offset2] = img

    # Erode and dilate more
    #Erode
    kernel_erode = np.ones((kernel_erode_size, kernel_erode_size), np.uint8)  # the square kernel
    mask_dilated = cv2.erode(bin_img_larger, kernel_erode, iterations=nb_kernel_iterations)
    #Dilate
    kernel_dilate = np.ones((kernel_dilate_size, kernel_dilate_size), np.uint8)  # the square kernel
    mask_eroded = cv2.dilate(mask_dilated, kernel_dilate, iterations=nb_kernel_iterations)

    # Invert
    mask_eroded_inverted = 255 - mask_eroded
    new_mask = np.ones((h, w), np.uint8) * 255

    # Back to original dimensions
    output_mask = mask_eroded_inverted[larger_offset2:h + larger_offset2, larger_offset2:w + larger_offset2]

    return output_mask

def morpho_mat_cleaning(img):
    # MorphoMat: Dilatation and Erosion
    nb_kernel_iterations = 1
    kernel_1_size = 1
    kernel_1 = np.ones((kernel_1_size, kernel_1_size), np.uint8)
    kernel_2_size = 1
    kernel_2 = np.ones((kernel_2_size, kernel_2_size), np.uint8)

    hole_size_to_remove = 3

    img_eroded = cv2.erode(img, kernel_1, iterations=nb_kernel_iterations)
    img_dilated = cv2.dilate(img_eroded, kernel_2, iterations=nb_kernel_iterations)
    img_mor = img

    # We fill the small holes in the binary in order not to need to erode too much
    inverted = cv2.bitwise_not(255 - img_mor)
    ret, labels, stats, centroid = cv2.connectedComponentsWithStats(inverted, connectivity=4)  # each hole is a connected component in the inverted image
    # stats[labels][:,:,4] gives us the sizes of all the holes
    #img_holes = np.where(stats[labels][:, :, 4] < img.size * 2e-4, 255, 0)  # we fill the holes whose size is small in proportion to the image size
    img_holes = np.where(stats[labels][:, :, 4] < hole_size_to_remove, 255,
                         0)  # we fill the holes whose size is small in proportion to the image size
    output_img = np.bitwise_or(img_holes, 255 - img_mor).astype(np.uint8)
    output_img = 255 - output_img

    return output_img

def compute_skeleton(img):
    # invert
    inverted_bin_img = 1 - img / 255

    # skeletonize
    skel_img = skeletonize(inverted_bin_img)

    # Fit to 4-connexity
    skeleton_img_4connexity = repare_skel(skel_img * 255)

    # Removing the 4,4 squares left
    skeleton_img_4connexity_without_squares = remove_squares(skeleton_img_4connexity)

    return skeleton_img_4connexity_without_squares

def extract_pk_from_skel(img_skel, img_mask):
    line_min_length = 20 # 12
    count_neighbors_K = np.array([[0, 1, 0],
                                  [1, 4, 1],
                                  [0, 1, 0]], dtype=np.uint8)
    neighbors_count = cv2.filter2D(img_skel / 255, -1, count_neighbors_K, borderType=cv2.BORDER_CONSTANT)

    bifurcations = np.where(neighbors_count >= 7)
    bifurcations = list(zip(bifurcations[0], bifurcations[1]))  # zip the 2 arrays to get the exact coordinates

    skeleton_without_bifurcations = np.copy(img_skel)
    for bifurcation in bifurcations:
        skeleton_without_bifurcations[bifurcation[0], bifurcation[1]] = 0

    skeleton_without_bifurcations = skeleton_without_bifurcations.astype(np.uint8)

    # Find ends of lines
    ends_of_lines = np.where(neighbors_count == 5)
    ends_of_lines = list(zip(ends_of_lines[0], ends_of_lines[1]))  # zip the 2 arrays to get the exact coordinates

    # Label connected components
    ret, labels = cv2.connectedComponents(skeleton_without_bifurcations, connectivity=4) # compute connected components

    # Storing the points of each component in a dictionary
    components = dict()
    lines_to_delete = []
    for i in range(1, ret):
        components[i] = np.where(labels == i)
        components[i] = list(zip(components[i][0], components[i][1]))
        # for each component we have all the points that belong to it

        for point in components[i]:
            if point in ends_of_lines:  # if the line ends with at least one end of line (not two bifurcations)
                if len(components[i]) < line_min_length:  # if the line is short
                    lines_to_delete.append(i)  # we will delete this line
                    continue

    # Deleting the irrelevant lines
    new_skeleton = np.copy(img_skel)
    new_skeleton = new_skeleton.astype(np.uint8)
    for line in lines_to_delete:
        for point in components[line]:
            new_skeleton[point] = 0

    # We have to reskelotnize to remove the possible square of pixels
    final_skeleton_reskeletonize = repare_skel(new_skeleton).astype(np.uint8)

    # 2nd pass
    new_skeleton2 = np.copy(new_skeleton)
    neighbors_count2 = cv2.filter2D(new_skeleton2 / 255, -1, count_neighbors_K, borderType=cv2.BORDER_CONSTANT)

    bifurcations2 = np.where(neighbors_count2 >= 7)
    bifurcations2 = list(zip(bifurcations2[0], bifurcations2[1]))  # zip the 2 arrays to get the exact coordinates

    # Find ends of lines
    ends_of_lines2 = np.where(neighbors_count2 == 5)
    ends_of_lines2 = list(zip(ends_of_lines2[0], ends_of_lines2[1]))  # zip the 2 arrays to get the exact coordinates

    skeleton_without_bifurcations2 = np.copy(new_skeleton2)
    for bifurcation in bifurcations2:
        skeleton_without_bifurcations2[bifurcation[0], bifurcation[1]] = 0

    skeleton_without_bifurcations2 = skeleton_without_bifurcations2.astype(np.uint8)

    skeleton_without_minutiae = np.copy(skeleton_without_bifurcations2)
    for eol in ends_of_lines2:
        skeleton_without_minutiae[eol[0], eol[1]] = 0

    # Compute directions of minutiae
    minutiae3 = []
    # ends of lines
    for xx, yy in ends_of_lines2:
        if img_mask[xx, yy] > 0:
            length = 0
            if skeleton_without_minutiae[xx - 1, yy] > 0:
                local_used_map = copy.deepcopy(skeleton_without_minutiae)
                aa, bb, length = loop_on_skel(local_used_map, 10, 0, xx - 1, yy)
            if skeleton_without_minutiae[xx + 1, yy] > 0:
                local_used_map = copy.deepcopy(skeleton_without_minutiae)
                aa, bb, length = loop_on_skel(local_used_map, 10, 0, xx + 1, yy)
            if skeleton_without_minutiae[xx, yy - 1] > 0:
                local_used_map = copy.deepcopy(skeleton_without_minutiae)
                aa, bb, length = loop_on_skel(local_used_map, 10, 0, xx, yy - 1)
            if skeleton_without_minutiae[xx, yy + 1] > 0:
                local_used_map = copy.deepcopy(skeleton_without_minutiae)
                aa, bb, length = loop_on_skel(local_used_map, 10, 0, xx, yy + 1)
            if length >= 10:
                local_orientation = atan2(bb - yy, aa - xx)
                local_minutiae = [xx, yy, local_orientation, 'end_of_line']
                minutiae3.append(local_minutiae)

    # bifurcations
    for xx, yy in bifurcations2:
        if img_mask[xx, yy] > 0:
            local_3_or = []
            if skeleton_without_minutiae[xx - 1, yy] > 0:
                local_used_map = copy.deepcopy(skeleton_without_minutiae)
                aa, bb, length = loop_on_skel(local_used_map, 10, 0, xx - 1, yy)
                local_3_or.append([aa, bb, length])
            if skeleton_without_minutiae[xx + 1, yy] > 0:
                local_used_map = copy.deepcopy(skeleton_without_minutiae)
                aa, bb, length = loop_on_skel(local_used_map, 10, 0, xx + 1, yy)
                local_3_or.append([aa, bb, length])
            if skeleton_without_minutiae[xx, yy - 1] > 0:
                local_used_map = copy.deepcopy(skeleton_without_minutiae)
                aa, bb, length = loop_on_skel(local_used_map, 10, 0, xx, yy - 1)
                local_3_or.append([aa, bb, length])
            if skeleton_without_minutiae[xx, yy + 1] > 0:
                local_used_map = copy.deepcopy(skeleton_without_minutiae)
                aa, bb, length = loop_on_skel(local_used_map, 10, 0, xx, yy + 1)
                local_3_or.append([aa, bb, length])

            if len(local_3_or) == 3:
                local_orientation1 = atan2(local_3_or[0][1] - yy, local_3_or[0][0] - xx)
                local_orientation2 = atan2(local_3_or[1][1] - yy, local_3_or[1][0] - xx)
                local_orientation3 = atan2(local_3_or[2][1] - yy, local_3_or[2][0] - xx)
                local_orientation = estimate_bifurcation_orientation(local_orientation1, local_orientation2,
                                                                     local_orientation3)
                local_minutiae = [xx, yy, local_orientation, 'bifurcation']
                minutiae3.append(local_minutiae)
            #else : problem ?

    return minutiae3

def clean_minutiae(minutiae):
    minutiae2 = minutiae[:]

    dist_thresh = 20.0 * 20.0 # in pixels * pixels
    minutiae_to_delete = []
    for xx1, yy1, orient1, type1 in minutiae:
        for xx2, yy2, orient2, type2 in minutiae:
            if( xx1 != xx2 or yy1 != yy2 ):
                xy_dist = ((xx1.astype(np.float) - xx2.astype(np.float))*(xx1.astype(np.float) - xx2.astype(np.float)) + (yy1.astype(np.float) - yy2.astype(np.float))*(yy1.astype(np.float) - yy2.astype(np.float)))
                if( xy_dist < dist_thresh):
                    # Remove both
                    m1 = [xx1, yy1, orient1, type1]
                    m2 = [xx2, yy2, orient2, type2]
                    minutiae_to_delete.append(m1)
                    minutiae_to_delete.append(m2)

    for pk in minutiae_to_delete:
        if (pk in minutiae2):
            minutiae2.remove(pk)

    return minutiae2

def extract_minutiae(img, dump=0):

    # Image enhancement
    img_enh = img

    # Binarize
    img_blur = cv2.GaussianBlur(img_enh, (3, 3), 0)
    bin_img = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Binary image cleaning
    img_mor = morpho_mat_cleaning(bin_img)

    # Compute mask
    img_mask = compute_mask(img_mor)

    # Compute skeleton
    img_skel = compute_skeleton(img_mor)

    # Extract minutiae from skeleton
    minutiae = extract_pk_from_skel(img_skel, img_mask)

    # Clean minutiae
    minutiae_cleaned = clean_minutiae(minutiae)

    # Dump images
    if dump == 1:
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4)
        ax0.imshow(img, cmap='gray', interpolation='nearest')
        ax1.imshow(bin_img, cmap='gray', interpolation='nearest')
        ax2.imshow(img_skel, cmap='gray', interpolation='nearest')
        ax3.imshow(img_skel, cmap='gray', interpolation='nearest')

        # Dump minutiae
        for xx, yy, orient, type in minutiae_cleaned:
            if type == 'bifurcation':
                local_color = 'blue'
            else:
                local_color = 'green'
            circ = Circle((yy, xx), 3, color=local_color)
            ax3.add_patch(circ)

        plt.show()
        cv2.waitKey()

    return minutiae_cleaned

def match_minutiae(minutiae_1, width_1, height_1, minutiae_2, width_2, height_2):

    # Vote in the hough accumulator
    hough_accumulator = np.zeros((40, 100, 100), dtype=np.uint64)
    size_vote = 1
    for i in range(len(minutiae_1)):
        xx1, yy1, orient1, type1 = minutiae_1[i]
        xx1_centered = width_1 / 2 - xx1
        yy1_centered = height_1 / 2 - yy1
        for j in range(len(minutiae_2)):
            xx2, yy2, orient2, type2 = minutiae_2[j]

            transf_orient = orient2 - orient1
            if transf_orient < -math.pi:
                transf_orient += 2.0 * math.pi
            if transf_orient > math.pi:
                transf_orient -= 2.0 * math.pi

            xx2_centered = width_2 / 2 - xx1
            yy2_centered = height_2 / 2 - yy2

            translation_h = yy2_centered - (cos(transf_orient) * yy1_centered) - (sin(transf_orient) * xx1_centered)
            translation_w = xx2_centered - (sin(transf_orient) * yy1_centered) + (cos(transf_orient) * xx1_centered)

            transf_orient = int(4.0 * (math.pi + transf_orient))
            translation_h = 50 + int(translation_h / 20)
            translation_w = 50 + int(translation_w / 20)

            # Vote Hough
            hough_accumulator[transf_orient, translation_h, translation_w] += 1

            for aa in range(transf_orient-size_vote,transf_orient+size_vote):
                for bb in range(translation_h - size_vote, translation_h + size_vote):
                    for cc in range(translation_w - size_vote, translation_w + size_vote):
                            hough_accumulator[aa, bb, cc] += 1

    score = np.max(hough_accumulator)

    return score

def image_detect_and_compute(detector, img_name):
    """Detect and compute interest points and their descriptors."""
    img = cv2.imread(os.path.join(img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = detector.detectAndCompute(img, None)
    return img, kp, des

def draw_image_matches(detector, img1_name, img2_name, nmatches=10):
    """Draw ORB feature matches of the given two images."""
    img1, kp1, des1 = image_detect_and_compute(detector, img1_name)
    img2, kp2, des2 = image_detect_and_compute(detector, img2_name)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)  # Sort matches by distance.  Best come first.

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:nmatches], img2, flags=2)  # Show top 10 matches
    plt.figure(figsize=(16, 16))
    plt.title(type(detector))
    plt.imshow(img_matches);
    plt.show()

def compare_minutiae_sets( minutiae, gt_minutiae):
    nb_minutiae = len(minutiae)
    nb_gt_minutiae = len(gt_minutiae)

    dist_tolerance_2 = 5 * 5 + 5 * 5

    nb_common = 0
    for xx, yy, orient, type in minutiae:
        for xx_gt, yy_gt, orient_gt in gt_minutiae:
            xy_dist = np.float32((xx - xx_gt) * (xx - xx_gt) + (yy - yy_gt) * (yy - yy_gt))
            if xy_dist < dist_tolerance_2:
                nb_common += 1

    return nb_common, nb_minutiae, nb_gt_minutiae

if __name__ == '__main__':

    mode = 'simple'

    # Simple tests
    if mode == 'simple':
        img1 = cv2.imread('Databases/db_C/00002225_04.png', 0)
        img2 = cv2.imread('Databases/db_H/00002225_04.png', 0)

        simple_mode = 'orb'
        #simple_mode = 'minutiae'
        #simple_mode = 'akaze'

        # Minutiae example
        if simple_mode == 'minutiae':
            minutiae_1 = extract_minutiae(img1)
            minutiae_2 = extract_minutiae(img2)
            minutiae_score = match_minutiae(minutiae_1, img1.shape[1], img1.shape[0], minutiae_2, img2.shape[1], img2.shape[0])
            print(minutiae_score)
            exit(0)

        # Orb matching example
        elif simple_mode == 'orb':
            orb = cv2.ORB_create()
            draw_image_matches(orb, 'Databases/db_C/00002225_04.png', 'Databases/db_H/00002225_04.png', nmatches=100)
            exit(0)

        # Akaze matching example
        elif simple_mode == 'akaze':
            akaze = cv2.AKAZE_create()
            draw_image_matches(akaze, 'Databases/db_C/00002225_04.png', 'Databases/db_H/00002225_04.png', nmatches=100)
            exit(0)

        else:
            print('Incorrect simple_mode given')

        exit(0)

    # Code, match analyze databases
    elif mode == 'code_and_match':
        use_orb = 1
        use_akaze = 0
        use_minutiae = 0
        dump_matching_result = 0

        # Input data
        input_dir_1 = 'Databases/db_C/'
        input_dir_2 = 'Databases/db_H/'

        liste_img_1 = 'Databases/liste_images.txt'
        liste_img_2 = 'Databases/liste_images.txt'

        match_list_name = 'Databases/liste_match.txt'
        #non_match_list_name = 'Databases/liste_non_match.txt'
        non_match_list_name = 'Databases/liste_non_match_mini.txt'

        # Read images
        print('Read images 1')
        imgs_1 = []
        with open(liste_img_1) as f:
            for line in f:
                inner_list = [elt.strip() for elt in line.split(' ')]
                img_name=input_dir_1 + inner_list[0] + '.png'
                d = {'name': inner_list[0], 'image': cv2.imread(img_name, 0)}
                imgs_1.append( d )

        print('Read images 2')
        imgs_2 = []
        with open(liste_img_2) as f:
            for line in f:
                inner_list = [elt.strip() for elt in line.split(' ')]
                img_name = input_dir_2 + inner_list[0] + '.png'
                d = {'name': inner_list[0], 'image': cv2.imread(img_name, 0)}
                imgs_2.append(d)

        # Code
        print('Coding imgs_1...')
        for img in imgs_1:
            img['width'] = img['image'].shape[1]
            img['height'] = img['image'].shape[0]
            # Akaze
            if use_akaze == 1:
                kpts, desc = akaze_code(img['image'])
                img['akaze_kpts'] = kpts
                img['akaze_desc'] = desc
            # Orb
            if use_orb == 1:
                kpts, desc = orb_code(img['image'])
                img['orb_kpts'] = kpts
                img['orb_desc'] = desc
            # Extract minutiae
            if use_minutiae == 1:
                minutiae = extract_minutiae(img['image'])
                img['minutiae'] = minutiae

        print('Coding imgs_2...')
        for img in imgs_2:
            img['width'] = img['image'].shape[1]
            img['height'] = img['image'].shape[0]
            # Akaze
            if use_akaze == 1:
                kpts, desc = akaze_code(img['image'])
                img['akaze_kpts'] = kpts
                img['akaze_desc'] = desc
            # Orb
            if use_orb == 1:
                kpts, desc = orb_code(img['image'])
                img['orb_kpts'] = kpts
                img['orb_desc'] = desc
            # Extract minutiae
            if use_minutiae == 1:
                minutiae = extract_minutiae(img['image'])
                img['minutiae'] = minutiae

        # Lists
        # Match
        match_list = []
        with open(match_list_name) as f:
            for line in f:
                inner_list = [elt.strip() for elt in line.split(' ')]
                d = {'search': inner_list[0], 'reference': inner_list[1]}
                match_list.append(d)

        non_match_list = []
        with open(non_match_list_name) as f:
            for line in f:
                inner_list = [elt.strip() for elt in line.split(' ')]
                d = {'search': inner_list[0], 'reference': inner_list[1]}
                non_match_list.append(d)

        print('Match mates')
        results_mates = []
        for couple in match_list:
            # Get indices
            for i in range(len(imgs_1)):
                if imgs_1[i]['name'] == couple['search']:
                    index_search = i
            for i in range(len(imgs_2)):
                if imgs_2[i]['name'] == couple['reference']:
                    index_reference = i

            # Akaze matching
            d = 0
            if use_akaze == 1:
                a, b, c, d, e, f = akaze_match(imgs_1[index_search]['akaze_kpts'], imgs_1[index_search]['akaze_desc'],
                                       imgs_2[index_reference]['akaze_kpts'], imgs_2[index_reference]['akaze_desc'])

            # Orb matching
            orb_matches = []
            if use_orb == 1:
                orb_matches = orb_match(imgs_1[index_search]['orb_desc'], imgs_2[index_reference]['orb_desc'])

            # Minutiae matching
            minutiae_score = 0
            if use_minutiae == 1:
                minutiae_score = match_minutiae(imgs_1[index_search]['minutiae'], imgs_1[index_search]['width'],
                                            imgs_1[index_search]['height'], imgs_2[index_reference]['minutiae'],
                                            imgs_2[index_reference]['width'], imgs_2[index_reference]['height'])

            res = {'search': couple['search'], 'reference': couple['reference'], 'akaze_score': d, 'orb_score': len(orb_matches), 'minutiae_score': minutiae_score}
            results_mates.append(res)

        print('Match non mates')
        results_non_mates = []
        for couple in non_match_list:
            for i in range(len(imgs_1)):
                if imgs_1[i]['name'] == couple['search']:
                    index_search = i
            for i in range(len(imgs_2)):
                if imgs_2[i]['name'] == couple['reference']:
                    index_reference = i
            # Get indices
            for i in range(len(imgs_1)):
                if imgs_1[i]['name'] == couple['search']:
                    index_search = i
            for i in range(len(imgs_2)):
                if imgs_2[i]['name'] == couple['reference']:
                    index_reference = i

            # Akaze matching
            d = 0
            if use_akaze == 1:
                a, b, c, d, e, f = akaze_match(imgs_1[index_search]['akaze_kpts'],
                                           imgs_1[index_search]['akaze_desc'],
                                           imgs_2[index_reference]['akaze_kpts'],
                                           imgs_2[index_reference]['akaze_desc'])

            # Orb matching
            orb_matches = []
            if use_orb == 1:
                orb_matches = orb_match(imgs_1[index_search]['orb_desc'], imgs_2[index_reference]['orb_desc'])

            # Minutiae matching
            minutiae_score = 0
            if use_minutiae == 1:
                minutiae_score = match_minutiae(imgs_1[index_search]['minutiae'], imgs_1[index_search]['width'],
                                            imgs_1[index_search]['height'], imgs_2[index_reference]['minutiae'],
                                            imgs_2[index_reference]['width'], imgs_2[index_reference]['height'])

            res = {'search': couple['search'], 'reference': couple['reference'], 'akaze_score': d,
                   'orb_score': len(orb_matches), 'minutiae_score': minutiae_score}
            results_non_mates.append(res)




        # Dump results
        if dump_matching_result == 1:
            print('Save text results file')
            with open('output_mates.txt', 'w') as fp:
                json.dump(results_mates, fp)

            with open('output_non_mates.txt', 'w') as fp:
                json.dump(results_non_mates, fp)





        print('Compute FAR FRR')
        mates_scores = []
        for res in results_mates:
            if use_akaze == 1:
                mates_scores.append(res['akaze_score'])
            if use_orb == 1:
                mates_scores.append(res['orb_score'])
            if use_minutiae == 1:
                mates_scores.append(res['minutiae_score'])
        non_mates_scores = []
        for res in results_non_mates:
            if use_akaze == 1:
                non_mates_scores.append(res['akaze_score'])
            if use_orb == 1:
                non_mates_scores.append(res['orb_score'])
            if use_minutiae == 1:
                non_mates_scores.append(res['minutiae_score'])

        max = np.max(mates_scores)
        min = 0
        histo_m = np.histogram(mates_scores, int(max), (min, max))[0]
        histo_nm = np.histogram(non_mates_scores, int(max), (min, max))[0]

        print('Nb Mates: ', len(mates_scores), ' Nb Non Mates: ', len(non_mates_scores))
        frr = histo_m.cumsum() / len(mates_scores) * 100.0
        far = histo_nm[::-1].cumsum()[::-1] / len(non_mates_scores) * 100.0

        seuils = [100, 50, 10, 1e-0, 1e-1]
        for thr in seuils:
            print(np.where(far < thr)[0][0], frr[np.where(far < thr)[0][0]], thr)

        exit(0)

    # Results analysis
    elif mode == 'results_analysis':
        with open('output_mates.txt', 'r') as fp:
            results_mates = json.load(fp)

        with open('output_non_mates.txt', 'r') as fp:
            results_non_mates = json.load(fp)

        mates_scores = []
        for res in results_mates:
            mates_scores.append(res['akaze_score'])
        non_mates_scores = []
        for res in results_non_mates:
            non_mates_scores.append(res['akaze_score'])

        max = np.max(mates_scores)
        min = 0
        histo_m = np.histogram(mates_scores, int(max), (min, max))[0]
        histo_nm = np.histogram(non_mates_scores, int(max), (min, max))[0]

        frr = histo_m.cumsum() / len(mates_scores) * 100.0
        far = histo_nm[::-1].cumsum()[::-1] / len(non_mates_scores) * 100.0

        seuils = [100, 10, 1e-0, 1e-1, 1e-2]
        for thr in seuils:
            print(np.where(far < thr)[0][0], frr[np.where(far < thr)[0][0]], thr)

        exit(0)

    # Estimate minutiae detection precision
    elif mode == 'minutiae':
        input_dir_1 = 'Databases/db_C/'
        input_dir_txt_1 = 'Databases/db_C_txt/'
        liste_img_1 = 'Databases/liste_images.txt'

        # Read images
        print('Read images 1')
        imgs_1 = []
        with open(liste_img_1) as f:
            for line in f:
                inner_list = [elt.strip() for elt in line.split(' ')]
                img_name = input_dir_1 + inner_list[0] + '.png'
                d = {'name': inner_list[0], 'image': cv2.imread(img_name, 0), 'id_name': inner_list[0]}
                imgs_1.append(d)

        # Code, extract groundtruth and compare
        print('Coding imgs_1...')
        total_nb_common = 0
        total_nb_minutiae = 0
        total_nb_gt_minutiae = 0
        for img in imgs_1:

            # Compute minutiae
            img['width'] = img['image'].shape[1]
            img['height'] = img['image'].shape[0]
            minutiae = extract_minutiae(img['image'])
            img['minutiae'] = minutiae

            # Read GrountTruth
            gt_filename = input_dir_txt_1 + img['id_name'] + '.txt'
            gt_minutiae = []
            with open(gt_filename) as f:
                for line in f:
                    inner_list = [elt.strip() for elt in line.split(' ')]
                    xx = int(inner_list[1])
                    yy = int(inner_list[0])
                    orient = float(inner_list[2])
                    gt_min = [xx, yy, orient]
                    gt_minutiae.append(gt_min)
            img['gt_minutiae'] = gt_minutiae

            # Compare minutiae sets
            nb_common, nb_minutiae, nb_gt_minutiae = compare_minutiae_sets( minutiae, gt_minutiae)
            total_nb_common += nb_common
            total_nb_minutiae += nb_minutiae
            total_nb_gt_minutiae += nb_gt_minutiae

        total_nb_outliers = total_nb_minutiae - total_nb_common

        print('Nb images: ', len(imgs_1))
        print('Minutiae in common: ', 100.0 * total_nb_common / total_nb_gt_minutiae, ' ( ', total_nb_common, ' / ',
              total_nb_gt_minutiae, ' )')
        print('Minutiae outliers: ', 100.0 * total_nb_outliers / total_nb_gt_minutiae, ' ( ', total_nb_outliers, ' / ',
              total_nb_gt_minutiae, ' )')

        exit(0)

    else:
        print('Mode is either simple, code_and_match, results_analysis or minutiae')

    exit(0)
