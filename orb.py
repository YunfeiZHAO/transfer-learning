import cv2
from skimage.feature import (match_descriptors, ORB, plot_matches)

""" Clear matches for which NN ratio is > than threshold """
def filter_distance(matches):
    ratio = 0.75
    dist = [m.distance for m in matches]
    thres_dist = (sum(dist) / len(dist)) * ratio

    # keep only the reasonable matches
    sel_matches = [m for m in matches if m.distance < thres_dist]
    #print('#selected matches:%d (out of %d)' % (len(sel_matches), len(matches)))
    return sel_matches

def orb_code(img, mode='cv2'):

    if mode=='sk':
        descriptor_extractor = ORB(n_keypoints=100)
        descriptor_extractor.detect_and_extract(img)
        kpts = descriptor_extractor.keypoints
        desc = descriptor_extractor.descriptors
    elif mode=='cv2':
        orb = cv2.ORB_create(1000)
        kpts, desc = orb.detectAndCompute(img, None)
    else:
        print('ORB mode is either sk or cv2')

    return kpts, desc

def orb_match(desc1, desc2, mode='cv2'):
    if mode=='sk':
        matches = match_descriptors(desc1, desc2, cross_check=True)
    elif mode=='cv2':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = filter_distance(matches)
    else:
        print('ORB mode is either sk or cv2')

    return matches



if __name__ == '__main__':



    # Test Sarah
    img1 = cv2.imread('Databases/db_C/00002225_02.png',0)
    img2 = cv2.imread('Databases/db_F/00002225_02.png',0)
    img3 = cv2.imread('Databases/db_H/00002225_02.png',0)
    img4 = cv2.imread('Databases/db_M/00002225_02.png',0)

    jmg1 = cv2.imread('Databases/db_C/00002229_02.png',0)
    jmg2 = cv2.imread('Databases/db_F/00002241_02.png',0)
    jmg3 = cv2.imread('Databases/db_H/00002225_01.png',0)
    jmg4 = cv2.imread('Databases/db_M/00002229_07.png',0)

    # cv2.imshow('test1', img1)
    # cv2.imshow('test2', img2)
    # cv2.waitKey()

    kpts1, desc1 = orb_code(img1)
    out_img1 = img1.copy()
    out_img1 = cv2.drawKeypoints(img1, kpts1, out_img1, color=(255,0,0), flags=0)
    # cv2.imshow('kpts1', out_img1)
    kpts2, desc2 = orb_code(img2)
    out_img2 = img2.copy()
    out_img2 = cv2.drawKeypoints(img2, kpts2, out_img2, color=(255,0,0), flags=0)
    # cv2.imshow('kpts2', out_img2)
    _, desc3 = orb_code(img3)
    _, desc4 = orb_code(img4)

    cv2.waitKey()


    _, jesc1 = orb_code(jmg1)
    _, jesc2 = orb_code(jmg2)
    _, jesc3 = orb_code(jmg3)
    _, jesc4 = orb_code(jmg4)


    matches12 = orb_match(desc1, desc2)
    matches13 = orb_match(desc1, desc3)
    matches14 = orb_match(desc1, desc4)
    matches23 = orb_match(desc2, desc3)
    matches24 = orb_match(desc3, desc4)
    matches34 = orb_match(desc3, desc4)

    mout = cv2.drawMatches(img1, kpts1, img2, kpts2, matches12, img1)
    cv2.imshow('m', mout)
    cv2.waitKey()

    print("nb matches", len(matches12), len(matches13), len(matches14),
          len(matches23), len(matches23), len(matches34))


    jatches12 = orb_match(desc1, jesc2)
    jatches13 = orb_match(desc1, jesc3)
    jatches14 = orb_match(desc1, jesc4)
    jatches23 = orb_match(desc2, jesc3)
    jatches24 = orb_match(desc3, jesc4)
    jatches34 = orb_match(desc3, jesc4)

    print("nb jatches", len(jatches12), len(jatches13), len(jatches14),
          len(jatches23), len(jatches23), len(jatches34))

    print("hello")