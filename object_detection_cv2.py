import cv2 as cv
import numpy as np
import sys


def resize(image):
    scale = 1
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    img_resized = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)
    return img_resized


def show(image, title=' '):
    cv.imshow(f'{title}', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


org_img = resize(cv.imread('dublin.jpg'))
edit_img = resize(cv.imread('dublin_edited.jpg'))
org_img_ac = org_img.copy()
edit_img_ac = edit_img.copy()
# show(org_img_ac)
# show(edit_img_ac)

org_img_ac_gr = cv.cvtColor(org_img_ac, cv.COLOR_BGR2GRAY)
edit_img_ac_gr = cv.cvtColor(edit_img_ac, cv.COLOR_BGR2GRAY)
# show(org_img_ac_gr)
# show(edit_img_ac_gr)

org_img_ac = cv.GaussianBlur(org_img_ac_gr, (5, 5), 0)
edit_img_ac = cv.GaussianBlur(edit_img_ac_gr, (5, 5), 0)
# show(org_img_ac)
# show(edit_img_ac)

org_img_trs, _ = cv.threshold(org_img_ac, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
edit_trs, _ = cv.threshold(edit_img_ac, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

org_img_ac = cv.Canny(org_img_ac_gr, org_img_trs * 0.5, org_img_trs)
edit_img_ac = cv.Canny(edit_img_ac_gr, edit_trs*0.5, edit_trs)
sub = cv.subtract(edit_img_ac, org_img_ac)
# show(sub)

sub = cv.dilate(sub, np.ones((6, 6)), iterations=1)
# show(sub)
sub = cv.erode(sub, np.ones((5, 5)), iterations=3)
# show(sub)
sub = cv.dilate(sub, np.ones((5, 5)), iterations=5)
# show(sub)

contours, _ = cv.findContours(sub, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
rec = []
[rec.append(cv.boundingRect(cv.approxPolyDP(c, 3, True))) for c in contours]
[[int(s) for s in sub] for sub in rec]
img_with_rec = edit_img.copy()
[cv.rectangle(img_with_rec, pt1=(rec[i][0], rec[i][1]),
              pt2=((rec[i][0] + rec[i][2]), (rec[i][1] + rec[i][3])),
              color=(0, 0, 255), thickness=3) for i in range(len(contours))]
show(img_with_rec, 'Founded Kevins')
# cut = img2[rec[1][1]:(rec[1][1] + rec[1][3]), rec[1][0]:(rec[1][0] + rec[1][2])]
# show(cut)

org_crop = []
[org_crop.append(org_img[(rec[i][1] - 5):(rec[i][1]+rec[i][3] + 5),
                 (rec[i][0] - 5):(rec[i][0] + rec[i][2] + 5)]) for i in range(len(rec))]
edit_crop = []
[edit_crop.append(edit_img[(rec[i][1] - 5):(rec[i][1]+rec[i][3] + 5),
                  (rec[i][0] - 5):(rec[i][0]+rec[i][2] + 5)]) for i in range(len(rec))]
# for i in range(len(rec)):
#     show(org_crop[i])
#     show(edit_crop[i])

for org, edit in zip(org_crop, edit_crop):
    # ********************************* THIRD APPROACH
    edit_gr = cv.cvtColor(edit, cv.COLOR_BGR2GRAY)
    org_gr = cv.cvtColor(org, cv.COLOR_BGR2GRAY)
    result = cv.add(cv.subtract(edit_gr, org_gr), cv.subtract(org_gr, edit_gr))
    result = cv.GaussianBlur(result, (3, 3), 8)
    result = cv.GaussianBlur(result, (3, 3), 8)
    result = cv.Canny(result, 0, 20)
    result = cv.dilate(result, np.ones((3, 3)), iterations=4)
    result = cv.erode(result, np.ones((3, 3)), iterations=4)

    edit[result == 0] = [255, 255, 255]
    show(edit)

    # result = cv.Canny(cv.add(cv.subtract(edit_gr, org_gr), cv.subtract(org_gr, edit_gr)), 220, 221)
    # dil = cv.dilate(result, np.ones((3, 3)), iterations=8)
    # er = cv.erode(dil, np.ones((3, 3)), iterations=5)
    # # copy = edit.copy()
    # # edit[er == 0] = [255, 255, 255]
    # # show(edit)
    #
    # unknown = cv.subtract(dil, er)
    # ret, markers = cv.connectedComponents(er)
    # # Add one to all labels so that sure background is not 0, but 1
    # markers = markers + 1
    # # Now, mark the region of unknown with zero
    # markers[unknown == 255] = 0
    # markers = cv.watershed(edit, markers)
    # edit[markers == 1] = [255, 255, 255]
    #
    # # contours, _ = cv.findContours(edit, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # # cv.drawContours(edit, contours, 0, (0, 0, 255))
    # show(edit)
    # #********************************* FIRST APPROACH
    # out = edit
    # org = np.round(org/255, 1)
    # edit = np.round(edit/255, 1)
    # for i in range(edit.shape[0]):
    #     for j in range(edit.shape[1]):
    #         for k in range(edit.shape[2]):
    #             if edit[i][j][k] == org[i][j][k]:
    #                 out[i][j][0] = 0
    #                 out[i][j][1] = 0
    #                 out[i][j][2] = 0
    #                 break

    #********************************* SECOND APPROACH

    # crop = cv.subtract(edit, org)
    # org_gr = cv.cvtColor(org, cv.COLOR_BGR2GRAY)
    # org = cv.GaussianBlur(org_gr, (5, 5), 0)
    # org_trs, _ = cv.threshold(org, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # org = cv.Canny(org_gr, org_trs * 0.5, org_trs)
    #
    # edit_gr = cv.cvtColor(edit, cv.COLOR_BGR2GRAY)
    # edit = cv.GaussianBlur(edit_gr, (5, 5), 0)
    # edit_trs, _ = cv.threshold(edit, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # edit = cv.Canny(edit_gr, edit_trs * 0.5, edit_trs)
    #
    # sub = cv.subtract(edit, org)
    # sub = cv.dilate(sub, np.ones((5, 5)), iterations=1)
    # sub = cv.erode(sub, np.ones((5, 5)), iterations=2)
    # sub = cv.dilate(sub, np.ones((5, 5)), iterations=3)
    # sub2 = cv.dilate(edit, np.ones((5, 5)), iterations=1)
    # show(sub)

    # print(sub.shape)
    # print(out.shape)
    # for i in range(out.shape[0]):
    #     for j in range(out.shape[1]):
    #         if sub[i][j] == 0:
    #             out[i][j][0] = 0
    #             out[i][j][1] = 0
    #             out[i][j][2] = 0

    # _, markers = cv.connectedComponents(sub)
    # markers = markers + 1
    # unknown = sub2 - sub
    # markers[unknown == 255] = 0
    #
    # kevin_blur = cv.GaussianBlur(crop, (3, 3), 0)
    # markers = cv.watershed(kevin_blur, markers)
    # crop[markers == 1] = alpha = 1.0

    # crop = cv.resize(crop, (3 * crop.shape[1], 3 * crop.shape[0]), interpolation=cv.INTER_LINEAR)
    # show(crop)
