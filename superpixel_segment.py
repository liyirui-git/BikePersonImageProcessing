import os
import cv2
import utils
import keypoint_detection


# SLIC超像素分割函数
# filename：做分割的图片的相对路径
# show：是否展示分割结果
# ite：分割的迭代次数
# region_size： 越大，分割得到的块越少
# 返回值：BGR格式的opencv图片格式
def superpixel_slic(filename, show=False, ite=10, region_size=10):
    img = cv2.imread(filename)
    lsc = cv2.ximgproc.createSuperpixelLSC(img, region_size=region_size)
    lsc.iterate(ite)
    mask_lsc = lsc.getLabelContourMask()
    label_lsc = lsc.getLabels()
    number_lsc = lsc.getNumberOfSuperpixels()
    mask_inv_lsc = cv2.bitwise_not(mask_lsc)
    img_lsc = cv2.bitwise_and(img, img, mask=mask_inv_lsc)
    if show:
        cv2.imshow("img_lsc", img_lsc)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img_lsc


def superpixel_seeds(filename, show=False, ite=10):
    img = cv2.imread(filename)
    # 初始化seeds项，注意图片长宽的顺序
    seeds = cv2.ximgproc.createSuperpixelSEEDS(img.shape[1], img.shape[0], img.shape[2], 2000, 15, 3, 5, True)
    try:
        seeds.iterate(img, ite)  # 输入图像大小必须与初始化形状相同，迭代次数为10
    except:
        print("\n error in " + filename)
    mask_seeds = seeds.getLabelContourMask()
    label_seeds = seeds.getLabels()
    number_seeds = seeds.getNumberOfSuperpixels()
    mask_inv_seeds = cv2.bitwise_not(mask_seeds)
    img_seeds = cv2.bitwise_and(img, img, mask=mask_inv_seeds)
    if show:
        cv2.imshow("img_seeds", img_seeds)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img_seeds


def superpixel_lsc(filename, show=False, ite=10):
    img = cv2.imread(filename)
    lsc = cv2.ximgproc.createSuperpixelLSC(img)
    lsc.iterate(ite)
    mask_lsc = lsc.getLabelContourMask()
    label_lsc = lsc.getLabels()
    number_lsc = lsc.getNumberOfSuperpixels()
    mask_inv_lsc = cv2.bitwise_not(mask_lsc)
    img_lsc = cv2.bitwise_and(img, img, mask=mask_inv_lsc)
    if show:
        cv2.imshow("img_lsc", img_lsc)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img_lsc


def display_superpixel(folder_path):
    image_name_list = os.listdir(folder_path)
    ct = 0
    kpd = keypoint_detection.KeyPointDetection("BikePerson-700")
    for image_name in image_name_list:
        image_path = os.path.join(folder_path, image_name)
        img1 = superpixel_slic(image_path)
        img2 = superpixel_seeds(image_path)
        thickness = int(min(img1.shape[1], img1.shape[0]) / 90 + 1)
        image_array = [cv2.cvtColor(img1, cv2.COLOR_BGR2RGB),
                       cv2.cvtColor(
                           kpd.paint_pose_on_images(image_name, cv_img=img1, thickness=thickness),
                           cv2.COLOR_BGR2RGB),
                       cv2.cvtColor(img2, cv2.COLOR_BGR2RGB),
                       cv2.cvtColor(
                           kpd.paint_pose_on_images(image_name, cv_img=img2, thickness=thickness),
                           cv2.COLOR_BGR2RGB)]
        tag_list = ["SLIC", "keypoint", "SEEDS", "keypoint"]
        utils.img_display_array(image_array, "superpixel_"+image_name,
                                tag_list=tag_list, fig_show=False, fig_size=(16,9))
        ct = ct + 1
        utils.progress_bar(ct, len(image_name_list))
