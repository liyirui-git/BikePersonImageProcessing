import utils
import random
ROOT_PATH = "./BikePersonDatasetProcess"


# 展示当前得到的四种图片
# 这里其实应该将函数里面的这个 imgArr作为一个参数传进来
# 或者是把文件名放在一个数组里面传进来
def img_display_after_segment(img_name):
    img_name_list = [ROOT_PATH + "/img" + "/" + img_name + ".jpg",
                     ROOT_PATH + "/mask" + "/" + img_name + ".jpg",
                     ROOT_PATH + "/seg/segPerson" + "/" + img_name + ".jpg",
                     ROOT_PATH + "/seg/segOther" + "/" + img_name + ".jpg"]

    save_img_name = utils.get_origin_name(img_name)

    utils.img_display(img_name_list, save_img_name)


# 展示不同来源的分割结果的图像
def img_display_from_diff_source(img_name, fig_show=True):
    img_name_list = [ROOT_PATH + "/img" + "/" + img_name + ".jpg",
                     ROOT_PATH + "/mask" + "/" + img_name + ".jpg",
                     ROOT_PATH + "/LIP" + "/" + img_name + ".png",
                     ROOT_PATH + "/ATR" + "/" + img_name + ".png",
                     ROOT_PATH + "/PASCAL" + "/" + img_name + ".png"]
    # 加一个文件名的反映射
    save_img_name = "diff_source_" + utils.get_origin_name(img_name)
    # 图片名列表
    tag_list = ['origin', 'isk-LIP', 'schp-LIP', 'schp-ATR', 'schp-PASCAL']
    utils.img_display(img_name_list, save_img_name, tag_list, fig_show=fig_show)


def random_display_img(num, num_range, seed=0, fig_show=True):
    random.seed(seed)
    img_list = [str(random.randint(0, num_range)) for _ in range(num)]
    for img in img_list:
        img_display_from_diff_source(img, fig_show=fig_show)
