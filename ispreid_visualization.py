import cv2
import os
import matplotlib
import glob
import utils
from matplotlib import pyplot


N = 10
root = "/home/liyirui/PycharmProjects/dataset"
dataset = "Market-1501"
origin_name = "bounding_box_train"
pseudo_name = "train_pseudo_labels-ISP-7"
parsing_name = "bounding_box_train_mask"
output_name = pseudo_name + "_revisualize"
folder_path = os.path.join(root, dataset, pseudo_name)
origin_path = os.path.join(root, dataset, origin_name)
parsing_path = os.path.join(root, dataset, parsing_name)
out_folder_path = os.path.join(root, dataset, output_name)
if not os.path.exists(out_folder_path): os.mkdir(out_folder_path)


matplotlib.use('Agg')# 解决没有图形界面的问题
fig_size = (12, 7)
color_list = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
              [255, 255, 0], [255, 0, 255], [0, 255, 255]]
image_name_dir = os.listdir(folder_path)
person_id_dic = {}


def plot_image(image, height, len, x):
    # resize
    image = cv2.resize(image, (64, 128))
    # convert from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plot
    pyplot.subplot(height, len+1, x)
    pyplot.imshow(image)
    pyplot.xticks([])
    pyplot.yticks([])

for image_name in image_name_dir:
    pid = int(image_name.split("_")[0])
    if pid in person_id_dic:
        person_id_dic[pid].append(image_name)
    else:
        person_id_dic[pid] = [image_name]

partition = 0
for k in range(5000):
    # 大小是3行10列
    pyplot.figure(figsize=fig_size)
    if k not in person_id_dic:
        continue
    image_name_list = person_id_dic[k]

    ct = 0   
    for image_name in image_name_list:
        # get image path
        image_name_without_type = image_name.split(".")[0]
        image_path = glob.glob(os.path.join(folder_path, image_name_without_type)+".*")[0]
        origin_image_path = glob.glob(os.path.join(origin_path, image_name_without_type)+".*")[0]
        parsing_flag = True
        try:
            parsing_image_path = glob.glob(os.path.join(parsing_path, image_name_without_type)+".*")[0]
        except:
            parsing_flag = False
        
        # update ct
        ct = ct + 1
        if ct > N: break

        # read and plot image 
        ### origin image
        img_o = cv2.imread(origin_image_path, cv2.IMREAD_COLOR)
        plot_image(img_o, 3, N, ct+1)

        ### presudo image
        img_g = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_c = cv2.imread(image_path, cv2.IMREAD_COLOR)
        for i in range(img_g.shape[0]):
            for j in range(img_g.shape[1]):
                img_c[i][j] = color_list[img_g[i][j]]
        plot_image(img_c, 3, N, ct+N+2)

        ### text
        if ct == 1:
            pyplot.text(-150, -100, "orgin", fontdict={'size': 18})
            pyplot.text(-150, 70, "isp-seg", fontdict={'size': 18})
            pyplot.text(-150, 240, "schp-seg", fontdict={'size': 18})
        
        ### plot shcp image
        if parsing_flag:    
            img_p = cv2.imread(parsing_image_path, cv2.IMREAD_COLOR)
            plot_image(img_p, 3, N, ct+N+N+3)


    output_image_path = os.path.join(out_folder_path, str(k) + ".png")
    pyplot.savefig(output_image_path)
    pyplot.close()

    partition = partition + 1
    utils.progress_bar(partition, 5000)
