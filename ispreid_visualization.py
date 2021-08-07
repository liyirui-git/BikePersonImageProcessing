import cv2
import os
import matplotlib
import glob
import utils
from matplotlib import pyplot

# 解决没有图形界面的问题
matplotlib.use('Agg') 

root = "/home/liyirui/PycharmProjects/dataset"
dataset = "Market-1501"
origin_name = "bounding_box_train"
pseudo_name = "train_pseudo_labels-ISP-7"
parsing_name = "bounding_box_train_mask"
output_name = pseudo_name + "_revisualize"

fig_size = (12, 7)
N = 10

folder_path = os.path.join(root, dataset, pseudo_name)
origin_path = os.path.join(root, dataset, origin_name)
parsing_path = os.path.join(root, dataset, parsing_name)
out_folder_path = os.path.join(root, dataset, output_name)

color_list = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
              [255, 255, 0], [255, 0, 255], [0, 255, 255]]

if not os.path.exists(out_folder_path):
    os.mkdir(out_folder_path)

image_name_dir = os.listdir(folder_path)
person_id_dic = {}

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
        image_name_without_type = image_name.split(".")[0]
        # get img
        image_path = glob.glob(os.path.join(folder_path, image_name_without_type)+".*")[0]
        origin_image_path = glob.glob(os.path.join(origin_path, image_name_without_type)+".*")[0]
        parsing_flag = True
        try:
            parsing_image_path = glob.glob(os.path.join(parsing_path, image_name_without_type)+".*")[0]
        except:
            parsing_flag = False
        
        img_g = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_c = cv2.imread(image_path, cv2.IMREAD_COLOR)
        for i in range(img_g.shape[0]):
            for j in range(img_g.shape[1]):
                img_c[i][j] = color_list[img_g[i][j]]
        img_o = cv2.imread(origin_image_path, cv2.IMREAD_COLOR)
        if parsing_flag:    
            img_p = cv2.imread(parsing_image_path, cv2.IMREAD_COLOR)

        # resize 
        img_c = cv2.resize(img_c, (64, 128))
        img_o = cv2.resize(img_o, (64, 128))
        if parsing_flag:
            img_p = cv2.resize(img_p, (64, 128))

        # convert from BGR to RGB
        img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
        img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
        if parsing_flag:
            img_p = cv2.cvtColor(img_p, cv2.COLOR_BGR2RGB)
        
        # pyplot
        ct = ct + 1
        if ct > N: 
            break

        pyplot.subplot(3, N+1, ct+1)
        pyplot.imshow(img_o)
        pyplot.xticks([])
        pyplot.yticks([])
        pyplot.subplot(3, N+1, ct+N+2)
        pyplot.imshow(img_c)
        pyplot.xticks([])
        pyplot.yticks([])
        if parsing_flag:
            pyplot.subplot(3, N+1, ct+N+N+3)
            pyplot.imshow(img_p)
            pyplot.xticks([])
            pyplot.yticks([])

        # plot text
        if ct == 1:
            pyplot.text(-150, -270, "orgin", fontdict={'size': 18})
            pyplot.text(-150, -100, "isp-seg", fontdict={'size': 18})
            pyplot.text(-150, 70, "schp-seg", fontdict={'size': 18})

        if ct == 10: break
       
        

    output_image_path = os.path.join(out_folder_path, str(k) + ".png")

    pyplot.savefig(output_image_path)
    pyplot.close()

    partition = partition + 1
    utils.progress_bar(partition, 5000)
