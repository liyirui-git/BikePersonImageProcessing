import os
import cv2
import utils
import keypoint_detection


class SuperPixel:
    # 这个色块的编号
    number = -1
    # 平均颜色，即对color_array求平均值
    average_color = []
    # 需要存放这个色块的颜色，RGB三通道
    # [[255,255,255], [0,0,0], ..., [1,1,1]]
    color_array = []
    # 存放这个区域在图片中的像素，只有两种值0和1，包含在该区域的像素位置，设置为 1，否则为 0。
    region_matrix = []


class SuperPixelSegmentation:
    # input_path：做分割的图片的相对路径
    # skeleton: [[]] 骨架信息，为一个与输入的图片大小相同的矩阵
    # method: 使用的分割方法
    # ite：分割的迭代次数
    # region_ration： 越大，分割得到的块越多
    def __init__(self, filename, input_folder, output_folder, skeleton, method='SLIC', ite=10, region_ratio=40):
        self.filename = filename
        self.filepath = os.path.join(input_folder, filename)
        self.input_folder = input_folder
        self.output_folder = utils.makedir_from_name_list([output_folder])[0]
        self.image = cv2.imread(self.filepath)
        self.output_image = cv2.imread(self.filepath)
        # cv2.imshow("before", self.output_image)
        # cv2.waitKey()
        self.height, self.width = self.image.shape[0], self.image.shape[1]
        region_size = int(self.height / region_ratio)
        # 做超像素分割
        if method == 'SLIC':
            self.spresult = cv2.ximgproc.createSuperpixelSLIC(self.image, region_size=region_size)
            self.spresult.iterate(ite)
        elif method == 'SLC':
            self.spresult = cv2.ximgproc.createSuperpixelSLC(self.image, region_size=region_size)
            self.spresult.iterate(ite)
        # 骨架信息
        self.skeleton = skeleton

    # 展示超像素分割的结果
    def display(self):
        mask = self.spresult.getLabelContourMask()
        mask_inv = cv2.bitwise_not(mask)
        img = cv2.bitwise_and(self.image, self.image, mask=mask_inv)
        cv2.imshow("img_lsc", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return img

    '''
    def superpixel_seeds(self, show=False, ite=10):
        img = cv2.imread(self.filename)
        # 初始化seeds项，注意图片长宽的顺序
        seeds = cv2.ximgproc.createSuperpixelSEEDS(img.shape[1], img.shape[0], img.shape[2], 2000, 15, 3, 5, True)
        try:
            seeds.iterate(img, ite)  # 输入图像大小必须与初始化形状相同，迭代次数为10
        except:
            print("\n error in " + self.filename)
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
    
    def display_segmentation(self, folder_path):
        image_name_list = os.listdir(folder_path)
        ct = 0
        kpd = keypoint_detection.KeyPointDetection("BikePerson-700")
        for image_name in image_name_list:
            image_path = os.path.join(folder_path, image_name)
            img1 = self.superpixel_slic(image_path)
            img2 = self.superpixel_seeds(image_path)
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
            utils.img_display_array(image_array, "superpixel_" + image_name,
                                    tag_list=tag_list, fig_show=False, fig_size=(16, 9))
            ct = ct + 1
            utils.progress_bar(ct, len(image_name_list))
    '''

    # 将超像素分割得到的色块合并
    def part_combination(self, threshold=24, ite=10, decreasing_ratio=0.6, display=False):
        # 1. 得到超像素分割的结果，得到存放着每一块的SuperPixel的数组，还有色块与色块之间的邻接表
        # getLabels 得到的是一个与图片相同大小的矩阵，矩阵中的每个点的值就表示图像中该点的像素所属的色块
        label_lsc = self.spresult.getLabels()
        # getNumberOfSuperPixels 得到的是图像中的像素聚成了几类
        number_lsc = self.spresult.getNumberOfSuperpixels()
        # 得到各个色块的哈希表
        superpixel_map = {}     # key是编号，value是SuperPixel对象
        for i in range(self.height):
            for j in range(self.width):
                # 对每一个像素，操作一下
                part_number = label_lsc[i][j]
                if part_number in superpixel_map:
                    superpixel_map[part_number].color_array.append(self.image[i][j])
                    superpixel_map[part_number].region_matrix[i][j] = 1
                else:
                    super_p = SuperPixel()
                    super_p.number = part_number
                    super_p.color_array = [self.image[i][j]]
                    super_p.region_matrix = []
                    for x in range(self.height):
                        line = []
                        for y in range(self.width):
                            line.append(0)
                        super_p.region_matrix.append(line)
                    super_p.region_matrix[i][j] = 1
                    superpixel_map[part_number] = super_p
        # 得到色块与色块之间的邻接表
        adjacency_list = []
        for i in range(number_lsc):
            line = []
            for j in range(number_lsc):
                line.append(0)
            adjacency_list.append(line)
        # 在八邻域内计算邻接
        for i in range(self.height):
            for j in range(self.width):
                color1 = label_lsc[i][j]
                if i > 0:
                    color2 = label_lsc[i-1][j]
                    adjacency_list[color1][color2] = 1
                    adjacency_list[color2][color1] = 1
                if i < self.height-1:
                    color2 = label_lsc[i+1][j]
                    adjacency_list[color1][color2] = 1
                    adjacency_list[color2][color1] = 1
                if j > 0:
                    color2 = label_lsc[i][j-1]
                    adjacency_list[color1][color2] = 1
                    adjacency_list[color2][color1] = 1
                if j < self.width - 1:
                    color2  = label_lsc[i][j+1]
                    adjacency_list[color1][color2] = 1
                    adjacency_list[color2][color1] = 1
                if i > 0 and j > 0:
                    color2 = label_lsc[i-1][j-1]
                    adjacency_list[color1][color2] = 1
                    adjacency_list[color2][color1] = 1
                if i > 0 and j < self.width - 1:
                    color2 = label_lsc[i-1][j+1]
                    adjacency_list[color1][color2] = 1
                    adjacency_list[color2][color1] = 1
                if i < self.height - 1 and j > 0:
                    color2 = label_lsc[i+1][j-1]
                    adjacency_list[color1][color2] = 1
                    adjacency_list[color2][color1] = 1
                if i < self.height - 1 and j < self.width - 1:
                    color2 = label_lsc[i+1][j+1]
                    adjacency_list[color1][color2] = 1
                    adjacency_list[color2][color1] = 1
        # 求色块的均值
        for i in range(number_lsc):
            if i in superpixel_map:
                c1, c2, c3 = 0, 0, 0
                size = len(superpixel_map[i].color_array)
                for color in superpixel_map[i].color_array:
                    c1 = c1 + color[0]
                    c2 = c2 + color[1]
                    c3 = c3 + color[2]
                c1, c2, c3 = c1 / size, c2 / size, c3 / size
                superpixel_map[i].average_color = [c1, c2, c3]
        # 2. 关键点检测的骨架信息放在 self.skeleton
        # 3. 将pose经过的地方作为初始点
        # 建立一个种子集合，将skeleton路过的种子点都加入进来
        seeds_set = set()
        for i in range(self.height):
            for j in range(self.width):
                if self.skeleton[i][j] == 1:
                    seeds_set.add(label_lsc[i][j])
        if display:
            self.display_region(seeds_set, label_lsc)
        # 4. 逐跳操作
        # 对于当前集合中的色块，根据邻接表，判断相邻色块的平均颜色与当前色块的距离是否满足阈值
        # 循环n轮，或者直到没有新的色块可以加进来为止
        # 每一次循环之后，可以将阈值减小到一定的比例
        d_threshold = threshold
        counter = 0
        add_seeds = 1
        while counter < ite and add_seeds != 0:
            add_seeds = 0
            # 对于seeds_set中的seed，判断它与它的邻接色块是否满足要求
            for i in list(seeds_set):
                for j in range(number_lsc):
                    if adjacency_list[i][j] == 1 and j not in seeds_set:
                        color1 = superpixel_map[i].average_color
                        color2 = superpixel_map[j].average_color
                        delta = abs(color1[0]-color2[0]) \
                                + abs(color1[1]-color2[1]) \
                                + abs(color1[2]-color2[2])
                        if delta < d_threshold:
                            seeds_set.add(j)
                            add_seeds = add_seeds + 1
            if display:
                self.display_region(seeds_set, label_lsc)
            counter = counter + 1
            d_threshold = d_threshold * decreasing_ratio
        self.save_img(os.path.join(self.output_folder, self.filename), seeds=seeds_set, labels=label_lsc)

    # 展示结果
    def display_region(self, seeds_set, label_lsc):
        for i in range(self.height):
            for j in range(self.width):
                if label_lsc[i][j] in seeds_set:
                    self.image[i][j] = [255, 0, 255]
        self.display()

    # 导出结果
    def save_img(self, output_file, seeds, labels):
        for i in range(self.height):
            for j in range(self.width):
                if labels[i][j] not in seeds:
                    self.output_image[i][j] = [0, 0, 0]
        cv2.imwrite(output_file, self.output_image)

    def run(self):
        self.part_combination()


if __name__ == '__main__':
    kdp = keypoint_detection.KeyPointDetection("BikePerson-700")

    folder_list = ["query", "bounding_box_test", "bounding_box_train"]
    input_path = "BikePerson-700/BikePerson-700-origin"
    output_path = "BikePerson-700/BikePerson-700-superpixel-seg"
    utils.makedir_from_name_list([output_path])

    for folder in folder_list:
        input_folder_path = os.path.join(input_path, folder)
        output_folder_path = os.path.join(output_path, folder)
        filename_list = os.listdir(input_folder_path)
        print(" begin at " + input_folder_path)
        total = len(filename_list)
        counter = 0
        for filename in filename_list:
            sps = SuperPixelSegmentation(skeleton=kdp.get_skeleton_matrix(filename),
                                         filename=filename,
                                         input_folder=input_folder_path,
                                         output_folder=output_folder_path)
            sps.run()
            counter = counter+1
            utils.progress_bar(counter, total)