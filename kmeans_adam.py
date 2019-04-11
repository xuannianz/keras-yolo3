import numpy as np


class YOLOKmeans:
    def __init__(self, cluster_number, annotation_paths, anchors_path):
        self.cluster_number = cluster_number
        self.annotation_paths = annotation_paths
        self.anchors_path = anchors_path

    def iou(self, boxes, clusters):
        """
        假设 boxes 中的 box 和 clusters 中的 box 中心点相同, 计算两者的交并比
        Args:
            boxes: 所有 gt_boxes
            clusters: 从 gt_boxes 中选出 num_clusters 个 box

        Returns:

        """
        n = boxes.shape[0]
        k = self.cluster_number
        # (n, )
        box_area = boxes[:, 0] * boxes[:, 1]
        # 每个元素重复 k 次, (n * k, )
        # 如 box_area 为 [1,2,3], repeat 之后变为 [1,1,..,1,2,2,...,2,3,3,...,3]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        # 如 cluster_area 为 [1,2,3], repeat 之后变为 [1,2,...,3,1,2,...,3,1,2,...,3]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        # init k clusters
        clusters = boxes[np.random.choice(box_number, k, replace=False)]
        while True:
            distances = 1 - self.iou(boxes, clusters)
            # 每个 gt_box 和哪个 cluster 距离最小
            current_nearest = np.argmin(distances, axis=1)
            # clusters won't change
            if (last_nearest == current_nearest).all():
                break
            for cluster in range(k):
                # update clusters
                # np.median 就是排好序在取中位数, 如果为偶数个数, 取中间两个数的平均值
                # dist 此处就是求每个 cluster 的 boxes 的 width,height 的中位数
                clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)
            print('clusters={}'.format(clusters))
            last_nearest = current_nearest

        return clusters

    def result2txt(self, data):
        f = open(self.anchors_path, 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):
        """
        得到所有 gt_boxes 的 width 和 height
        :return:
        """
        dataset = []
        for annotation_path in self.annotation_paths:
            f = open(annotation_path, 'r')
            for line in f:
                infos = line.split(" ")
                length = len(infos)
                for i in range(1, length):
                    width = int(infos[i].split(",")[2]) - int(infos[i].split(",")[0])
                    height = int(infos[i].split(",")[3]) - int(infos[i].split(",")[1])
                    dataset.append([width, height])
            f.close()
        result = np.array(dataset)
        return result

    def txt2clusters(self):
        all_boxes = self.txt2boxes()
        # (num_clusters, 2) wh
        result = self.kmeans(all_boxes, k=self.cluster_number)
        # lexsort 接受一个 k 个元素的 tuple, 或者 (k, N) 的二维数组
        # 好比一个 excel 有 k 列, 先按最后一列就行排序, 再对前一列的进行排序
        # result.T[0, None] 的 shape 是 (1, num_clusters), 也就是所有 anchor_boxes 的宽组成的 np.array
        # 最终得到的 result 就是按宽进行排序的 anchor_boxes
        # 参考 https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.lexsort.html
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    num_clusters = 9
    # annotation_paths = ["/home/adam/.keras/datasets/udacity_self_driving_car/object-detection-crowdai/train_yolo.csv",
    #                     "/home/adam/.keras/datasets/udacity_self_driving_car/object-detection-crowdai/val_yolo.csv"]
    annotation_paths = ["/home/adam/.keras/datasets/VOCdevkit/trainval/train.txt",
                        "/home/adam/.keras/datasets/VOCdevkit/test/test.txt"]
    kmeans = YOLOKmeans(num_clusters, annotation_paths, 'yolo_anchors_voc.txt')
    kmeans.txt2clusters()
