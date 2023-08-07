from lib import *
from make_datapath import make_datapath_list
from transform import DataTransform
from extract_inform_annotation import Anno_xml


class VOC2012Dataset(data.Dataset):
    def __init__(self, img_list, anno_list, transform, anno_xml):
        self.img_list = img_list
        self.anno_list = anno_list
        self.transform = transform
        self.anno_xml = anno_xml

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img, gt, height, width = self.pull_item(index)

        return img, gt

    def pull_item(self, index):
        img_file_path = self.img_list[index]
        img = cv2.imread(img_file_path)  # BGR
        height, width, channels = img.shape

        # get anno information
        anno_file_path = self.anno_list[index]
        ann_info = self.anno_xml(anno_file_path, width, height)

        # preprocessing
        img, boxes, labels = self.transform(img, ann_info[:, :4], ann_info[:, 4])

        # BGR -> RGB, (height, width, channels) -> (channels, height, width)
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)

        # ground truth
        gt = gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width


def my_collate_fn(batch):
    targets = []
    imgs = []

    for sample in batch:
        imgs.append(sample[0])  # sample[0]=img
        targets.append(torch.FloatTensor(sample[1]))  # sample[1]=annotation
    # [3, 300, 300]
    # (batch_size, 3, 300, 300)
    imgs = torch.stack(imgs, dim=0)

    return imgs, targets



def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


def base_transform(image, size, mean):
    x = cv2.resize(image, (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    return x


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform(image, self.size, self.mean), boxes, labels



if __name__ == "__main__":
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

    # prepare train, valid, annotation list
    root_path = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_annotation_list, val_img_list, val_annotation_list, trainval_img_list, trainval_annotation_list = make_datapath_list(root_path)

    # prepare data transform
    color_mean = (104, 117, 123)
    input_size = 300

    train_dataset = VOC2012Dataset(train_img_list, train_annotation_list,
                                   transform=DataTransform(input_size, color_mean), anno_xml=Anno_xml(classes))

    val_dataset = VOC2012Dataset(val_img_list, val_annotation_list,
                                 transform=DataTransform(input_size, color_mean), anno_xml=Anno_xml(classes))

    # print(len(train_dataset))
    # print(train_dataset.__getitem__(1))

    batch_size = 4
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)

    dataloader_dict = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    batch_iter = iter(dataloader_dict["val"])
    images, targets = next(batch_iter)  # get 1 sample
    print(images.size())
    print(len(targets))
    print(targets[0].size())  # xmin, ymin, xmax, ymax, label
