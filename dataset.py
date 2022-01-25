import torch
import torch.utils.data as data
import os
from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
import json

from utils import transforms

class TransDataset(data.Dataset):
    def __init__(self, args, dataset_name, train_type='train'):
        super(TransDataset, self).__init__()
        self.dataset = dataset_name.split('_')[0].upper()
        self.datalevel = dataset_name.split('_')[-1].lower()
        self.dataset_name = dataset_name
        self.train_type = train_type
        self.images_root = './Dataset/{}_image/'.format(self.dataset)

        name2cls = {'pipa_fine': 16, 'pipa_coarse': 5, 'pisc_fine': 6, 'pisc_coarse': 3}
        self.num_classes = name2cls[dataset_name]
        name2max_person = {'PIPA': 5, 'PISC': 8}
        self.max_person = name2max_person[self.dataset]

        file_prefix = './relation_split/{}/{}_{}_relation_{}'.format(self.dataset, self.dataset, self.datalevel, train_type)
        self.images_list_filename = file_prefix + '_images.txt'
        self.bboxes_list_filename = file_prefix + '_bbox.json'
        self.relations_list_filename = file_prefix + '_relation.json'
        self.image_names = []
        with open(self.images_list_filename, 'r') as fin:
            for line in fin:
                self.image_names.append(line.split()[0])
        with open(self.bboxes_list_filename, 'r') as fin:
            self.image_bboxes = json.load(fin)  # list of bboxes for each image [[bbox1_img1, bbox2_img1], [bbox1_img2]]
        with open(self.relations_list_filename, 'r') as fin:
            self.relations = json.load(fin)

        self.labels = []  # use to calculate class weight
        for image_relations in self.relations:
            img_labels = [image_relation[2] for image_relation in image_relations]
            self.labels.extend(img_labels)

        if train_type == 'train':
            self.input_transform = transforms.Compose([
                # transforms.Resize((cache_size, cache_size)),
                # transforms.RandomCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.input_transform = transforms.Compose([
                # transforms.Resize((cache_size, cache_size)),
                # transforms.CenterCrop(args.image_size),
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.images_root, self.image_names[index])).convert('RGB')  # convert gray to rgb

        # # get object information
        # obj_dict = self.object_dict[self.names[index].split('.')[0]]
        # obj_array = np.zeros(80)
        # for obj_cls, obj_score in zip(obj_dict['categories'], obj_dict['scores']):
        #     if obj_score >= 0.7:
        #         obj_array[obj_cls] += 1
        # obj_tensor = torch.from_numpy(obj_array).float()

        bbox_num = len(self.image_bboxes[index])
        image_bboxes = np.zeros((self.max_person, 4), dtype=np.float32)
        image_bboxes[0: bbox_num, :] = np.array(self.image_bboxes[index])
        image_bboxes = torch.from_numpy(image_bboxes)
        bbox_mask = np.zeros((self.max_person, 1), dtype=np.int32)
        bbox_mask[:bbox_num] = 1
        bbox_mask = torch.from_numpy(bbox_mask)

        if self.input_transform:
            img, image_bboxes = self.input_transform(img, image_bboxes)

        relation_mask = np.zeros((self.max_person, self.max_person), dtype=np.int32)
        relation_id = np.zeros((self.max_person, self.max_person), dtype=np.int32)
        image_relations = self.relations[index]
        for i in range(len(image_relations)):
            image_relation = image_relations[i]
            relation_mask[image_relation[0]][image_relation[1]] = 1

            relation_id[image_relation[0]][image_relation[1]] = image_relation[2]
            relation_id[image_relation[1]][image_relation[0]] = image_relation[2]

        relation_mask = torch.from_numpy(relation_mask).long()
        relation_id = torch.from_numpy(relation_id).long()

        return {'img': img, 'image_bboxes': image_bboxes, 'relations_id': relation_id,
                'relation_mask': relation_mask, 'img_index': index, 'bbox_mask': bbox_mask}


if __name__ == '__main__':
    dataset = TransDataset('pisc_fine', 'train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False,
                                             worker_init_fn=np.random.seed(13))
    # training
    for epoch in range(10):
        for batch in dataloader:
            # your training code here
            print(batch)
            pass