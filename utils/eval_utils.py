import numpy as np
import os
import json
import torch

def cal_AP(scores_list,labels_list):
    list_len = len(scores_list)
    assert(list_len == len(labels_list)), 'score and label lengths are not same'
    dtype = [('score',float), ('label',int)]
    values = []
    for i in range(list_len):
        values.append((scores_list[i],labels_list[i]))
    np_values = np.array(values, dtype=dtype)
    np_values = np.sort(np_values, order='score')
    np_values = np_values[::-1]
    class_num = sum(labels_list)
    max_pre = np.zeros(class_num)
    pos = 0
    for i in range(list_len):
        if (np_values[i][1] == 1):
            max_pre[pos] = (pos + 1) * 1.0 / (i + 1)
            pos = pos + 1
    for i in range(class_num-2, -1, -1):
        if (max_pre[i] < max_pre[i + 1]):
            max_pre[i] = max_pre[i + 1]
    return sum(max_pre) / (len(max_pre) + 1e-6)

def normnp(scores_np):
    shape_x = scores_np.shape
    for i in range(shape_x[0]):
        scores_np[i,:] = scores_np[i,:] / sum(scores_np[i,:])
    return scores_np

def compute_map(confs, labels):
    # confs: confidence of each class, shape is [num_samples, num_classed]
    # labels: label for each sample, shape is [num_samples, 1]
    csn = normnp(confs)
    num_class = confs.shape[-1]
    per_class_ap = []
    for i in range(num_class):
        class_scores = list(csn[:, i])
        class_labels = [l == i for l in labels]

        per_class_ap.append(cal_AP(class_scores, class_labels))
    print(per_class_ap)
    return np.mean(per_class_ap)

def save_model(tosave_model, result, args, logger):
    best_result_path = os.path.join(args.output_dir, 'best_result.json')
    if os.path.exists(best_result_path):
        with open(best_result_path, 'r') as fin:
            best_result = json.load(fin)
    else:
        best_result = {}

    if args.dataset not in best_result:
        best_result[args.dataset] = result
    elif best_result[args.dataset] > result:
        logger.info("Not better than best model, no saving...")
        return None
    else:
        best_result[args.dataset] = result

    model_path = 'model_' + args.dataset + '.pth'
    save_path = os.path.join(args.output_dir, model_path)
    torch.save(tosave_model.module.state_dict(), save_path)

    with open(best_result_path, 'w') as fin:
        json.dump(best_result, fin)

    logger.info("Saving model to {}, best result is {}".format(save_path, best_result[args.dataset]))

def load_model(args, unload_model, logger):
    model_path = 'model_{}.pth'.format(args.dataset)
    unload_model.load_state_dict(torch.load(os.path.join(args.model_path, model_path)))
    logger.info("Loading from model {}".format(os.path.join(args.model_path, model_path)))
    return unload_model

class edge_loss(torch.nn.Module):
    def __init__(self, num_class):
        super(edge_loss, self).__init__()
        self.num_class = num_class
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, scores, labels, masks):  # labels masks shape [batch_size, max_person, max_person]
        masks = masks.view(-1, 1).bool()
        masks = masks.detach()  # [batch_size*max_person*max_person, 1]
        labels = labels.view((-1, 1))[masks]  # [batch_size*max_person*max_person, 1]

        # scores shape [batch_size, max_person, max_person, num_classes]
        scores = scores.view(-1, self.num_class)  # [batch_size*max_person*max_person, num_classes]
        scores = scores[masks.repeat(1, self.num_class)].view(-1, self.num_class)
        losses = self.criterion(scores, labels)

        return losses, scores, labels

if __name__ == '__main__':
    conf = np.array([0.9, 0.1, 0.8, 0.4])
    pred_cls = np.array([0, 1, 2, 0])
    target_cls = np.array([0, 0, 2, 1])
    print(compute_map(conf, pred_cls, target_cls))
