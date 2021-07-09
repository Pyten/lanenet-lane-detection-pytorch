import torch
from torch.autograd import Variable
import numpy as np

def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
 
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

class Eval_Score():
    # IoU and F1(Dice)

    def __init__(self, y_pred, y_true, threshold = 0.5):
        input_flatten = np.int32(y_pred.flatten() > threshold)
        target_flatten = np.int32(y_true.flatten() > threshold)
        self.intersection = np.sum(input_flatten * target_flatten)
        self.sum = np.sum(target_flatten) + np.sum(input_flatten)
        self.union = self.sum - self.intersection
    
    def Dice(self, eps=1):
        return np.clip(((2. * self.intersection) / (self.sum + eps)), 1e-5, 0.99999)
    
    def IoU(self):
        if self.union == 0: return 0.0
        return self.intersection / self.union

class Multi_Eval_Score():

    def __init__(self, y_pred, y_true, classNumber, threshold=0.5):
        self.y_pred = y_pred
        self.y_true = y_true
        self.classNumber = classNumber
        self.threshold = threshold

    def get_cur_label(self, classIdx, neededMap):
        map = np.zeros((self.y_pred.shape[0], self.y_pred.shape[1]))
        # y_true = self.y_true.cpu().numpy()
        map_idx = neededMap == classIdx
        map[map_idx] = classIdx
        return map

    def get_cur_label_map(self, classIdx):
        cur_label = self.get_cur_label(classIdx, self.y_true)
        cur_pred = self.get_cur_label(classIdx, self.y_pred)

        return cur_pred, cur_label

    def IoU(self):
        eval_list = [0.0 for i in range(self.classNumber - 1)]
        count_label = [0 for i in range(self.classNumber - 1)]
        # import pdb

        for i in range(1, self.classNumber):
            # pdb.set_trace()
            cur_pred, cur_label = self.get_cur_label_map(i)
            numbers = np.unique(cur_label)
            for j in numbers[1:]:
                count_label[int(j - 1)] += 1
            iou = Eval_Score(cur_pred, cur_label).IoU()
            eval_list[i - 1] = iou

        return eval_list, count_label
