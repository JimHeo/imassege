from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

class ConfusionMatrix(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
    
    def __str__(self):
        return "confusion_matrix"
    
    def forward(self, y_pred, y_true, threshold=0.5):
        if self.num_classes == 1:
            _y_pred = torch.sigmoid(y_pred)
            _y_pred = (y_pred > threshold).to(torch.float32)
        else:
            _y_pred = torch.softmax(y_pred, dim=1)
            _y_true = F.one_hot(y_true, self.num_classes)
            _y_true = _y_true.permute(0, 3, 1, 2)
            
        axis = (0, 2, 3)
        # calculate the true positive, false positive, false positive and true negative
        tp = torch.sum(_y_pred * _y_true, dim=axis).to(torch.float32)
        fp = torch.sum(_y_pred * (1 - _y_true), dim=axis).to(torch.float32)
        fn = torch.sum((1 - _y_pred) * _y_true, dim=axis).to(torch.float32)
        tn = torch.sum((1 - _y_pred) * (1 - _y_true), dim=axis).to(torch.float32)
        
        return tp, fp, fn, tn

class Accuracy(nn.Module):
    def __init__(self, num_classes=1, epsilon=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        
    def __str__(self):
        return "accuracy"
    
    def forward(self, y_pred, y_true, threshold=0.5):
        tp, fp, fn, tn = ConfusionMatrix(self.num_classes)(y_pred, y_true, threshold)
        
        # calculate the accuracy
        accuracy = (tp + tn) / (tp + fp + fn + tn + self.epsilon)
        accuracy = torch.sum(accuracy) / self.num_classes
        # return the accuracy
        return accuracy
        

class Precision(nn.Module):
    def __init__(self, num_classes=1, epsilon=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
    
    def __str__(self):
        return "precision"
        
    def forward(self, y_pred, y_true, threshold=0.5):
        tp, fp, fn, tn = ConfusionMatrix(self.num_classes)(y_pred, y_true, threshold)
        
        # calculate the precision
        precision = tp / (tp + fp + self.epsilon)
        precision = torch.sum(precision) / self.num_classes
        # return the precision
        return precision
    
class Recall(nn.Module):
    def __init__(self, num_classes=1, epsilon=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
    
    def __str__(self):
        return "recall"
        
    def forward(self, y_pred, y_true, threshold=0.5):
        tp, fp, fn, tn = ConfusionMatrix(self.num_classes)(y_pred, y_true, threshold)
        
        # calculate the recall
        recall = tp / (tp + fn + self.epsilon)
        recall = torch.sum(recall) / self.num_classes
        # return the recall
        return recall
    
class F1Score(nn.Module):
    def __init__(self, num_classes=1, epsilon=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
    
    def __str__(self):
        return "f1_score"
        
    def forward(self, y_pred, y_true, threshold=0.5):
        tp, fp, fn, tn = ConfusionMatrix(self.num_classes)(y_pred, y_true, threshold)
        # calculate the f1 score
        f1_score = 2 * tp / (2 * tp + fp + fn + self.epsilon)
        f1_score = torch.sum(f1_score) / self.num_classes
        # return the f1 score
        return f1_score
    
class IoU(nn.Module):
    def __init__(self, num_classes=1, epsilon=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        
    def __str__(self):
        return "iou"
    
    def forward(self, y_pred, y_true, threshold=0.5):
        tp, fp, fn, tn = ConfusionMatrix(self.num_classes)(y_pred, y_true, threshold)
        
        # calculate the iou (as jaccard index)
        iou = tp / (tp + fp + fn + self.epsilon)
        iou = torch.sum(iou) / self.num_classes
        # return the iou
        return iou
    
class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2., epsilon=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        if isinstance(self.alpha, float):
            alpha = self.alpha
        
        y_pred_sig = torch.sigmoid(y_pred)
        y_pred_sig = torch.clamp(y_pred_sig, self.epsilon, 1. - self.epsilon)
        y_true_expand = y_true.unsqueeze(dim=1)
        
        # compute the actual focal loss
        modular = torch.pow(1. - y_pred_sig, self.gamma)
        focal = -modular * (alpha * y_true_expand * torch.log(y_pred_sig) + (1 - alpha) * (1. - y_true_expand) * torch.log(1. - y_pred_sig))
        
        # loss : (B, H, W) -> Scalar
        loss = torch.sum(focal, dim=1)
        loss = torch.mean(loss)
        
        return loss

class CategoricalFocalLoss(nn.Module):
    def __init__(self, num_classes=3, alpha=0.25, gamma=2., epsilon = 1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        if isinstance(self.alpha, float):
            alpha = self.alpha
        elif isinstance(self.alpha, list):
            alpha = torch.from_numpy(np.array(self.alpha)).to(y_pred.device)
            # alpha : (B, C, H, W)
            alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(y_pred)
        elif isinstance(self.alpha, np.ndarray):
            alpha = torch.from_numpy(self.alpha).to(y_pred.device)
            # alpha : (B, C, H, W)
            alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(y_pred)
        elif isinstance(self.alpha, torch.Tensor):
            # alpha : (B, C, H, W)
            alpha = self.alpha.view(-1, len(self.alpha), 1, 1).expand_as(y_pred).to(y_pred.device)
        
        # compute softmax over the classes axis
        # y_pred_soft : (B, C, H, W)
        y_pred_soft = F.softmax(y_pred, dim=1)
        y_pred_soft = torch.clamp(y_pred_soft, self.epsilon, 1. - self.epsilon)
        
        # create the labels one hot tensor
        # y_true_one_hot : (B, C, H, W)
        y_true_one_hot = F.one_hot(y_true, self.num_classes)
        y_true_one_hot = y_true_one_hot.permute(0, 3, 1, 2)

        # compute the actual focal loss
        modular = torch.pow(1. - y_pred_soft, self.gamma)
        
        # alpha, weight, y_pred_soft : (B, C, H, W)
        # focal : (B, C, H, W)
        focal = -modular * alpha * y_true_one_hot * torch.log(y_pred_soft)
        
        # loss : (B, H, W) -> Scalar
        loss = torch.sum(focal, dim=1)
        loss = torch.mean(loss)
        
        return loss
