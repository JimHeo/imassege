from torch import nn
import torch
from torch.nn.functional import one_hot

class Confusion_Matrix(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
    
    def __str__(self):
        return "confusion_matrix"
    
    def forward(self, y_pred, y_true, threshold=0.5):
        if self.num_classes == 1:
            y_pred = torch.sigmoid(y_pred)
            y_pred = (y_pred > threshold).to(torch.float32)
        else:
            y_pred = torch.softmax(y_pred, dim=1)
            y_true = one_hot(y_true, self.num_classes)
            y_true = y_true.permute(0, 3, 1, 2)
            
        axis = (0, 2, 3)
        # calculate the true positive, false positive, false positive and true negative
        tp = torch.sum(y_pred * y_true, dim=axis).to(torch.float32)
        fp = torch.sum(y_pred * (1 - y_true), dim=axis).to(torch.float32)
        fn = torch.sum((1 - y_pred) * y_true, dim=axis).to(torch.float32)
        tn = torch.sum((1 - y_pred) * (1 - y_true), dim=axis).to(torch.float32)
        
        return tp, fp, fn, tn

class Accuracy(nn.Module):
    def __init__(self, num_classes=1, epsilon=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        
    def __str__(self):
        return "accuracy"
    
    def forward(self, y_pred, y_true, threshold=0.5):
        tp, fp, fn, tn = Confusion_Matrix(self.num_classes)(y_pred, y_true, threshold)
        
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
        tp, fp, fn, tn = Confusion_Matrix(self.num_classes)(y_pred, y_true, threshold)
        
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
        tp, fp, fn, tn = Confusion_Matrix(self.num_classes)(y_pred, y_true, threshold)
        
        # calculate the recall
        recall = tp / (tp + fn + self.epsilon)
        recall = torch.sum(recall) / self.num_classes
        # return the recall
        return recall
    
class F1_Score(nn.Module):
    def __init__(self, num_classes=1, epsilon=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
    
    def __str__(self):
        return "f1_score"
        
    def forward(self, y_pred, y_true, threshold=0.5):
        tp, fp, fn, tn = Confusion_Matrix(self.num_classes)(y_pred, y_true, threshold)
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
        tp, fp, fn, tn = Confusion_Matrix(self.num_classes)(y_pred, y_true, threshold)
        
        # calculate the iou (as jaccard index)
        iou = tp / (tp + fp + fn + self.epsilon)
        iou = torch.sum(iou) / self.num_classes
        # return the iou
        return iou