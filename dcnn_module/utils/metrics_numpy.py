from torch import nn
import numpy as np

def to_categorical(y, num_classes=None, dtype=np.float32):
    input_shape = y.shape
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
  
    return categorical

class ConfusionMatrix(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
    
    def __str__(self):
        return "confusion_matrix"
    
    def forward(self, y_pred, y_true):
        if self.num_classes > 1: y_true = to_categorical(y_true, self.num_classes)

        axis = (0, 1)
        # calculate the true positive, false positive, false positive and true negative
        tp = np.sum(y_pred * y_true, axis=axis, dtype=np.float32)
        fp = np.sum(y_pred * (1 - y_true), axis=axis, dtype=np.float32)
        fn = np.sum((1 - y_pred) * y_true, axis=axis, dtype=np.float32)
        tn = np.sum((1 - y_pred) * (1 - y_true), axis=axis, dtype=np.float32)

        return tp, fp, fn, tn
    
class Accuracy(nn.Module):
    def __init__(self, num_classes=1, epsilon=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
    
    def __str__(self):
        return "accuracy"
    
    def forward(self, y_pred, y_true):
        tp, fp, fn, tn = ConfusionMatrix(self.num_classes)(y_pred, y_true)
    
        acc = (tp + tn) / (tp + fp + fn + tn + self.epsilon)
        acc = np.sum(acc) / self.num_classes
    
        return acc
    
class Precision(nn.Module):
    def __init__(self, num_classes=1, epsilon=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
    
    def __str__(self):
        return "precision"
    
    def forward(self, y_pred, y_true):
        tp, fp, fn, tn = ConfusionMatrix(self.num_classes)(y_pred, y_true)
    
        precision = tp / (tp + fp + self.epsilon)
        precision = np.sum(precision) / self.num_classes
    
        return precision
    
class Recall(nn.Module):
    def __init__(self, num_classes=1, epsilon=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        
    def __str__(self):
        return "recall"
    
    def forward(self, y_pred, y_true):
        tp, fp, fn, tn = ConfusionMatrix(self.num_classes)(y_pred, y_true)
        
        recall = tp / (tp + fn + self.epsilon)
        recall = np.sum(recall) / self.num_classes
        
        return recall
    
class F1Score(nn.Module):
    def __init__(self, num_classes=1, epsilon=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
    
    def __str__(self):
        return "f1_score"
    
    def forward(self, y_pred, y_true):
        tp, fp, fn, tn = ConfusionMatrix(self.num_classes)(y_pred, y_true)
        
        f1_score = 2 * tp / (2 * tp + fp + fn + self.epsilon)
        f1_score = np.sum(f1_score) / self.num_classes
        
        return f1_score
    
class IoU(nn.Module):
    def __init__(self, num_classes=1, epsilon=1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        
    def __str__(self):
        return "iou"
    
    def forward(self, y_pred, y_true):
        tp, fp, fn, tn = ConfusionMatrix(self.num_classes)(y_pred, y_true)
        
        iou = tp / (tp + fp + fn + self.epsilon)
        iou = np.sum(iou) / self.num_classes
    
        return iou
