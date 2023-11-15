import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

def evaluator(preds, gts, probs=None):

    acc, f1, auc = None, None, None


    preds = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    gts = gts.cpu().numpy() if isinstance(gts, torch.Tensor) else gts
    probs = probs.detach().cpu().numpy() if isinstance(probs, torch.Tensor) else probs
    
    acc = accuracy_score(preds, gts)
    f1 = f1_score(preds, gts, average="macro")
    auc = roc_auc_score(gts, probs, multi_class="ovr")
    return acc, f1, auc