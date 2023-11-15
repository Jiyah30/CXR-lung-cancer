import torch
from evaluator import evaluator

def train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device):
    model.train()
    train_loss = .0
    predictions, ground_truths, probabilities = [], [], []
    for before_images, after_images, labels in train_loader:
        before_images = before_images.to(device=device, dtype=torch.float)
        after_images = after_images.to(device=device, dtype=torch.float)
        labels = labels.to(device=device, dtype=torch.long)
        
        optimizer.zero_grad()
        logits = model(before_images, after_images)
        loss = criterion(logits, labels)
        loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        train_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        probs = logits.softmax(dim=1)
        
        predictions.append(preds)
        ground_truths.append(labels)
        probabilities.append(probs)
        
    train_loss /= len(train_loader)
    
    predictions = torch.cat(predictions)
    ground_truths = torch.cat(ground_truths)
    probabilities = torch.cat(probabilities)
    train_acc, train_f1, train_auc = evaluator(predictions, ground_truths, probabilities)
    
    return train_loss, 100*train_acc, 100*train_f1, 100*train_auc
        
def validation(model, valid_loader, criterion, device):
    model.eval()
    valid_loss = .0
    predictions, ground_truths, probabilities = [], [], []
    with torch.no_grad():
        for before_images, after_images, labels in valid_loader:
            before_images = before_images.to(device=device, dtype=torch.float)
            after_images = after_images.to(device=device, dtype=torch.float)
            labels = labels.to(device=device, dtype=torch.long)

            logits = model(before_images, after_images)
            loss = criterion(logits, labels)
            
            valid_loss += loss.item()
            probs = logits.softmax(dim=1)
            preds = torch.argmax(logits, dim=1)
        
            predictions.append(preds)
            ground_truths.append(labels)
            probabilities.append(probs)
        
    valid_loss /= len(valid_loader)
    
    predictions = torch.cat(predictions)
    ground_truths = torch.cat(ground_truths)
    probabilities = torch.cat(probabilities)
    valid_acc, valid_f1, valid_auc = evaluator(predictions, ground_truths, probabilities)
    return valid_loss, 100*valid_acc, 100*valid_f1, 100*valid_auc
    