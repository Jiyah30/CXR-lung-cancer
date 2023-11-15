import numpy as np
import torch 
import wandb
import torch.nn as nn

from siamese import Siamese
from sklearn.model_selection import train_test_split
from dataloader import get_dataloader
from utils import seed_everything, wandb_settings
from trainer import train_one_epoch, validation

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')
    parser.add_argument('--base_root', type=str, default='/home/s311657019/CXR/Photometric_Interpretation/', help='path to the base data directory')
    parser.add_argument('--csv_name', type=str, default='lung_cancer.csv', help='name of the CSV file containing the dataset')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--save_name', type=str, default='model.pth', help='name of the file to save the trained model')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for the optimizer')
    parser.add_argument('--device', type=str, default="cuda", help='weight decay for the optimizer')
    parser.add_argument('--num_classes', type=int, default=3, help='number of classes')
    parser.add_argument('--model_name', type=str, default="resnet18", help='model name')
    parser.add_argument('--rank', type=int, default=0, help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='model name')
    args = parser.parse_args()
    return args



def main():

    seed_everything(42)

    args = parse_args()

    # d = vars(args)
    # name = args.model_name + "-" + args.loss_func

    device = f"cuda:{args.rank}"
    model = Siamese(args.num_classes, args.model_name, True, True)
    
    # device = f"cuda"
    # model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5)

    trainloader, validloader, testloader = get_dataloader(args.base_root, args.csv_name, args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(trainloader))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, min_lr=1e-6, mode="min")

    best_val_loss = float("inf")
    best_val_acc = 0
    history = {
        "train": {
            "loss": [],
            "acc": [],
            "f1": []
        },
        "valid": {
            "loss": [],
            "acc": [],
            "f1": []
        },
    }
    for epoch in range(args.epochs): 
        train_loss, train_acc, train_f1, tarin_auc = train_one_epoch(model, trainloader, optimizer, scheduler, criterion, device)
        valid_loss, valid_acc, valid_f1, valid_auc = validation(model, validloader, criterion, device)
        # test_loss, test_acc, test_f1 = validation(model, testloader, criterion, device)

        # scheduler.step(valid_loss)
        
        # Log the loss and validation result
        history["train"]["loss"].append(train_loss)
        history["train"]["acc"].append(train_acc)
        history["train"]["f1"].append(train_f1)
        history["valid"]["loss"].append(valid_loss)
        history["valid"]["acc"].append(valid_acc)
        history["valid"]["f1"].append(valid_f1)

        print(f'Epoch[{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Train F1: {train_f1:.2f}% | Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_acc:.2f}%, Valid F1: {valid_f1:.2f}% | LR: {optimizer.state_dict()["param_groups"][0]["lr"]:.6f}')
        # print(f'Epoch[{epoch+1}/{args.epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, Test F1: {train_f1:.2f}% | Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%, Test F1: {test_f1:.2f}% | LR: {optimizer.state_dict()["param_groups"][0]["lr"]:.6f}')
    
        if valid_acc > best_val_acc:
            save_file = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "args": args
            }
            best_val_acc = valid_acc
            torch.save(save_file, f"res/{args.save_name}")

    best_ckpt = torch.load(f"res/{args.save_name}", map_location=device)
    model.load_state_dict(best_ckpt["model"])

    test_loss, test_acc, test_f1, test_auc = validation(model, testloader, criterion, device)
    print("*"*10, " Test ", "*"*10)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%, Test F1: {test_f1:.2f}%, Test AUC: {test_auc:.2f}%")

if __name__ == "__main__":
    main()