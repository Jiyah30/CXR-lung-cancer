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
    parser.add_argument('--base_root', type=str, default='/home/s311657007/Project/new_images/224/', help='path to the base data directory')
    parser.add_argument('--csv_name', type=str, default='lung_cancer.csv', help='name of the CSV file containing the dataset')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--save_name', type=str, default='best_80.pth', help='name of the file to save the trained model')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for the optimizer')
    parser.add_argument('--device', type=str, default="cuda", help='weight decay for the optimizer')
    parser.add_argument('--num_classes', type=int, default=3, help='number of classes')
    parser.add_argument('--model_name', type=str, default="tv_densenet121", help='model name')
    parser.add_argument('--rank', type=int, default=0, help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='model name')
    args = parser.parse_args()
    return args



def main():

    seed_everything(42)

    args = parse_args()

    model = Siamese(args.num_classes, args.model_name, True)
    device = f"cuda"
    model.to("cuda")

    best_ckpt = torch.load(f"res/{args.save_name}", map_location=device)
    model.load_state_dict(best_ckpt["model"])

    _, validloader, testloader = get_dataloader(args.base_root, args.csv_name, args.batch_size)
    criterion = nn.CrossEntropyLoss()

    valid_loss, valid_acc, valid_f1, valid_auc = validation(model, validloader, criterion, device)
    print("*"*10, " Valid ", "*"*10)
    print(f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_acc:.2f}%, Valid F1: {valid_f1:.2f}%, Valid AUC: {valid_auc:.2f}%")
    test_loss, test_acc, test_f1, test_auc = validation(model, testloader, criterion, device)
    print("*"*10, " Test ", "*"*10)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%, Test F1: {test_f1:.2f}%, Test AUC: {test_auc:.2f}%")

if __name__ == "__main__":
    main()