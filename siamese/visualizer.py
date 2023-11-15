import torch 
import torch.nn as nn

from siamese import Siamese
from dataloader import get_dataloader
from utils import seed_everything
from trainer import train_one_epoch, validation
import torchvision 
import matplotlib.pyplot as plt
import cv2
import numpy as np

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
    model = Siamese(args.num_classes, args.model_name, True, True)

    device = f"cuda"

    model.to("cuda")

    criterion = nn.CrossEntropyLoss()
    
    trainloader, validloader, testloader = get_dataloader(args.base_root, args.csv_name, 1)
    
    best_ckpt = torch.load(f"res/{args.save_name}", map_location=device)
    model.load_state_dict(best_ckpt["model"])

    weight = list(model.classifier.parameters())[0]
    features_len = int(weight.shape[1]/2)
    with torch.no_grad():
        for idx, (before_images, after_images, labels) in enumerate(validloader):
            before_images = before_images.to(device=device, dtype=torch.float)
            after_images = after_images.to(device=device, dtype=torch.float)
            labels = labels.to(device=device, dtype=torch.long)

            before_faeture_maps, after_feature_maps = model(before_images, after_images)
            before_faeture_maps = torch.mm(weight[labels][:,:features_len], before_faeture_maps.reshape(features_len, -1))
            after_feature_maps = torch.mm(weight[labels][:,features_len:], after_feature_maps.reshape(features_len, -1))
            
            before_faeture_maps = cv2.resize(before_faeture_maps.reshape(7, 7).detach().cpu().numpy(), (224, 224))
            after_feature_maps = cv2.resize(after_feature_maps.reshape(7, 7).detach().cpu().numpy(), (224, 224))
            before_faeture_maps = np.uint8(255 * before_faeture_maps)
            after_feature_maps = np.uint8(255 * after_feature_maps)
            before_faeture_maps = cv2.applyColorMap(before_faeture_maps, cv2.COLORMAP_JET)
            after_feature_maps = cv2.applyColorMap(after_feature_maps, cv2.COLORMAP_JET)
            superimposed_img_before = before_faeture_maps * 0.4 + 255 * (before_images[0].permute(1,2,0).detach().cpu().numpy())
            superimposed_img_after = after_feature_maps * 0.4 + 255 * (after_images[0].permute(1,2,0).detach().cpu().numpy())
            # before_faeture_maps = torch.nn.functional.interpolate(before_faeture_maps.reshape(7, 7)[None,None], size=(224, 224))
            # after_feature_maps = torch.nn.functional.interpolate(after_feature_maps.reshape(7, 7)[None,None], size=(224, 224))

            # before_faeture_maps = torch.cat([before_faeture_maps, before_faeture_maps, before_faeture_maps], dim=1)
            # after_feature_maps = torch.cat([after_feature_maps, after_feature_maps, after_feature_maps], dim=1)

            #image = torchvision.utils.make_grid([before_images[0], after_images[0], superimposed_img_before[0], superimposed_img_after[0]], nrow=1, normalize=True)
            #print(image.shape)
            #plt.imsave(f"image{idx}.jpg", image.permute(1,2,0).detach().cpu().numpy())
            superimposed_img_before = (superimposed_img_before-superimposed_img_before.min())/(superimposed_img_before.max()-superimposed_img_before.min())
            superimposed_img_after = (superimposed_img_after-superimposed_img_after.min())/(superimposed_img_after.max()-superimposed_img_after.min())
            new = np.concatenate([superimposed_img_before, superimposed_img_after], axis = 1)
            plt.imsave(f"image{idx}_{labels[0]}.jpg", new)
            

if __name__ == "__main__":
    main()