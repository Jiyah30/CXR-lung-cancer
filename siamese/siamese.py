import timm
import torch
import torch.nn as nn
import torchxrayvision as xrv
import numpy as np

class Siamese(nn.Module):

    def __init__(self, num_classes=3, model_name="densenet121.ra_in1k", pretrained=True, features_only=True):
        super().__init__()

        self.features_only = features_only
        if features_only:
            self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.features = timm.create_model(model_name, num_classes=num_classes, pretrained=pretrained, features_only=features_only, out_indices=[-1])
        else:
            # self.features = timm.create_model(model_name, num_classes=0, pretrained=pretrained)
            self.features = xrv.models.DenseNet(weights="densenet121-res224-nih", drop_rate=0.1)
            self.features.op_threshs = None
            self.features.classifier = nn.Identity()

        in_channels = self.__get_in_channels()
        # self.classifier = nn.Sequential(
        #     nn.Linear(in_channels, int(in_channels // 2)),
        #     nn.BatchNorm1d(int(in_channels // 2)),
        #     nn.CELU(),
        #     nn.Linear(int(in_channels // 2), int(in_channels // 4)),
        #     nn.BatchNorm1d(int(in_channels // 4)),
        #     nn.CELU(),
        #     nn.Linear(int(in_channels // 4), num_classes)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, num_classes)
        )

    def __get_in_channels(self):
        x = torch.randn(1,3,224,224)
        _, _, h, w = self.features(x)[0].shape

        # if self.features_only:
        #     return self.pool(self.features(x)[-1]).shape[1]
        # return self.features(x).shape[-1]
        return h*w*h*w

    def forward(self, before_image, after_image):
        if self.features_only:
            # before_feature_maps = self.features(before_image)[-1]
            # after_feature_maps = self.features(after_image)[-1]
            before_feature_maps = self.features(before_image)[0]
            after_feature_maps = self.features(after_image)[0]

            # before_features = self.pool(before_feature_maps)[...,0,0]
            # after_features = self.pool(after_feature_maps)[...,0,0]
            # return before_feature_maps, after_feature_maps
        else:
            before_features = self.features(before_image)
            after_features = self.features(after_image)

        # before_features = before_features.permute(1, 0).detach().cpu().numpy()
        # after_features = after_features.permute(1, 0).detach().cpu().numpy()
        
        bs, c, h, w = before_feature_maps.shape
        maps = []
        for i in range(before_feature_maps.size(0)):
            before_feature = before_feature_maps[i].reshape(before_feature_maps[i].size(0), -1)
            after_feature = after_feature_maps[i].reshape(after_feature_maps[i].size(0), -1)
            combine = torch.cat([before_feature, after_feature], dim = -1)
            combine = combine.permute(1, 0)
            combine = torch.corrcoef(combine)
            combine = combine[:h*w, h*w:]
            maps.append(combine)

        fusion_features = torch.stack(maps, dim = 0)
        fusion_features = fusion_features.reshape(fusion_features.size(0), -1)   
                
        # fusion_features = before_features - after_features
        # fusion_features = torch.cat([before_features, after_features], dim=1)
        out = self.classifier(fusion_features)
        return out
    
    def get_optim_params(self):
        return [
            {"params": self.classifier.parameters(), "lr": 1e-3},
        ]