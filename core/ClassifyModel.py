import torch
from torch import nn
import torch.nn.functional as F
from monai.networks.nets import ResNetFeatures, DenseNet121, SEResNet50
import os

class CrossModalNet(nn.Module):
    def __init__(self, image_network_name='DenseNet', num_image_channels=1, num_csv_input_features=10, num_classes=2, pretrained_path='./model/resnet_18_23dataset.pth'):

        super().__init__()

        network_dims = {
            'DenseNet': 1024,
            'ResNet': 512,
            'SeResNet': 2048,
        }

        # ---------------- 图像特征网络 ----------------
        if image_network_name == 'ResNet':
            self.image_feature_network = ResNetFeatures(
                spatial_dims=3, in_channels=num_image_channels, model_name='resnet18', pretrained=False
            )
            if pretrained_path and os.path.exists(pretrained_path):
                checkpoint = torch.load(pretrained_path, map_location="cpu")
                state_dict = checkpoint.get("state_dict") or checkpoint
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model_dict = self.image_feature_network.state_dict()
                pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
                if pretrained_dict:
                    self.image_feature_network.load_state_dict(pretrained_dict, strict=False)
                    print(f"成功从 {pretrained_path} 加载了预训练权重。")
                else:
                    print(f"警告: {pretrained_path} 中没有与图像网络匹配的预训练权重。")
        elif image_network_name == 'DenseNet':
            self.image_feature_network = DenseNet121(
                spatial_dims=3, in_channels=num_image_channels, out_channels=network_dims[image_network_name]
            )
        elif image_network_name == 'SeResNet':
            self.image_feature_network = SEResNet50(
                spatial_dims=3, in_channels=num_image_channels
            )
        else:
            raise ValueError(f"不支持的影像网络: {image_network_name}.")

        self.image_network_name = image_network_name
        image_feature_dim = network_dims[image_network_name]

        # ---------------- CSV 升维 + 门控（始终存在） ----------------
        self.csv_mlp = nn.Sequential(
            nn.Linear(num_csv_input_features, image_feature_dim),
            nn.BatchNorm1d(image_feature_dim),
            nn.ReLU()
        )
        self.csv_gate = nn.Sequential(
            nn.Linear(image_feature_dim, image_feature_dim),
            nn.Sigmoid()
        )

        # ---------------- 融合层 ----------------
        combined_input_dim = image_feature_dim  # 门控后 CSV 与图像特征同维度，直接相加
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # ---------------- 分类输出层 ----------------
        self.num_classes = num_classes  # 新增类别数量参数
        self.classification_head = nn.Linear(256, num_classes)

    def forward(self, image_input, csv_input):
        # 1. 图像特征提取
        if self.image_network_name == 'ResNet':
            image_features = self.image_feature_network(image_input)[-1]
        elif self.image_network_name == 'DenseNet':
            image_features = self.image_feature_network.features(image_input)
        elif self.image_network_name == 'SeResNet':
            image_features = self.image_feature_network.features(image_input)
        else:
            raise ValueError(f"无法识别的网络名称: {self.image_network_name}")

        image_features_pooled = F.adaptive_avg_pool3d(image_features, (1, 1, 1))
        image_features_flattened = torch.flatten(image_features_pooled, 1)

        # 2. CSV 特征处理与门控融合
        csv_emb = self.csv_mlp(csv_input)
        gate = self.csv_gate(csv_emb)
        csv_features_scaled = csv_emb * gate
        combined_features = image_features_flattened + csv_features_scaled

        # 3. 融合层
        fused_features = self.fusion_layer(combined_features)

        # 4. 计算分类输出
        output = self.classification_head(fused_features)

        return output, image_features