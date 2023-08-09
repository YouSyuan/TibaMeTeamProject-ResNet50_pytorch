
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

def myResNet50(class_numbers=95, in_weights=ResNet50_Weights.IMAGENET1K_V2):
    model = resnet50(weights=in_weights)

    # Freeze model
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, class_numbers)  # 替換新的全連接層

    return model




