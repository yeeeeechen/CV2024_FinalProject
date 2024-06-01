import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class FastRCNN(nn.Module):
    def __init__(self):
        super(FastRCNN, self).__init__()

        self.fastrcnn = models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
            
        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        num_classes = 2  # 1 class (door) + background
        # get number of input features for the classifier
        in_features = self.fastrcnn.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.fastrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


        

    def forward(self, image, target=None):
        return self.fastrcnn(image, target)
    
if __name__ == '__main__':
    # model = MyNet()
    model = FastRCNN()
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Parameter #: {total_params}")
    print(model)
