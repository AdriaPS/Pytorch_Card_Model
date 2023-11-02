import timm
import torch.nn as nn


class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifier, self).__init__()  # To initialize the object with all from the parent class.
        # Here we will define all the parts of the model.
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)  # We are using a timm architecture
        # to train the model.
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])  # We want to remove the last layer of
        # the base model, so we add this line of code.
        enet_out_size = 1280  # This is the default size of the efficientnet architecture, we need to convert it to
        # our size, which in our case is 53, for that, we will make a classifier.
        self.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        # Here we connect the different parts of the model and return an output.
        x = self.features(x)
        output = self.classifier(x)
        return output
