import torch.nn as nn
import copy
import FrEIA.framework as Ff
import FrEIA.modules as Fm

import torchvision
from .decoder import Decoder
from .msff import MSFF

class pama(nn.Module):
    def __init__(self, memory_bank, feature_extractor):
        super(pama, self).__init__()

        self.memory_bank = memory_bank
        self.feature_extractor = feature_extractor
        self.msff = MSFF()
        self.decoder = Decoder()

        self.net = torchvision.models.resnet34(pretrained=True)
        inchannel = self.net.fc.in_features
        self.net.fc = nn.Linear(inchannel, 2)
        self.net.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, inputs,masks):
        # extract features
        features = self.feature_extractor(inputs)
        f_in = features[0]
        f_out = features[-1]
        f_ii = features[1:-1]

        features_for_update=[i for i in f_ii]
        # extract concatenated information(CI)
        concat_features = self.memory_bank.select(features = f_ii)

        self.memory_bank.update2(features_for_update,masks)

        # Multi-scale Feature Fusion(MSFF) Module
        msff_outputs = self.msff(features = concat_features)

        # decoder
        predicted_mask = self.decoder(
            encoder_output  = f_out,
            concat_features = [f_in] + msff_outputs.detach()
        )
        # print(predicted_mask.shape)
        # classifier=self.net(predicted_mask.detach())

        return predicted_mask,msff_outputs
