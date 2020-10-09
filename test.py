import torch
from data.voc import _give_val_loader
from utils.crf import DenseCRF
import numpy as np

# Create deeplabv2 model
model = torch.hub.load("kazuto1011/deeplab-pytorch", 
                       "deeplabv2_resnet101", 
                       pretrained="voc12", 
                       n_classes=21)

if torch.cuda.is_available():
    model.cuda()
model.eval()

# print(model)

# Get Val Loader
val_loader = _give_val_loader()

# Declare CRF
postprocessor = DenseCRF(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,
    )

with torch.no_grad():
    for image, label, raw_img in val_loader:

        image = image.cuda()
        print(image.shape) # [1, 3, H, W]

        logits = model(image)

        _, H, W = label.shape
        logits = torch.nn.functional.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )

        probs = torch.softmax(logits, dim=1)[0]
        probs = probs.cpu().numpy()
        # print(probs.shape) # [21, H, W]

        # Now run the post-processor
        raw_img = raw_img.squeeze(0)
        raw_img = np.array(raw_img).astype(np.uint8)
        print(raw_img.shape)

        probs = postprocessor(raw_img, probs)

        print(probs.shape)
        # print(probs)

        labelmap = np.argmax(probs, axis=0)
        print(labelmap.shape)
        print(labelmap.sum()) # why zero ??

        break
