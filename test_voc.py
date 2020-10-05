from data import VOCSegmentation
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import colorize_mask, MaskToTensor

# Create transforms
input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

target_transform = MaskToTensor()

val_data = VOCSegmentation('./',
                           image_set='val',
                           transform=input_transform,
                           target_transform=target_transform
                           )

# Create data loader
val_loader = DataLoader(val_data, batch_size=1, num_workers=4, shuffle=False)

# init the model
model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)

if torch.cuda.is_available():
    model.cuda()

model.eval()

with torch.no_grad():
    for image_tensor, mask in val_loader:
        image_tensor = image_tensor.cuda()

        # print(image_tensor.shape)
        # print(mask.shape)

        output = model(image_tensor)['out'][0]

        # print(output.shape) # [21, 256, 256]

        # Check output shape if confused

        # note, mask is on cpu
        label_for_image = mask.squeeze(0)
        print(label_for_image.shape)  # [256, 256]

        # now can calc ECE loss here, or whatever we need

        break
