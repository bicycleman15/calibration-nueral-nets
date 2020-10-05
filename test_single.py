import torch
import torchvision

from PIL import Image
from torchvision import transforms
from utils import colorize_mask

model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)

input_image = Image.open('input.jpg')

preprocess = transforms.Compose([
    # transforms.Resize([225, 225]), # we don't need to resize I guess the image
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

model.eval()

print(input_batch.shape)

with torch.no_grad():
    output = model(input_batch)['out'][0]  # here we have the predictions

predictions = output.max(0)[1].cpu().numpy()
mask = colorize_mask(predictions)
mask.save(open('output.png', 'wb'))
