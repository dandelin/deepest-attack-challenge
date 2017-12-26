from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image

import torch
import pickle
import urllib

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t.unsqueeze(0))
])

classes = pickle.load(urllib.request.urlopen('https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'))

model = models.resnet18(pretrained=True)
model.eval()

# Original deepest image
original = Image.open('deepest.png')
original_tensor = transform(original)

original_var = Variable(original_tensor)
original_infer = model(original_var)
_, original_argmax = torch.max(original_infer, dim=1)
print('Inferring original image.')
print('Softmax Score : {}'.format(torch.nn.functional.softmax(original_infer, dim=1).data[0, 109]))
print('Class : {}'.format(classes[original_argmax.data[0]]))

# Altered deepest image
img = Image.open('deepest_brain.png')
img_tensor = transform(img)
img_var = Variable(img_tensor)
infer = model(img_var)
_, argmax = torch.max(infer, dim=1)

print('Inferring altered image.')
print('Softmax Score : {}'.format(torch.nn.functional.softmax(infer, dim=1).data[0, 109]))
print('Average distance : {}'.format(torch.mean((original_tensor - img_tensor)**2)))
print('Class : {}'.format(classes[argmax.data[0]]))