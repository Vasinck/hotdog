import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
from torch import nn
import torchvision
from torch import optim


class_1_img_path = './Datasets/hotdog/train/hotdog/'
class_0_img_path = './Datasets/hotdog/train/not-hotdog/'
transfroms_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
])

class my_datasets(data.Dataset):
    def __init__(self, root, transforms):
        imgs = sorted(os.listdir(root))
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.transforms = transforms
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
#        pil_img.show()
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        return data

def get_datas():
    hotdog = my_datasets(class_1_img_path, transfroms_train)
    no_hotdog = my_datasets(class_0_img_path, transfroms_train)
    no_hotdog = torch.stack([no_hotdog[i] for i in range(100)], 0)
    hotdog = torch.stack([hotdog[i] for i in range(100)])
    
    train_datas = torch.cat([no_hotdog, hotdog], 0)
    labels = [0 if i < 100 else 1 for i in range(200)]
    labels = torch.tensor(labels, dtype=torch.long)
    
    train_databets = data.TensorDataset(train_datas, labels)
    data_iter = data.DataLoader(train_databets, 50, True)
    
    model = torchvision.models.resnet18(True)
    for parma in model.parameters():
        parma.requires_grad = False
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(512,2)
    )
        
    l = nn.CrossEntropyLoss()
    op = optim.RMSprop(model.fc.parameters(), 0.003)
    return data_iter, model, l, op

def train(data_iter, model, loss_function, op):
    is_train = False
    save_path = './shibieregou.pkl'
    
    if is_train:
        epochs = 30
        for epoch in range(1, epochs+1):
            for x, y in data_iter:
                out = model(x)
                l = loss_function(out, y)
                op.zero_grad()
                l.backward()
                op.step()
            
            print('epoch %d, loss: %f' % (epoch, l.item()))
            if l.item() < 0.1:
                break
        
        torch.save(model.state_dict(), save_path)
    else:
        model.load_state_dict(torch.load(save_path))
        model.eval()
    

def test(model):
    test_img_path = './Datasets/hotdog/test/hotdog/1200.png'
    test_img = Image.open(test_img_path)
    test_img.show()

    test_img = transforms.Resize([224, 224])(test_img)
    test_img = transforms.ToTensor()(test_img)
    test_img = test_img.expand(1, 3, 224, 224)

    predict = torch.argmax(model(test_img)).item()
    if predict:
        print('有')
    else:
        print('没有')
    

data_iter, model, loss_function, op = get_datas()
train(data_iter, model, loss_function, op)
test(model)

