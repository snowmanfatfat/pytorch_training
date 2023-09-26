import torch 
import torch.nn as nn
import torchvision.models as models # 載入pretrained model
from torch.utils.data import Dataset, DataLoader

x=torch.randn(4,5) # n表standard normalization，即高斯白噪聲，而rand是均勻分布
y=torch.randn(4,5)
print(x)
print(y)

x_max1=torch.max(x) # 在所有tensor中找最大值，會回傳一個tensor，且預設keepdim=False，即不會保持維度除非特別設定
print(x_max1)

x_max2, idx=torch.max(x,0) # 還會回傳指標
print(x_max2,"\n",idx)

x_max3, idx=torch.max(x,0,True) # 保持維度
print(x_max3,"\n",idx)

p=(x_max3,idx)
torch.max(x,0,out=p) # 查看原始torch.max說明檔可發現out這個參數放在*後面，表示他是keyword parameter，所以必須加上名字才能使用，這種參數也可以不設定，另外在*前面的參數不用加上名字但一定要設定，除非有預設值
# out必須是tensor,logntensor格式才可，用途就是如果已經有一個定義好的空間，可以拿它來放max出來的值，也可不設定，就會產生新的tensor來裝
print(p)

x_max4=torch.max(x,y) # 把x,y用elementwise的方式比較
print(x_max4)

model = torch.nn.Linear(5,1).to("cuda:0") # Linear表fully-connected layer表fully-connected layer，(5,1)表input長度5 output長度1
x = torch.Tensor([1,2,3,4,5]).to("cuda:0") # 若寫"cpu"會runtime error，model和data要在同個device上才行
y = model(x)
print(y)

x=torch.randn(4,5)
y=torch.randn(5,4)
print(y)
y=y.transpose(0,1)
print(y)
z=x+y # 要shape相同才可以相加
print(x)

resnet18=models.resnet18().to("cuda:0")
data=torch.randn(2048,3,244,244) # create fake data
# out=resnet18(data.to("cuda:0")) 一次全放進去會out of memory
for d in data:
    out=resnet18(d.to("cuda:0").unsqueeze(0)) # 改一次放一筆就不會了，增加維度0，如此一來才能變成4d input
print(out.shape)

L=nn.CrossEntropyLoss()
outs=torch.randn(5,5)
labels=torch.Tensor([1,2,3,4,0]).long()
loss_val=L(outs, labels) # label這樣寫其實是浮點數，但label不能是float所以用long轉成整數
print(loss_val)

# dataset:一堆有組織的資料 dataloader:iterate從dataset中取出資料，他必須知道資料的長度，可以suffle資料，回傳資料的index，此時dataset必須依據index吐出對應的資料，所以dataset要有兩個功能__getitem__()和__len__()，以供dataloader使用
# dataset = "abcdefghijklmnopqrstuvwxyz"
# for datapoint in dataset:
#     print(datapoint) # 可使用for loop來實現dataloader
    
# class ExampleDataset(Dataset):
#     def __init__(self):
#         self.data="abcdefghijklmnopqrstuvwxyz"
        
#     def __getitem__(self, index):
#         return self.data[index]
    
#     def __len__(self):
#         return len(self.data)
    
# dataset=ExampleDataset()
# dataloader=DataLoader(dataset,batch_size=1,shuffle=True)
# # for datapoint in dataloader: # 用dataloader的好處是可以shuffle
# #     print(datapoint)
    
# 用dataloader還可以進行data augmentation
class ExampleDatasetAug(Dataset):
    def __init__(self):
        self.data="abcdefghijklmnopqrstuvwxyz"
        
    def __getitem__(self, index):
        if index>=len(self.data): # 注意是>=，因為編號26就已經超出長度了
            return self.data[index%26].upper() # 當回傳超過原本長度的index時，輸出資料是原本位置的大寫
        else:
            return self.data[index]
    
    def __len__(self):
        return 2*len(self.data) # 把dataset放大2倍
    
datasetA=ExampleDatasetAug()
dataloaderA=DataLoader(datasetA,batch_size=1,shuffle=True)
out=[]
for datapoint in dataloaderA: # 用dataloader的好處是可以shuffle
    out.append(datapoint)