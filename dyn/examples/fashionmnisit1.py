import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt


# 訓練データをdatasetsからダウンロードa
training_data = datasets.FashionMNIST(
      root="data",
      train=True,
      download=True,
      transform=ToTensor(),
  )

# テストデータをdatasetsからダウンロード
test_data = datasets.FashionMNIST(
      root="data",
      train=False,
      download=True,
      transform=ToTensor(),
  )


### データの中身を確認してみよう

#print(len(training_data))
#print(len(test_data))
#print(training_data[0][0].shape)

#training_data[0][0][0][14][14]
#test_data[10]
#test_data[0][1]

#plt.imshow(training_data[30000][0][0],cmap="gray")
#print(training_data[30000][1])






batch_size = 64

# データローダーの作成
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)

    #最初の１つのデータだけ表示してデータローダーの動作を確認してみる
    break



## google colab の無料プランではcpuしか使えない
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


# modelを定義します
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            #nn.Linear(512, 512),  #ここをコメントアウトしない場合に精度は上がる？
            #nn.ReLU(),           
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
#model = NeuralNetwork()

## どんなニューラルネットワークか確認してみよう
#print(model)

# 初期状態ではニューラルネットワークのパラメータは乱数だが、適当な画像を入力してみよう

#乱数の3x4行列を生成してみる例
torch.rand((3,4))

# 乱数の1x28x28行列をニューラルネットワークに入力してみてどんな出力になるか確認してみよう
print(model(torch.rand((1,28,28)).to(device)))
# こんな感じの１０次元ベクトルになる
##tensor([[0.0000, 0.0371, 0.0000, 0.0000, 0.0000, 0.0914, 0.0466, 0.0678, 0.0229,0.1512]], grad_fn=<ReluBackward0>)


loss_fn = nn.CrossEntropyLoss() #どんな誤差を小さくしたいか定義する
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) #どんな勾配法を実行するか選択する

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # 損失誤差を計算
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  #勾配法を実行

        if batch % 100 == 0:  	# 100回に一回のデバックプリント
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


#epochs = 1
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Done!")


#x=model(torch.tensor(training_data[30000][0]))
#print(x)

x=model(torch.tensor(training_data[30000][0]).to(device))
print(x)
#tensor([[0.5800, 0.0000, 0.0000, 3.5463, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,0.0000]], device='cuda:0', grad_fn=<ReluBackward0>)
## 3番目の値が大きいので３番目の　Dress 	ドレス　と判定されている

#y=x.to('cpu').detach().numpy().copy() 
#print(y)
plt.imshow(training_data[30000][0][0],cmap="gray")
plt.show()
