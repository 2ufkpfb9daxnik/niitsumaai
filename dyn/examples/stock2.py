class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3, 10),　#３日
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),　#１日
            nn.ReLU()
        )
    def forward(self, x):
        ret = self.linear_relu_stack(x)
        return ret
