import torch
from torch import nn
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1).to(device)
y = weight * X + bias


train_spilt = int(0.8 * len(X))
X_train, y_train = X[:train_spilt], y[:train_spilt]
X_test, y_test = X[train_spilt:], y[train_spilt:]

# def plot_predictions(train_data=X_train,
#                      train_labels=y_train,
#                      test_data=X_test,
#                      test_labels=y_test,
#                      predictions=None):
#     plt.figure()
#     plt.scatter(train_data, train_labels, c='b', s=8, label='Training data')
#     plt.scatter(test_data, test_labels, s=8, c='g', label='Testing data')
    
#     if predictions is not None:
#         plt.scatter(test_data, predictions, s=8, c='r', label='Predictions')
#     plt.legend(prop={'size':14})

# plot_predictions()

class LinearRegressionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = nn.Parameter(torch.randn(1, dtype=torch.float32, device=device), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float32, device=device), requires_grad=True)
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias

model_0 = LinearRegressionModel()

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

epochs = 400

train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    model_0.train()
    y_pred = model_0(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_0.eval()
    
    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, y_test)

        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach())
            test_loss_values.append(test_loss.detach())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
