import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class TorchNeuralNetwork(nn.Module):
    def __init__(self, input_dim, depth, width, activation_fn='relu'):
        super(TorchNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(depth):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, width))
            else:
                self.layers.append(nn.Linear(width, width))
        
        self.output_layer = nn.Linear(width, 1)
        self.activation_fn = activation_fn
        
        if activation_fn == 'tanh':
            self._initialize_weights_xavier()
        elif activation_fn == 'relu':
            self._initialize_weights_he()
    
    def _initialize_weights_xavier(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    def _initialize_weights_he(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if self.activation_fn == 'tanh':
                x = torch.tanh(x)
            elif self.activation_fn == 'relu':
                x = torch.relu(x)
        
        x = self.output_layer(x)
        return torch.sigmoid(x) 

    def train_model(self, X_train, y_train, num_epochs=100, lr=0.01, batch_size=32, verbose=False):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0
            
            for X_batch, y_batch in data_loader:
                optimizer.zero_grad()
                
                outputs = self(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            

    def predict(self, X_test, threshold=0.5):
        self.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        
        with torch.no_grad():
            predictions = self(X_test_tensor).squeeze()
            return (predictions >= threshold).float()
