'''
Plots confusion matrices based on the predictions of the model (results from cnn, lstm, lstm_fcn)

'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

class CNN1D(nn.Module):
    def __init__(self, num_classes=5):
        super(CNN1D, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=32, padding=16)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.7)
        self.pool1 = nn.MaxPool1d(kernel_size=4)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=12, padding=6)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.7)
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=6, padding=3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.7)
        self.pool3 = nn.MaxPool1d(kernel_size=4)

        # Calculate the size of the input for the first fully connected layer
        self.fc_input_size = 64 * 5

        self.fc1 = nn.Linear(self.fc_input_size, 100)
        self.relu_fc1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(100, num_classes)
        self.dropout_fc2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool1(self.dropout1(self.relu1(self.conv1(x))))
        x = self.pool2(self.dropout2(self.relu2(self.conv2(x))))
        x = self.pool3(self.dropout3(self.relu3(self.conv3(x))))

        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = self.dropout_fc1(self.relu_fc1(self.fc1(x)))
        x = self.fc2(x)

        return x

class LSTMClassifier(nn.Module):
  def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.layer_dim = layer_dim
    self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob)
    self.fc = nn.Linear(hidden_dim, output_dim)
    self.dropout = nn.Dropout(dropout_prob)
    self.batch_norm = nn.BatchNorm1d(hidden_dim)

  def forward(self, x):
    h0, c0 = self.init_hidden(x)
    out, (hn, cn) = self.rnn(x, (h0, c0))
    out = self.batch_norm(out[:, -1, :])
    out = self.dropout(out)
    out = self.fc(out)
    return out

  def init_hidden(self, x):
    h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
    c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
    return [t for t in (h0, c0)]

class LSTM_FCN(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, num_lstm_layers, num_classes, lstm_dropout=0.2):
        super(LSTM_FCN, self).__init__()
        
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers

        # LSTM part
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, num_lstm_layers, 
                            batch_first=True, dropout=lstm_dropout if num_lstm_layers > 1 else 0)
        
        # Dropout after LSTM
        self.dropout_lstm_out = nn.Dropout(lstm_dropout)
        
        # Fully Convolutional Network part
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=8, padding=3)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        
        # Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(num_features=128, momentum=0.99, eps=1e-4)
        self.bn2 = nn.BatchNorm1d(num_features=256, momentum=0.99, eps=1e-4)
        self.bn3 = nn.BatchNorm1d(num_features=128, momentum=0.99, eps=1e-4)
        
        # Global Average Pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully Connected layer for the combined output
        self.fc = nn.Linear(lstm_hidden_dim + 128, num_classes)
        
    def forward(self, x):
        lstm_input = x.transpose(1, 2)  # Transpose to (batch_size, input_dim, sequence_length)
        
        # FCN part
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_avg_pool(x)
        fcn_out = x.squeeze(-1)  # Remove the last dimension
        
        # LSTM part
        h0 = torch.zeros(self.num_lstm_layers, lstm_input.size(0), self.lstm_hidden_dim).to(lstm_input.device)
        c0 = torch.zeros(self.num_lstm_layers, lstm_input.size(0), self.lstm_hidden_dim).to(lstm_input.device)
        lstm_out, _ = self.lstm(lstm_input, (h0, c0))
        lstm_out = self.dropout_lstm_out(lstm_out[:, -1, :])  # Apply dropout to LSTM output
        
        # Concatenate LSTM and FCN outputs
        combined = torch.cat((lstm_out, fcn_out), dim=1)
        
        # Fully connected layer for classification
        out = self.fc(combined)
        
        return out

class CustomDataset(Dataset):
    def __init__(self, data_tensor, labels_tensor):
        self.data = data_tensor#.unsqueeze(1)  # Add a channel dimension
        self.labels = labels_tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def data_setup(catalogue, all_data):
    '''
    Takes the data in directly from the files of stacked data and returns
    the data and labels in separate X and y arrays
    '''

    # Generate a permutation of indices
    permutation = np.random.permutation(len(all_data))

    # Shuffle the NumPy array using the permutation
    all_data = all_data[permutation]

    # Shuffle the DataFrame using the permutation
    catalogue = catalogue.iloc[permutation].reset_index(drop=True)

    # Dictionary to hold class data
    class_data = {
        'antenna': [],
        'bus': [],
        'cone': [],
        'panel': [],
        'rod': []
    }

    # Classify data based on catalogue labels
    for index, row in enumerate(all_data):
        obj = catalogue.iloc[index]['Object']
        if 'antenna' in obj:
            class_data['antenna'].append(row)
        elif 'bus' in obj:
            class_data['bus'].append(row)
        elif 'cone' in obj:
            class_data['cone'].append(row)
        elif 'panel' in obj:
            class_data['panel'].append(row)
        else:
            class_data['rod'].append(row)

    # Determine the size of the smallest class
    smallest_class_size = min(len(class_data[key]) for key in class_data)

    # Print the smallest class size
    print(f'Smallest class size: {smallest_class_size}')

    # Trim all classes to the size of the smallest class
    for key in class_data:
        class_data[key] = class_data[key][:smallest_class_size]

    # Convert lists to numpy arrays
    for key in class_data:
        class_data[key] = np.array(class_data[key])

    # Create labels for each class
    labels = {
        'antenna': 0,
        'bus': 1,
        'cone': 2,
        'panel': 3,
        'rod': 4
    }

    class_labels = {key: np.full(len(class_data[key]), labels[key]) for key in class_data}

    # Combine the data and labels
    X_stack = np.concatenate(list(class_data.values()))
    y_stack = np.concatenate(list(class_labels.values()))

    # Reshuffle the data and labels
    permutation = np.random.permutation(len(X_stack))
    X = X_stack[permutation]
    y = y_stack[permutation]

    return X, y

def convert_to_tensor(X_train, X_val,X_test, y_train, y_val, y_test):
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.int64)
    y_val_tensor = torch.tensor(y_val, dtype=torch.int64)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64)

    # print(y_test)
    # print(y_val)

    return X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor

batch_size = 128

best_path = '/home/anne/scripts/obj_sim/figures/models/lstm_best1.pth'
X_test_path = '/home/anne/scripts/obj_sim/figures/models/Xtest_tensor_lstm1.pt' # Change for cnn, lstm, or lstmfcn
y_test_path = '/home/anne/scripts/obj_sim/figures/models/ytest_tensor_lstm1.pt'

# model = CNN1D(num_classes=5)
model = LSTMClassifier(input_dim=1, hidden_dim=64, layer_dim=4, output_dim=5, dropout_prob=0.2) # need to use parameters form BO here
# model = LSTM_FCN(input_dim=1, lstm_hidden_dim=64, num_lstm_layers=4, num_classes=5)

# ---------------------------------------------------------------------------------------------------------------------


#Import test data
X_test = torch.load(X_test_path)
y_test = torch.load(y_test_path)

test_set = CustomDataset(X_test, y_test)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

# model = CNN1D(num_classes=5)

model.load_state_dict(torch.load(best_path, map_location=torch.device('cpu')))  
# model.load_state_dict(torch.load('/home/aadriano/projects/def-ka3scott/aadriano/amos2024/cnn1d_best1.pth'))

model.eval()

# Run model on test data
predictions = []
for batch, _ in test_loader:
  # batch = batch.permute(0, 2, 1)
  out = model(batch)
  y_hat = F.log_softmax(out, dim=1).argmax(dim=1)
  predictions += y_hat.tolist()

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Calculate F1 score
f1 = f1_score(y_test, predictions, average='weighted')
print(f"F1 Score: {f1:.2f}")

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Get a classification report
class_report = classification_report(y_test, predictions)
print("Classification Report:")
print(class_report)

classes = ['Dish', 'Cuboid', 'Cone', 'Panel', 'Rod']
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix: CNN Shape Classification')
plt.show()