import os
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F

class AudioDataset(Dataset):
    def __init__(self, audio_files, labels):
        self.audio_files = audio_files
        self.labels = labels

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file = self.audio_files[idx]
        label = self.labels[idx]
        y, sr = librosa.load(file, sr=16000)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc = np.expand_dims(mfcc, axis=0)

        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(32 * 6 * 31, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_data(root_dir):
    audio_files = []
    labels = []
    for split in ['training', 'validation', 'testing']:
        for label, folder in zip([0, 1], ['real', 'fake']):
            path = os.path.join(root_dir, split, folder)
            for filename in os.listdir(path):
                audio_files.append(os.path.join(path, filename))
                labels.append(label)

    return audio_files, labels

def train(model, dataloaders, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in tqdm(dataloaders['training']):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloaders['training'])}, Accuracy: {100 * correct / total}%")

if __name__ == "__main__":
    root_dir = '/content/drive/MyDrive/for-2sec/for-2seconds'
    audio_files, labels = load_data(root_dir)

    train_files, val_files, train_labels, val_labels = train_test_split(audio_files, labels, test_size=0.2, random_state=42)

    train_dataset = AudioDataset(train_files, train_labels)
    val_dataset = AudioDataset(val_files, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    dataloaders = {'training': train_loader, 'validation': val_loader}

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, dataloaders, criterion, optimizer)
    torch.save(model.state_dict(), 'audio_model.pth')
