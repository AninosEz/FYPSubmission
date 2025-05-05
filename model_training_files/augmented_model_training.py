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
import random

class AudioDataset(Dataset):
    def __init__(self, audio_files, labels, augment=False):
        self.audio_files = audio_files
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.audio_files)

    def augment_audio(self, y):
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.005, size=y.shape)
            y = y + noise
        if random.random() < 0.5:
            shift = np.random.randint(-1600, 1600)
            y = np.roll(y, shift)
        return y

    def __getitem__(self, idx):
        file = self.audio_files[idx]
        label = self.labels[idx]
        y, sr = librosa.load(file, sr=16000)

        if self.augment:
            y = self.augment_audio(y)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # Increased MFCCs
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)  # Normalize
        mfcc = np.expand_dims(mfcc, axis=0)

        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(32 * 20 * 31, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
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

def train(model, dataloaders, criterion, optimizer, num_epochs=10, device='cpu'):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(dataloaders['train'], desc=f"🧠 Epoch {epoch+1}/{num_epochs} - Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        acc = 100 * train_correct / train_total
        print(f"✅ Train Epoch {epoch+1} | Loss: {train_loss/len(dataloaders['train']):.4f} | Accuracy: {acc:.2f}%")

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = 100 * val_correct / val_total
        print(f"📊 Validation Accuracy: {val_acc:.2f}%")

        save_dir = '/content/drive/MyDrive/for-2sec/augmented_models'
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, f'augmented_model1_epoch{epoch+1}.pth'))
        print(f"💾 Saved model at {os.path.join(save_dir, f'augmented_model1_epoch{epoch+1}.pth')}")

if __name__ == "__main__":
    root_dir = '/content/drive/MyDrive/for-2sec/for-2seconds'
    audio_files, labels = load_data(root_dir)

    train_files, val_files, train_labels, val_labels = train_test_split(audio_files, labels, test_size=0.2, random_state=42, stratify=labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = AudioDataset(train_files, train_labels, augment=True)
    val_dataset = AudioDataset(val_files, val_labels, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    dataloaders = {'train': train_loader, 'val': val_loader}

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, dataloaders, criterion, optimizer, num_epochs=10, device=device)

    torch.save(model.state_dict(), 'augmented_model_final.pth')
    print("✅ Final model saved to augmented_model_final.pth")
