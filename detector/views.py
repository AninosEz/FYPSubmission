import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

UPLOAD_DIR = os.path.join(settings.BASE_DIR, 'uploads')
MODEL_DIR = os.path.join(settings.BASE_DIR, 'detector', 'models')
os.makedirs(UPLOAD_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MFCC 40
class CNNModel1(nn.Module):
    def __init__(self):
        super().__init__()
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
        return self.fc2(x)

# MFCC 13
class CNNModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 6 * 31, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model1 = CNNModel1().to(device)
model1.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'augmented_model1_epoch8.pth'), map_location=device))
model1.eval()

model2 = CNNModel2().to(device)
model2.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'audio_model2.pth'), map_location=device))
model2.eval()

def run_model(model, mfcc_input):
    tensor = torch.tensor(mfcc_input, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        probs = F.softmax(out, dim=1)
        return torch.argmax(probs).item(), probs[0].cpu().numpy()

@api_view(["POST"])
def predict_audio_view(request):
    if "audio_file" not in request.FILES:
        return Response({"error": "No file uploaded."}, status=400)

    file = request.FILES["audio_file"]
    filepath = os.path.join(UPLOAD_DIR, file.name)
    
    with open(filepath, "wb+") as f:
        for chunk in file.chunks():
            f.write(chunk)

    try:
        y, sr = librosa.load(filepath, sr=16000)
        if len(y) < 32000:
            return Response({"error": "Audio too short. Must be at least 2 seconds."}, status=400)

        chunk_size = 32000
        num_chunks = len(y) // chunk_size

        total_probs1 = np.zeros(2)
        total_probs2 = np.zeros(2)
        labels = ["Real", "Fake"]

        for i in range(num_chunks):
            chunk = y[i * chunk_size:(i + 1) * chunk_size]
            mfcc1 = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=40)
            mfcc1 = (mfcc1 - np.mean(mfcc1)) / (np.std(mfcc1) + 1e-6)
            mfcc1 = np.expand_dims(mfcc1, axis=0)

            mfcc2 = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
            mfcc2 = np.expand_dims(mfcc2, axis=0)

            _, probs1 = run_model(model1, mfcc1)
            _, probs2 = run_model(model2, mfcc2)

            total_probs1 += probs1
            total_probs2 += probs2

        avg_probs1 = total_probs1 / num_chunks
        avg_probs2 = total_probs2 / num_chunks

        ensemble_probs = avg_probs1 * 0.7 + avg_probs2 * 0.3 # Weighted average

        pred1 = np.argmax(avg_probs1)
        pred2 = np.argmax(avg_probs2)
        ensemble_pred = np.argmax(ensemble_probs)

        result = {
            "model1": {
                "label": labels[pred1],
                "real_prob": round(float(avg_probs1[0]), 4),
                "fake_prob": round(float(avg_probs1[1]), 4)
            },
            "model2": {
                "label": labels[pred2],
                "real_prob": round(float(avg_probs2[0]), 4),
                "fake_prob": round(float(avg_probs2[1]), 4)
            },
            "ensemble": {
                "label": labels[ensemble_pred],
                "real_prob": round(float(ensemble_probs[0]), 4),
                "fake_prob": round(float(ensemble_probs[1]), 4)
            }
        }

        return Response(result)

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
