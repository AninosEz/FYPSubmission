# Audio Deepfake Detection App

This full-stack application detects deepfake audio using AI.
It is developed using Django for the backend, and Vue for the frontend for a lightweight and responsive website. This script does not have an .exe file, and needs to be hosted locally for testing.

All the training scripts for the used models are in the `model_training_files/` folder.

## Installation and Setup

Follow these steps to install and run the application:

```bash
pip install -r requirements.txt
```

```bash
cd frontend
npm install
```

```bash
npm run build
```

```bash
cd ../
python manage.py migrate
python manage.py runserver
```

The server will start, accessible at `http://127.0.0.1:8000/`.
