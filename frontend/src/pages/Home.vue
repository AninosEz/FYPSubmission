<template>
    <div class="layout">
      <header class="main-title">
        <h1>Deepfake Audio Detection</h1>
        <p class="subtitle">Upload or record audio to analyze if it's fake using our ensemble AI model.</p>
      </header>
  
      <main class="main-content">
        <div class="detection-card">
          <div class="tabs">
            <button :class="{ active: activeTab === 'upload' }" @click="switchTab('upload')">Upload</button>
            <button :class="{ active: activeTab === 'record' }" @click="switchTab('record')">Record</button>
          </div>
  
          <div v-if="activeTab === 'upload'" class="tab-content">
            <input type="file" accept=".wav" @change="onUpload" class="file-input" />
            <button class="analyze-btn" @click="submitUpload" :disabled="!uploadFile">Analyze</button>
            <p v-if="shortAudioWarning" class="warning-text">Audio must be at least 2 seconds long.</p>
          </div>
  
          <div v-if="activeTab === 'record'" class="tab-content">
            <button class="record-btn" @click="toggleRecording">
              {{ isRecording ? 'Stop Recording' : 'Start Recording' }}
            </button>
            <audio v-if="audioURL" :src="audioURL" controls class="mt-2"></audio>
            <button class="analyze-btn" @click="submitRecording" :disabled="!audioBlob">Analyze</button>
            <p v-if="shortAudioWarning" class="warning-text">Audio must be at least 2 seconds long.</p>
          </div>
  
          <div v-if="loading" class="loading">
            <p>This model is approximately <strong>90%</strong> accurate.</p>
            <p>Please wait while we analyze your audio...</p>
          </div>
  
          <div v-if="result && !loading" class="result-box">
            <h4>Ensemble Prediction</h4>
            <p>
                <strong>This audio is detected as </strong>
                <span :class="{ 'text-fake': result.ensemble.label === 'Fake', 'text-real': result.ensemble.label === 'Real' }">
                {{ result.ensemble.label }}
                </span>
                (Confident at 
                <span><strong>
                {{ result.ensemble.label === 'Real'
                    ? (result.ensemble.real_prob * 100).toFixed(2)
                    : (result.ensemble.fake_prob * 100).toFixed(2) }}%</strong>)
                </span>
            </p>
          </div>
        </div>
      </main>
      <footer class="footer">
        <small>&copy; 2025 Deepfake or not. All rights reserved. - Anas Hassein</small>
      </footer>
    </div>
  </template>
  
  <script setup lang="ts">
  import { ref } from 'vue'
  
  const activeTab = ref<'upload' | 'record'>('upload')
  const uploadFile = ref<File | null>(null)
  const isRecording = ref(false)
  const mediaRecorder = ref<MediaRecorder | null>(null)
  const audioChunks: Blob[] = []
  const audioBlob = ref<Blob | null>(null)
  const audioURL = ref<string | null>(null)
  const result = ref<any>(null)
  const loading = ref(false)
  const shortAudioWarning = ref(false)
  
  const switchTab = (tab: 'upload' | 'record') => {
    activeTab.value = tab
    result.value = null
    shortAudioWarning.value = false
    audioBlob.value = null
    audioURL.value = null
  }
  
  const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))
  
  const onUpload = (e: Event) => {
    const target = e.target as HTMLInputElement
    if (target.files?.length) {
      uploadFile.value = target.files[0]
    }
  }
  
  const submitUpload = async () => {
    if (!uploadFile.value) return
    shortAudioWarning.value = false
  
    const audio = new Audio(URL.createObjectURL(uploadFile.value))
    audio.onloadedmetadata = async () => {
      if (audio.duration < 2) {
        shortAudioWarning.value = true
        return
      }
  
      loading.value = true
      result.value = null
  
      const formData = new FormData()
      formData.append('audio_file', uploadFile.value!)
      const res = await fetch('http://localhost:8000/api/predict/', {
        method: 'POST',
        body: formData
      })
      const data = await res.json()
      await delay(3000)
      result.value = data
      loading.value = false
    }
  }
  
  const toggleRecording = async () => {
    if (!isRecording.value) {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      mediaRecorder.value = new MediaRecorder(stream)
      audioChunks.length = 0
      mediaRecorder.value.ondataavailable = e => audioChunks.push(e.data)
      mediaRecorder.value.onstop = () => {
        const blob = new Blob(audioChunks, { type: 'audio/wav' })
        audioBlob.value = blob
        audioURL.value = URL.createObjectURL(blob)
      }
      mediaRecorder.value.start()
      isRecording.value = true
    } else {
      mediaRecorder.value?.stop()
      isRecording.value = false
    }
  }
  
  const submitRecording = async () => {
    if (!audioBlob.value) return
    shortAudioWarning.value = false
  
    const tempAudio = new Audio()
    tempAudio.src = URL.createObjectURL(audioBlob.value)
  
    tempAudio.onloadedmetadata = async () => {
      if (tempAudio.duration < 2) {
        shortAudioWarning.value = true
        return
      }
  
      loading.value = true
      result.value = null
  
      const formData = new FormData()
      formData.append('audio_file', audioBlob.value!, 'recorded.wav')
      const res = await fetch('http://localhost:8000/api/predict/', {
        method: 'POST',
        body: formData
      })
  
      const data = await res.json()
      await delay(3000)
      result.value = data
      loading.value = false
    }
  }
  </script>
  
  <style scoped>
  .layout {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    background-color: #0f1117;
    color: #ffffff;
    font-family: 'Segoe UI', sans-serif;
  }
  
  .main-title {
    text-align: center;
    margin: 2rem 0 1rem;
  }
  
  .main-title h1 {
    font-size: 2.2rem;
    font-family: 'Montserrat', sans-serif;
    color: #00ffc8;
  }
  
  .subtitle {
    color: #cccccc;
    font-size: 1rem;
  }
  
  .main-content {
    flex: 1;
    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding: 1rem;
  }
  
  .detection-card {
    background-color: #1f2229;
    padding: 2rem;
    border-radius: 12px;
    max-width: 500px;
    width: 100%;
    box-shadow: 0 0 15px rgba(0, 0, 0, 0.3);
    text-align: left;
  }
  
  .tabs {
    display: flex;
    margin-bottom: 1rem;
  }
  
  .tabs button {
    flex: 1;
    padding: 0.75rem;
    background-color: #2b2d35;
    color: #fff;
    border: none;
    cursor: pointer;
  }
  
  .tabs button.active {
    background-color: #3dd68c;
    color: #000;
    font-weight: bold;
  }
  
  .tab-content {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  
  .file-input {
    background-color: #2c2c2c;
    color: #fff;
    border: 1px solid #555;
    padding: 0.5rem;
  }
  
  .analyze-btn,
  .record-btn {
    background-color: #3dd68c;
    color: #000;
    font-weight: bold;
    border: none;
    padding: 0.5rem;
    cursor: pointer;
  }
  
  .analyze-btn:disabled {
    background-color: #999;
    color: #eee;
    cursor: not-allowed;
  }
  
  .warning-text {
    color: #ff4d4d;
    font-size: 0.85rem;
    margin-top: -0.5rem;
  }
  
  .loading {
    margin-top: 1rem;
    font-style: italic;
    color: #dcdcdc;
  }
  
  .result-box {
    margin-top: 1.5rem;
    background-color: #2a2d35;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #444;
    color: #f0f0f0;
  }
  
  .result-box h4 {
    color: #ffffff;
    margin-bottom: 0.5rem;
  }
  
  .text-fake {
    color: #ff4d4d;
    font-weight: 600;
  }
  
  .text-real {
    color: #00e676;
    font-weight: 600;
  }
  
  .footer {
  position: fixed;
  bottom: 0;
  left: 0;
  width: 100%;
  background-color: #002b36;
  color: #cceeff;
  text-align: center;
  padding: 1rem 0;
  z-index: 100;
}

  </style>
  