<template>
  <div class="app-layout">
    <div v-if="showDisclaimer" class="disclaimer-backdrop">
      <div class="disclaimer-card">
        <h5 class="disclaimer-title">Disclaimer</h5>
        <p>
          This web app does <strong>not</strong> save or share your uploaded or recorded audio files.<br /><br />
          All audio is processed temporarily and automatically deleted after prediction.<br /><br />
          By continuing to use this app, you confirm that you understand and accept this policy.
        </p>
        <button class="btn btn-primary" @click="acceptDisclaimer">I Understand</button>
      </div>
    </div>

    <main class="main-content">
      <router-view />
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

const showDisclaimer = ref(false)

onMounted(() => {
  const expiry = localStorage.getItem('disclaimerExpiry')
  const now = Date.now()
  if (!expiry || now > parseInt(expiry)) {
    showDisclaimer.value = true
  }
})

const acceptDisclaimer = () => {
  const oneHourFromNow = Date.now() + 60 * 60 * 1000
  localStorage.setItem('disclaimerExpiry', oneHourFromNow.toString())
  showDisclaimer.value = false
}
</script>

<style scoped>
.app-layout {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background-color: #0f1117;
  color: #f1f1f1;
  font-family: 'Inter', sans-serif;
}

.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 1rem;
}

.disclaimer-backdrop {
  position: fixed;
  inset: 0;
  background-color: rgba(0, 0, 0, 0.75);
  z-index: 1000;
  display: flex;
  justify-content: center;
  align-items: center;
}

.disclaimer-card {
  background-color: #1c1f26;
  color: #f1f1f1;
  padding: 2rem;
  border-radius: 12px;
  width: 90%;
  max-width: 420px;
  text-align: center;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
}

.disclaimer-title {
  margin-bottom: 1rem;
  font-size: 1.5rem;
  color: #00ffc8;
}

.btn-primary {
  background-color: #00ffc8;
  color: #000;
  border: none;
  padding: 0.6rem 1.2rem;
  border-radius: 6px;
  font-weight: bold;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

.btn-primary:hover {
  background-color: #00ddb4;
}
</style>
