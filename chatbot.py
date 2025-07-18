import streamlit as st
import google.generativeai as genai
import os
import speech_recognition as sr

# 🔧 Konfigurasi API Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyAZ7bQA6g6dz3Zn-ZK63KZUkcELOMVZRqo"
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    available = True
except Exception as e:
    available = False
    error_message = str(e)

# ✅ Inisialisasi Session State
def init_session_state():
    defaults = {
        "chat": [
            {"role": "user", "parts": ["Kamu adalah asisten penyuluh pertanian khusus tanaman labu butternut."]}
        ],
        "voice_input": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# 🎙️ Fungsi rekam suara
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Silakan bicara sekarang...")
        audio = recognizer.listen(source, phrase_time_limit=5)
        try:
            text = recognizer.recognize_google(audio, language="id-ID")
            st.success(f"🗣️ Terdeteksi: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Tidak dapat mengenali suara.")
        except sr.RequestError as e:
            st.error(f"Gagal koneksi: {e}")
    return ""

# 💬 UI Chatbot
def render():
    init_session_state()  # penting agar semua state aman

    st.title("💬 Chatbot Smart Garden")

    if not available:
        st.error(f"Chatbot tidak tersedia: {error_message}")
        return

    st.markdown("Tanyakan apa saja seputar budidaya labu butternut!")

    # 📥 Input pengguna (teks atau suara)
    col1, col2 = st.columns([10, 1])

    with col1:
        user_input = st.chat_input("Tulis atau klik 🎙️ di kanan...", key="chatbox_input")

    with col2:
        if st.button("🎙️", help="Klik untuk bicara"):
            st.session_state.voice_input = recognize_speech()

    # Gunakan input dari suara jika teks kosong
    if st.session_state.get("voice_input") and not user_input:
        user_input = st.session_state.voice_input
        st.session_state.voice_input = ""

    # Proses input pengguna
    if user_input:
        st.session_state.chat.append({"role": "user", "parts": [user_input]})
        with st.spinner("Menjawab..."):
            try:
                response = model.generate_content(st.session_state.chat)
                reply = response.text
                st.session_state.chat.append({"role": "model", "parts": [reply]})
            except Exception as e:
                reply = f"Terjadi kesalahan saat menjawab: {str(e)}"
                st.session_state.chat.append({"role": "model", "parts": [reply]})

    # 💬 Tampilkan riwayat chat
    for msg in st.session_state.chat[1:]:
        role = "👩‍🌾 Kamu" if msg["role"] == "user" else "🤖 Asisten"
        with st.chat_message(role):
            st.markdown("\n".join(msg["parts"]))
