import streamlit as st
from utils.auth import is_authenticated, login_user, logout_user, register_user, reset_password
from utils.image_classifier import classify_image
import dashboard
import chatbot
import daily_tips
import os

# Konfigurasi halaman
st.set_page_config(page_title="Smart Garden Web Assistant", layout="wide")

# Cek session state untuk routing halaman
if 'page' not in st.session_state:
    st.session_state.page = 'Login'

# Tampilkan halaman login/register jika belum terautentikasi
if not is_authenticated():
    if st.session_state.page == 'Login':
        login_user()
    elif st.session_state.page == 'Register':
        register_user()
    elif st.session_state.page == 'Forgot':
        reset_password()

    st.markdown("---")
    # Tombol navigasi di bawah form
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 2])
    if col2.button("Login", key="nav_login", use_container_width=True):
        st.session_state.page = 'Login'
        st.rerun()
    if col3.button("Daftar", key="nav_register", use_container_width=True):
        st.session_state.page = 'Register'
        st.rerun()
    if col4.button("Lupa kata sandi", key="nav_forgot", use_container_width=True):
        st.session_state.page = 'Forgot'
        st.rerun()

else:
    # Mengambil username dari session state untuk ditampilkan sebagai judul
    if 'username' in st.session_state:
        username = st.session_state['username'].capitalize()
        st.sidebar.markdown(f"## 👋 Selamat Datang, **{username}**")
    else:
        st.sidebar.title("Smart Garden Web Assistant") # Fallback

    # Menu navigasi
    menu = st.sidebar.radio(
        "Pilih Halaman",
        ["Dashboard", "Klasifikasi Gambar", "Chatbot", "Tips Harian"]
    )
    
    st.sidebar.markdown("---")
    # Tombol logout di bagian bawah sidebar
    if st.sidebar.button("Keluar", use_container_width=True):
        logout_user()

    # Routing halaman utama
    if menu == "Dashboard":
        dashboard.render()
    elif menu == "Klasifikasi Gambar":
        st.title("📷 Klasifikasi Kematangan Labu Butternut")
        uploaded = st.file_uploader("Upload gambar labu butternut", type=["jpg", "png", "jpeg"])
        if uploaded:
            result = classify_image(uploaded)
            st.image(uploaded, width=450)
            if result['status'] == 'ok':
                st.success(f"✅ Status: {result['label']} (Confidence: {result['confidence'] * 100:.0f}%)")
                st.info(f"ℹ️ Keterangan: {result['description']}")
                st.warning(f"🛠️ Solusi: {result['solution']}")
            else:
                st.error(f" {result['message']}")
    elif menu == "Chatbot":
        chatbot.render()
    elif menu == "Tips Harian":
        daily_tips.render()