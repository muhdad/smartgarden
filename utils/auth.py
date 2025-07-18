import streamlit as st
import json
import os
import hashlib

DB_PATH = "data/users.json"

# Muat data pengguna dari file
def load_users():
    if not os.path.exists(DB_PATH):
        with open(DB_PATH, 'w') as f:
            json.dump({}, f)
    with open(DB_PATH, 'r') as f:
        return json.load(f)

# Simpan data pengguna ke file
def save_users(users):
    with open(DB_PATH, 'w') as f:
        json.dump(users, f, indent=4)

# Enkripsi password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Cek apakah sudah login
def is_authenticated():
    return 'user' in st.session_state

# Login pengguna
def login_user():
    st.title("🔐 Login Pengguna")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        users = load_users()
        if username in users and users[username]['password'] == hash_password(password):
            st.success("✅ Login berhasil!")
            st.session_state.user = username
            st.session_state["username"] = username  # untuk keperluan profil
            st.rerun()
        else:
            st.error("❌ Username atau password salah.")

# Registrasi pengguna baru
def register_user():
    st.title("📝 Daftar Akun Baru")
    username = st.text_input("Buat Username")
    password = st.text_input("Buat Password", type="password")
    confirm = st.text_input("Konfirmasi Password", type="password")

    if st.button("Daftar"):
        if password != confirm:
            st.error("❌ Password tidak cocok.")
            return
        users = load_users()
        if username in users:
            st.error("❌ Username sudah terdaftar.")
        else:
            users[username] = {"password": hash_password(password)}
            save_users(users)
            st.success("✅ Pendaftaran berhasil. Silakan login.")
            st.session_state.page = 'Login'
            st.rerun()

# Reset password
def reset_password():
    st.title("🔄 Lupa Password")
    username = st.text_input("Masukkan Username Anda")
    new_pass = st.text_input("Password Baru", type="password")
    confirm = st.text_input("Konfirmasi Password Baru", type="password")

    if st.button("Reset Password"):
        users = load_users()
        if username not in users:
            st.error("❌ Username tidak ditemukan.")
        elif new_pass != confirm:
            st.error("❌ Password tidak cocok.")
        else:
            users[username]['password'] = hash_password(new_pass)
            save_users(users)
            st.success("✅ Password berhasil diubah. Silakan login kembali.")
            st.session_state.page = 'Login'
            st.rerun()

# Logout pengguna
def logout_user():
    st.session_state.clear()
    st.success("✅ Berhasil logout.")
    st.session_state.page = 'Login'
    st.rerun()
