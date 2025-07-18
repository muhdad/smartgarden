import streamlit as st
from utils.image_classifier import label_info

def render():
    """Merender halaman Dashboard."""

    st.title("📊 Dashboard Informasi Tingkat Kematangan Labu Butternut")
    st.markdown("---")
    st.markdown("Selamat datang! Pelajari perbedaan antara labu yang belum matang, setengah matang, dan matang untuk memaksimalkan hasil panen Anda.")

    # PERUBAHAN: Path gambar sekarang menunjuk ke file lokal di folder 'assets'
    image_paths = {
        "belum_matang": "assets/belummatang.jpg",
        "setengah_matang": "assets/setengahmatang.jpg",
        "matang": "assets/matang.jpg"
    }

    # Membuat 3 kolom untuk setiap tingkat kematangan
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.subheader("belum_matang")
        st.image(image_paths["belum_matang"], caption="Contoh labu yang belum matang", width=300)
        st.info(f"**Keterangan:** {label_info['belum_matang']['description']}")

    with col2:
        st.subheader("setengah_matang")
        st.image(image_paths["setengah_matang"], caption="Contoh labu yang setengah matang", width=300)
        st.info(f"**Keterangan:** {label_info['setengah_matang']['description']}")

    with col3:
        st.subheader("matang")
        st.image(image_paths["matang"], caption="Contoh labu yang sudah matang", width=300)
        st.info(f"**Keterangan:** {label_info['matang']['description']}")