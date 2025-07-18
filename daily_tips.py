import streamlit as st
import random
import requests
from datetime import datetime, timedelta

# === Konfigurasi API Cuaca ===
OPENWEATHER_API_KEY = "51480f5c01b461807993bc493ca85878"

# Daftar nama negara
DAFTAR_NEGARA = {
    "Afrika Selatan": "ZA",
    "Amerika Serikat": "US",
    "Arab Saudi": "SA",
    "Argentina": "AR",
    "Australia": "AU",
    "Austria": "AT",
    "Belanda": "NL",
    "Belarus": "BY",
    "Belgia": "BE",
    "Bolivia": "BO",
    "Brasil": "BR",
    "Brunei": "BN",
    "Ceko": "CZ",
    "Chili": "CL", 
    "Denmark": "DK",
    "Ekuador": "EC",
    "Filipina": "PH",
    "Finlandia": "FI",
    "India": "IN",
    "Indonesia": "ID",
    "Inggris": "GB",
    "Irak": "IQ",
    "Iran": "IR",
    "Irlandia": "IE",
    "Italia": "IT",
    "Jepang": "JP",
    "Kamboja": "KH",
    "Kanada": "CA",
    "Kazakstan": "KZ",
    "Kolombia": "CO",
    "Korea Selatan": "KR",
    "Kuba": "CU",
    "Laos": "LA",
    "Malaysia": "MY",
    "Meksiko": "MX",
    "Mesir": "EG",
    "Myanmar": "MM",
    "Nigeria": "NG",
    "Norwegia": "NO",
    "Pakistan": "PK",
    "Paraguay": "PY",
    "Peru": "PE",
    "Polandia": "PL",
    "Portugal": "PT",
    "Prancis": "FR",
    "Rumania": "RO",
    "Rusia": "RU",
    "Singapura": "SG",
    "Spanyol": "ES",
    "Swedia": "SE",
    "Swiss": "CH",
    "Thailand": "TH",
    "Timor Leste": "TL",
    "Tiongkok": "CN",
    "Turki": "TR",
    "Ukraina": "UA",
    "Uruguay": "UY",
    "Uzbekistan": "UZ",
    "Venezuela": "VE",
    "Vietnam": "VN",
    "Yunani": "GR"
}

# Fungsi estimasi kelembaban tanah dari cuaca
def estimasi_kelembaban_tanah(suhu, kondisi, kelembaban_udara):
    base = 50
    if suhu is not None:
        base -= max(0, suhu - 30) * 1.5
    if kelembaban_udara is not None:
        base += (kelembaban_udara - 60) * 0.2
    if kondisi.lower() == "rain":
        base += 10
    elif kondisi.lower() == "clear":
        base -= 5
    return int(max(10, min(100, base)))

# Ambil cuaca dari API
def ambil_data_cuaca(kota, kode_negara):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={kota},{kode_negara}&appid={OPENWEATHER_API_KEY}&units=metric"
        res = requests.get(url)
        data = res.json()

        if res.status_code != 200 or 'main' not in data or 'weather' not in data:
            st.warning("❌ Kota atau negara tidak ditemukan. Coba periksa ejaan kota atau pilih negara yang sesuai.")
            return None

        suhu = data['main']['temp']
        kondisi = data['weather'][0]['main']
        kelembaban_udara = data['main']['humidity']
        timezone_offset = data['timezone']
        waktu_lokal = datetime.utcnow() + timedelta(seconds=timezone_offset)
        zona = get_timezone_label(timezone_offset)
        kelembaban_tanah = estimasi_kelembaban_tanah(suhu, kondisi, kelembaban_udara)

        return {
            "suhu": suhu,
            "cuaca": kondisi,
            "kelembaban_udara": kelembaban_udara,
            "kelembaban_tanah": kelembaban_tanah,
            "waktu_lokal": waktu_lokal,
            "zona": zona
        }

    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan sistem: {e}")
        return None

# Konversi offset zona waktu ke label

def get_timezone_label(offset_seconds):
    offset_hours = offset_seconds // 3600
    if offset_hours == 7:
        return "WIB"
    elif offset_hours == 8:
        return "WITA"
    elif offset_hours == 9:
        return "WIT"
    elif offset_hours == 0:
        return "UTC"
    elif offset_hours > 0:
        return f"UTC+{offset_hours}"
    else:
        return f"UTC{offset_hours}"

# Tips umum

tips_umum = [
    "Gunakan pupuk kompos alami setiap 2 minggu untuk pertumbuhan optimal.",
    "Panen buah saat kulit berubah coklat kekuningan dan tangkai mulai mengering.",
    "Pastikan labu butternut mendapat sinar matahari langsung minimal 6 jam per hari.",
]

# Tips berbasis data

def get_tips_berbasis_data(cuaca):
    tips = []
    suhu = cuaca['suhu']
    kondisi = cuaca['cuaca']
    kelembaban_tanah = cuaca['kelembaban_tanah']

    if suhu > 32:
        tips.append("Suhu terlalu tinggi. Gunakan naungan & siram pagi/sore.")
    elif suhu < 20:
        tips.append("Suhu rendah. Tutupi tanaman malam hari untuk menjaga suhu.")

    if kelembaban_tanah < 30:
        tips.append("Tanah terlalu kering. Segera lakukan penyiraman.")
    elif kelembaban_tanah > 70:
        tips.append("Tanah terlalu lembab. Kurangi frekuensi penyiraman.")

    if kondisi.lower() in ["rain", "clouds"]:
        tips.append("Cuaca mendung/hujan. Waspadai penyakit jamur dan overwatering.")
    elif kondisi.lower() == "clear":
        tips.append("Cuaca cerah! Waktu ideal untuk pemupukan dan penyiangan.")

    return tips

# === Streamlit UI ===
def render():
    st.title("🌱 Tips Harian Budidaya Labu Butternut")
    st.markdown("Disesuaikan dengan cuaca dan kondisi tanah harian secara otomatis.")

    negara_options = ["-- Pilih Negara --"] + list(DAFTAR_NEGARA.keys())
    default_index = 0  # Tidak langsung pilih negara mana pun

    with st.form("lokasi_form"):
        st.subheader("📍 Lokasi Tanam")
        negara_nama = st.selectbox("Pilih Negara", negara_options, index=default_index)
        kota = st.text_input("Nama Kota", value=st.session_state.get("kota"))
        submit = st.form_submit_button("Perbarui Lokasi")

    if submit:
        if negara_nama == "-- Pilih Negara --" or not kota.strip():
            st.warning("⚠️ Silakan isi kota dan pilih negara terlebih dahulu.")
            return
        st.session_state["negara"] = negara_nama
        st.session_state["kota"] = kota
        st.success(f"📍 Lokasi diperbarui: {kota}, {negara_nama}")

    # Cek apakah lokasi sudah dipilih
    if "negara" not in st.session_state or "kota" not in st.session_state:
        st.info("💡 Silakan pilih lokasi terlebih dahulu.")
        return

    kota_aktif = st.session_state["kota"]
    negara_nama_aktif = st.session_state["negara"]
    kode_negara_aktif = DAFTAR_NEGARA.get(negara_nama_aktif)

    if not kode_negara_aktif:
        st.warning("⚠️ Negara tidak valid. Silakan pilih ulang.")
        return

    cuaca_data = ambil_data_cuaca(kota_aktif, kode_negara_aktif)
    if not cuaca_data:
        return

    st.subheader("📊 Data Hari Ini")
    st.write(f"- Lokasi: {kota_aktif}, {negara_nama_aktif}")
    waktu_format = cuaca_data['waktu_lokal'].strftime('%d %B %Y, %H:%M')
    st.write(f"- 🕒 Waktu Lokal: {waktu_format} {cuaca_data['zona']}")
    st.write(f"- Suhu Udara: {cuaca_data['suhu']}°C")
    st.write(f"- Kelembaban Udara: {cuaca_data['kelembaban_udara']}%")
    st.write(f"- Kondisi Cuaca: {cuaca_data['cuaca']}")
    st.write(f"- Kelembaban Tanah (Estimasi): {cuaca_data['kelembaban_tanah']}%")

    tips_data = get_tips_berbasis_data(cuaca_data)
    tip_umum = random.choice(tips_umum)

    if tips_data:
        st.subheader("💡 Tips Berbasis Data")
        for tip in tips_data:
            st.success(f"- {tip}")
    else:
        st.success("✅ Kondisi lingkungan optimal hari ini!")

    st.info(f"📝 Tip Umum: {tip_umum}")
    st.caption(f"Tips harian untuk {datetime.now().strftime('%d %B %Y')}")

