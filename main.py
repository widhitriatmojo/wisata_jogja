import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import streamlit as st

# Inisialisasi stopwords dan stemmer
clean_spcl = re.compile('[/(){}\\[\\]\\|@,;]')

clean_symbol = re.compile('[^0-9a-z #+_]')
factory = StopWordRemoverFactory()
stopworda = factory.get_stop_words()
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = text.lower()  # Ubah menjadi huruf kecil
    text = clean_spcl.sub(' ', text)  # Hapus karakter khusus
    text = clean_symbol.sub('', text)  # Hapus simbol lainnya
    text = ' '.join(word for word in text.split() if word not in stopworda)  # Hapus stopwords
    text = stemmer.stem(text)  # Stemming
    return text

# Load dataset wisata
wisata_df = pd.read_csv("tour.csv")

# Bersihkan nama tempat wisata
wisata_df['Place_Name'] = wisata_df['Place_Name'].apply(clean_text)
wisata_df.reset_index(inplace=True)

# Set index pada nama tempat wisata
wisata_df.set_index('Place_Name', inplace=True)

# Inisialisasi TF-IDF Vectorizer
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=1)

# Membuat matriks TF-IDF
tfidf_matrix = tf.fit_transform(wisata_df.index)

# Hitung cosine similarity
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Membuat series untuk indeks nama wisata
indices = pd.Series(wisata_df.index)

# Fungsi untuk rekomendasi tempat wisata
def recommendations(name, top=10):
    recommended_wisata = []

    # Bersihkan input pengguna
    cleaned_name = clean_text(name)

    # Pisahkan input pengguna menjadi token (kata)
    input_tokens = set(cleaned_name.split())

    # Filter tempat wisata berdasarkan apakah ada token yang cocok
    filtered_indices = []
    for idx in wisata_df.index:
        if any(token in clean_text(idx) for token in input_tokens):
            filtered_indices.append(idx)

    if not filtered_indices:  # Jika tidak ada hasil yang cocok
        return [f"Tidak ada wisata yang cocok dengan kata kunci '{name}'"]

    # Ambil indeks numerik dari data yang difilter
    filtered_df = wisata_df.loc[filtered_indices]
    filtered_numeric_indices = [list(wisata_df.index).index(i) for i in filtered_indices]

    # Ulangi perhitungan cosine similarity hanya pada data yang terfilter
    tfidf_matrix_filtered = tfidf_matrix[filtered_numeric_indices]
    cos_sim_filtered = cosine_similarity(tfidf_matrix_filtered, tfidf_matrix_filtered)

    # Ambil indeks pertama dari data yang difilter
    idx = filtered_numeric_indices[0]  # Ambil indeks pertama dari input pengguna

    # Hitung skor kesamaan dan ambil 'top' rekomendasi teratas
    score_series = pd.Series(cos_sim_filtered[0]).sort_values(ascending=False)

    # Ambil indeks 'top' rekomendasi teratas
    top_indexes = list(score_series.iloc[1:top+1].index)

    # Pastikan input yang dimasukkan tetap berada di posisi pertama
    # Temukan posisi input pengguna di filtered_indices
    input_position = next((i for i, idx in enumerate(filtered_indices) if clean_text(idx) == cleaned_name), None)
    
    if input_position is not None:
        top_indexes.insert(0, input_position)  # Menambahkan input favorit ke urutan pertama

    # Hasil akhir dari data yang terfilter
    recommended_wisata = [filtered_indices[i] for i in top_indexes]

    return recommended_wisata[:top]


st.title("Sistem Rekomendasi Wisata")

# Input pengguna untuk nama wisata favorit
place_input = st.text_input("Masukkan nama wisata favorit Anda:")

if place_input:
    st.write("Mencari rekomendasi...")

    # Menampilkan hasil rekomendasi
    hasil_rekomendasi = recommendations(place_input, top=5)
    st.write("Rekomendasi Wisata untuk Anda:")
    for wisata in hasil_rekomendasi:
        st.write(wisata)

# Menampilkan dataframe
df = pd.read_csv("tour.csv")
st.dataframe(df)
