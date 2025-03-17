import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import datetime

# Set default style
sns.set_style("whitegrid")

# Fungsi untuk memuat dan membersihkan data
def load_data(file):
    data = pd.read_excel(file)
    data.columns = data.columns.str.strip()  # Bersihkan spasi tersembunyi
    
    date_cols = ['FIRST_PPC_DATE', 'FIRST_MPF_DATE', 'LAST_MPF_DATE', 'CONTRACT_ACTIVE_DATE', 'BIRTH_DATE']
    for col in date_cols:
        data[col] = pd.to_datetime(data[col], errors='coerce')
    
    # Tangani nilai NaN di BIRTH_DATE
    data['Usia'] = 2024 - data['BIRTH_DATE'].dt.year.fillna(0).astype(int)
    
    # Pastikan CUST_NO tidak kosong dan bertipe string
    data['CUST_NO'] = data['CUST_NO'].astype(str).fillna("Unknown")
    data = data.dropna(subset=['CUST_NO'])  

    # Tambahkan Repeat_Customer
    data['Repeat_Customer'] = data['TOTAL_PRODUCT_MPF'].apply(lambda x: 1 if x > 1 else 0)
    
    return data

# Fungsi untuk clustering dengan K-Means
def perform_clustering(data, n_clusters=4):
    today_date = datetime.datetime(2024, 12, 31)
    data['LAST_MPF_DATE'] = pd.to_datetime(data['LAST_MPF_DATE'], errors='coerce')

    # Isi nilai NaN dengan tanggal lama agar tidak menyebabkan NaN di Recency
    data['Recency'] = (today_date - data['LAST_MPF_DATE']).dt.days.fillna(9999).astype(int)
    
    # Pastikan semua kolom yang dibutuhkan ada
    required_columns = ['CUST_NO', 'LAST_MPF_DATE', 'TOTAL_PRODUCT_MPF', 'TOTAL_AMOUNT_MPF', 'Repeat_Customer', 'Usia']
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        st.warning(f"Kolom berikut tidak ditemukan dalam dataset: {missing_columns}")
        return pd.DataFrame()
    
    rfm = data.groupby('CUST_NO').agg({
        'Recency': 'min',
        'TOTAL_PRODUCT_MPF': 'sum',
        'TOTAL_AMOUNT_MPF': 'sum',
        'Repeat_Customer': 'max',
        'Usia': 'max'
    }).reset_index()
    
    rfm.rename(columns={'TOTAL_PRODUCT_MPF': 'Frequency', 'TOTAL_AMOUNT_MPF': 'Monetary'}, inplace=True)
    
    # Isi NaN sebelum log transform
    rfm['Frequency'].fillna(0, inplace=True)
    rfm['Monetary'].fillna(0, inplace=True)

    # Transformasi log
    rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
    rfm['Monetary_log'] = np.log1p(rfm['Monetary'])
    
    # Segmentasi usia
    rfm['Usia_Segment'] = rfm['Usia'].apply(lambda x: 1 if 25 <= x <= 50 else 0)

    # Pilih fitur untuk normalisasi
    features = ['Recency', 'Frequency_log', 'Monetary_log', 'Repeat_Customer', 'Usia_Segment']

    # **Hapus kolom dengan variansi nol**
    zero_variance_cols = [col for col in features if rfm[col].std() == 0]
    if zero_variance_cols:
        st.warning(f"Kolom dengan variansi nol ditemukan: {zero_variance_cols}. Kolom ini akan dihapus dari clustering.")
        features = [col for col in features if col not in zero_variance_cols]

    # **Gunakan StandardScaler sebagai pengganti zscore**
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[features])

    # **Konversi kembali ke DataFrame**
    rfm_norm = pd.DataFrame(rfm_scaled, columns=features, index=rfm.index)

    # **Pastikan tidak ada NaN setelah normalisasi**
    st.write("Jumlah NaN setelah normalisasi:", rfm_norm.isnull().sum())

    # **Hapus baris yang mengandung NaN sebelum K-Means**
    rfm_norm = rfm_norm.dropna()
    rfm = rfm.loc[rfm_norm.index]

    # **Tambahkan Cek Sebelum K-Means**
    if rfm_norm.isnull().values.any():
        st.error("Data masih memiliki NaN sebelum K-Means! Periksa dataset.")
        return pd.DataFrame()

    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_norm)
    
    # Mapping segmentasi
    segment_map_optimal = {
        0: "Potential Loyalists",
        1: "Responsive Customers",
        2: "Occasional Buyers",
        3: "Hibernating Customers"
    }
    
    invite_map_optimal = {
        "Potential Loyalists": "✅ Diundang",
        "Occasional Buyers": "✅ Diundang",
        "Responsive Customers": "❌ Tidak",
        "Hibernating Customers": "❌ Tidak"
    }
    
    rfm['Segmentasi_optimal'] = rfm['Cluster'].map(segment_map_optimal)
    rfm['Layak_Diundang_optimal'] = rfm['Segmentasi_optimal'].map(invite_map_optimal)
    
    return rfm

# STREAMLIT UI
st.title("Analisis Data Pelanggan SPEKTRA")

uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])
if uploaded_file is not None:
    st.success("File berhasil diunggah!")
    data = load_data(uploaded_file)
    st.write("### Data Awal")
    st.dataframe(data.head())
    
    st.write("### Clustering Pelanggan")
    num_clusters = st.slider("Pilih jumlah cluster:", min_value=2, max_value=10, value=4)
    clustered_data = perform_clustering(data, num_clusters)
    
    if not clustered_data.empty:
        st.write("Hasil Clustering:")
        st.dataframe(clustered_data.head())
        
        # Scatter plot
        plt.figure(figsize=(10,6))
        sns.scatterplot(x=clustered_data['Recency'], y=clustered_data['Monetary_log'], 
                        hue=clustered_data['Segmentasi_optimal'], size=clustered_data['Frequency_log'], 
                        sizes=(10, 200), alpha=0.8, palette='viridis')
        plt.title('Customer Segmentation after Optimization')
        plt.xlabel('Recency (Days)')
        plt.ylabel('Monetary (Log Transformed)')
        plt.legend()
        st.pyplot(plt)
        
        # Download button
        csv_data = clustered_data.to_csv(index=False, encoding='utf-8-sig')
        st.download_button("Download Hasil Clustering", csv_data, "rfm_segmentasi.csv", "text/csv")
