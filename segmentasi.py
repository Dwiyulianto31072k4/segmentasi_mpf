import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import datetime

# Set default style
sns.set_style("whitegrid")

# Fungsi untuk memuat dan membersihkan data
def load_data(file):
    data = pd.read_excel(file)
    # Konversi kolom tanggal
    date_cols = ['FIRST_PPC_DATE', 'FIRST_MPF_DATE', 'LAST_MPF_DATE', 'CONTRACT_ACTIVE_DATE', 'BIRTH_DATE']
    for col in date_cols:
        data[col] = pd.to_datetime(data[col], errors='coerce')
    # Hitung usia pelanggan
    data['Usia'] = 2024 - data['BIRTH_DATE'].dt.year
    return data

# Fungsi untuk clustering dengan K-Means berdasarkan logika pengolahan data
def perform_clustering(data, n_clusters=4):
    today_date = datetime.datetime(2024, 12, 31)
    data['LAST_MPF_DATE'] = pd.to_datetime(data['LAST_MPF_DATE'])
    data['Recency'] = (today_date - data['LAST_MPF_DATE']).dt.days
    
    rfm = data.groupby('CUST_NO').agg({
        'Recency': 'min',
        'TOTAL_PRODUCT_MPF': 'sum',
        'TOTAL_AMOUNT_MPF': 'sum',
        'Repeat_Customer': 'max',
        'Usia': 'max'
    }).reset_index()
    
    rfm.rename(columns={'TOTAL_PRODUCT_MPF': 'Frequency', 'TOTAL_AMOUNT_MPF': 'Monetary'}, inplace=True)
    
    # Log Transformation untuk mengurangi skewness
    rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
    rfm['Monetary_log'] = np.log1p(rfm['Monetary'])
    
    # Tambahkan fitur apakah pelanggan termasuk dalam range usia 25-45
    rfm['Usia_Segment'] = rfm['Usia'].apply(lambda x: 1 if 25 <= x <= 45 else 0)
    
    # Normalisasi data dengan Z-score untuk clustering
    features = ['Recency', 'Frequency_log', 'Monetary_log', 'Repeat_Customer', 'Usia_Segment']
    rfm_norm = rfm[features].apply(zscore)
    
    # K-Means++ Clustering dengan cluster baru yang lebih optimal
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_norm)
    
    # Mapping segmen baru berdasarkan hasil backtest
    segment_map_optimal = {
        0: "Potential Loyalists",
        1: "Responsive Customers",
        2: "Occasional Buyers",
        3: "Hibernating Customers"
    }
    
    invite_map_optimal = {
        0: "✅ Diundang",
        1: "✅ Diundang",
        2: "❌ Tidak",
        3: "❌ Tidak"
    }
    
    rfm['Segmentasi_optimal'] = rfm['Cluster'].map(segment_map_optimal)
    rfm['Layak_Diundang_optimal'] = rfm['Cluster'].map(invite_map_optimal)
    
    return rfm

# STREAMLIT UI
st.title("Analisis Data Pelanggan SPEKTRA")

# Upload file
uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])
if uploaded_file is not None:
    st.success("File berhasil diunggah!")
    data = load_data(uploaded_file)
    st.write("### Data Awal")
    st.dataframe(data.head())

    # Clustering
    st.write("### Clustering Pelanggan")
    num_clusters = st.slider("Pilih jumlah cluster:", min_value=2, max_value=10, value=4)
    clustered_data = perform_clustering(data, num_clusters)
    st.write("Hasil Clustering:")
    st.dataframe(clustered_data.head())

    # Visualisasi hasil clustering
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=clustered_data['Recency'], y=clustered_data['Monetary_log'], hue=clustered_data['Segmentasi_optimal'], palette='viridis')
    plt.title('Customer Segmentation after Optimization')
    plt.xlabel('Recency (Days)')
    plt.ylabel('Monetary (Log Transformed)')
    plt.legend()
    st.pyplot(plt)
    
    # Download hasil clustering
    st.download_button("Download Hasil Clustering", clustered_data.to_csv(index=False), "rfm_segmentasi.csv", "text/csv")
