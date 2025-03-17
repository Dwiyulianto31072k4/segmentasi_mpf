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
    data.columns = data.columns.str.strip()  # Bersihkan spasi tersembunyi
    
    date_cols = ['FIRST_PPC_DATE', 'FIRST_MPF_DATE', 'LAST_MPF_DATE', 'CONTRACT_ACTIVE_DATE', 'BIRTH_DATE']
    for col in date_cols:
        data[col] = pd.to_datetime(data[col], errors='coerce')
    
    data['Usia'] = 2024 - data['BIRTH_DATE'].dt.year
    data['CUST_NO'] = data['CUST_NO'].astype(str)  # Pastikan CUST_NO bertipe string
    data = data.dropna(subset=['CUST_NO'])  # Hilangkan NaN jika ada
    
    # **Tambahkan fitur Repeat_Customer**
    if 'TOTAL_PRODUCT_MPF' in data.columns:
        data['Repeat_Customer'] = data['TOTAL_PRODUCT_MPF'].apply(lambda x: 1 if x > 1 else 0)
    else:
        data['Repeat_Customer'] = 0  # Default jika kolom tidak ada
    
    return data


# Fungsi untuk clustering dengan K-Means
def perform_clustering(data, n_clusters=4):
    today_date = datetime.datetime(2024, 12, 31)
    data['LAST_MPF_DATE'] = pd.to_datetime(data['LAST_MPF_DATE'], errors='coerce')
    data['Recency'] = (today_date - data['LAST_MPF_DATE']).dt.days
    
    if 'CUST_NO' not in data.columns:
        st.error("Kolom 'CUST_NO' tidak ditemukan dalam dataset!")
        return pd.DataFrame()
    
    rfm = data.groupby('CUST_NO').agg({
        'Recency': 'min',
        'TOTAL_PRODUCT_MPF': 'sum',
        'TOTAL_AMOUNT_MPF': 'sum',
        'Repeat_Customer': 'max',
        'Usia': 'max'
    }).reset_index()
    
    rfm.rename(columns={'TOTAL_PRODUCT_MPF': 'Frequency', 'TOTAL_AMOUNT_MPF': 'Monetary'}, inplace=True)
    rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
    rfm['Monetary_log'] = np.log1p(rfm['Monetary'])
    rfm['Usia_Segment'] = rfm['Usia'].apply(lambda x: 1 if 25 <= x <= 50 else 0)
    
    features = ['Recency', 'Frequency_log', 'Monetary_log', 'Repeat_Customer', 'Usia_Segment']
    rfm_norm = rfm[features].apply(zscore)
    
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_norm)
    
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
    st.write("Hasil Clustering:")
    st.dataframe(clustered_data.head())
    
    plt.figure(figsize=(10,6))
    sns.scatterplot(x=clustered_data['Recency'], y=clustered_data['Monetary_log'], hue=clustered_data['Segmentasi_optimal'], palette='viridis')
    plt.title('Customer Segmentation after Optimization')
    plt.xlabel('Recency (Days)')
    plt.ylabel('Monetary (Log Transformed)')
    plt.legend()
    st.pyplot(plt)
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=clustered_data['Segmentasi_optimal'], y=clustered_data['Monetary_log'])
    plt.title('Distribusi Monetary Log per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Monetary (Log Transformed)')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    
    st.download_button("Download Hasil Clustering", clustered_data.to_csv(index=False), "rfm_segmentasi.csv", "text/csv")
