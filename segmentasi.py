def perform_clustering(data, n_clusters=4):
    today_date = datetime.datetime(2024, 12, 31)
    data['LAST_MPF_DATE'] = pd.to_datetime(data['LAST_MPF_DATE'], errors='coerce')

    # Calculate Recency
    data['Recency'] = (today_date - data['LAST_MPF_DATE']).dt.days.fillna(9999).astype(int)
    
    # Ensure required columns exist
    required_columns = ['CUST_NO', 'LAST_MPF_DATE', 'TOTAL_PRODUCT_MPF', 'TOTAL_AMOUNT_MPF', 'Repeat_Customer', 'Usia']
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        st.warning(f"Kolom berikut tidak ditemukan dalam dataset: {missing_columns}")
        return pd.DataFrame()
    
    # Aggregate data by customer
    rfm = data.groupby('CUST_NO').agg({
        'Recency': 'min',
        'TOTAL_PRODUCT_MPF': 'sum',
        'TOTAL_AMOUNT_MPF': 'sum',
        'Repeat_Customer': 'max',
        'Usia': 'max'
    }).reset_index()
    
    rfm.rename(columns={'TOTAL_PRODUCT_MPF': 'Frequency', 'TOTAL_AMOUNT_MPF': 'Monetary'}, inplace=True)
    
    # Handle NaN values in Frequency and Monetary
    rfm['Frequency'].fillna(0, inplace=True)
    rfm['Monetary'].fillna(0, inplace=True)

    # Log transformation (add 1 to avoid log(0))
    rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
    rfm['Monetary_log'] = np.log1p(rfm['Monetary'])
    
    # Segmentasi usia
    rfm['Usia_Segment'] = rfm['Usia'].apply(lambda x: 1 if 25 <= x <= 50 else 0)

    # Select features for clustering
    features = ['Recency', 'Frequency_log', 'Monetary_log', 'Repeat_Customer', 'Usia_Segment']

    # Check for zero variance columns
    zero_variance_cols = [col for col in features if rfm[col].std() == 0]
    if zero_variance_cols:
        st.warning(f"Kolom dengan variansi nol ditemukan: {zero_variance_cols}. Kolom ini akan dihapus dari clustering.")
        features = [col for col in features if col not in zero_variance_cols]

    # Standardize features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[features])

    # Convert back to DataFrame
    rfm_norm = pd.DataFrame(rfm_scaled, columns=features, index=rfm.index)

    # Check for NaN after normalization
    if rfm_norm.isnull().values.any():
        st.error("Data masih memiliki NaN setelah normalisasi! Periksa dataset.")
        st.dataframe(rfm_norm[rfm_norm.isnull().any(axis=1)])  # Show rows with NaN
        return pd.DataFrame()

    # Drop rows with NaN (if any remain)
    rfm_norm = rfm_norm.dropna()
    rfm = rfm.loc[rfm_norm.index]

    # Perform K-Means clustering
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
