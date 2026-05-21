# dashboard.py
# Dashboard Analisis Sentimen Kesehatan Mental
# Untuk UAS Rekayasa Data - Kelompok 11

import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import warnings
warnings.filterwarnings('ignore')

# ============ KONFIGURASI HALAMAN ============
st.set_page_config(
    page_title="Analisis Sentimen Kesehatan Mental",
    page_icon="🧠",
    layout="wide"
)

# ============ HEADER ============
st.markdown("""
<div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 2rem; border-radius: 20px; color: white; text-align: center; margin-bottom: 2rem;">
    <h1 style="margin: 0;">🧠 Analisis Sentimen Kesehatan Mental</h1>
    <p style="margin: 0.5rem 0;">Perbandingan Algoritma Naive Bayes, K-Nearest Neighbor, dan Random Forest</p>
    <p style="margin: 0; font-size: 0.9rem;">Febriana Afiyah | Sifah Nur Rizkiyah | Linda Yulia Sudrajat</p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem;">Program Studi Informatika - Fakultas Teknik - Universitas Siliwangi</p>
    <p style="margin: 0; font-size: 0.8rem;">Tasikmalaya, 2026</p>
</div>
""", unsafe_allow_html=True)

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("### 📊 Menu Navigasi")
    menu = st.radio("", [
        "📋 Dashboard Utama",
        "📊 Perbandingan Model", 
        "🔮 Prediksi Sentimen",
        "📖 Tentang Proyek"
    ])
    
    st.markdown("---")
    st.markdown("### 🏆 Model Terbaik")
    st.info("""
    **K-Nearest Neighbor (KNN)**
    
    Berdasarkan evaluasi, KNN 
    merupakan model dengan 
    performa terbaik:
    - Accuracy: 94.30%
    - F1-Score: 93.95%
    """)

# ============ PREPROCESSING ============
slang_dict = {
    "gak": "tidak", "ga": "tidak", "nggak": "tidak", "gk": "tidak",
    "tdk": "tidak", "banget": "sangat", "bgt": "sangat", "bikin": "membuat",
    "jg": "juga", "dgn": "dengan", "utk": "untuk", "sm": "sama",
    "sdh": "sudah", "udah": "sudah", "blm": "belum", "aja": "saja",
    "aj": "saja", "dri": "dari", "krn": "karena", "sgt": "sangat",
    "btw": "omong-omong", "tp": "tetapi", "tpi": "tetapi", "sih": "sih",
    "gaenak": "tidak enak", "ga jelas": "tidak jelas",
    "gangerti": "tidak mengerti", "gatau": "tidak tahu", "gabisa": "tidak bisa"
}

def preprocess_text(text):
    """Preprocessing teks - sama dengan file analisis Python"""
    text = str(text).lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    words = text.split()
    words = [slang_dict.get(word, word) for word in words]
    text = ' '.join(words)
    
    tokens = text.split()
    tokens = [word for word in tokens if len(word) > 2]
    
    return ' '.join(tokens)

# ============ LOAD DATASET ASLI ============
@st.cache_data
def load_dataset():
    """Load dataset asli dari file CSV"""
    
    files_exist = {
        'train': os.path.exists('datd_train.csv'),
        'test': os.path.exists('datd_test.csv'),
        'rand': os.path.exists('datd_rand.csv')
    }
    
    if all(files_exist.values()):
        train_df = pd.read_csv('datd_train.csv')
        test_df = pd.read_csv('datd_test.csv')
        rand_df = pd.read_csv('datd_rand.csv')
        df = pd.concat([train_df, test_df, rand_df], ignore_index=True)
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
        return df, True
    else:
        st.error("❌ File CSV tidak ditemukan!")
        st.info("Pastikan file datd_train.csv, datd_test.csv, datd_rand.csv ada di folder yang sama")
        return None, False

# ============ TRAINING MODEL ============
@st.cache_resource
def train_models():
    """Latih model - sama persis dengan file analisis Python"""
    
    df, is_real = load_dataset()
    
    if df is None:
        return None, None, None, None, None, None
    
    # Preprocessing
    df['text_processed'] = df['text'].apply(preprocess_text)
    
    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85,
        sublinear_tf=True
    )
    X = vectorizer.fit_transform(df['text_processed'])
    y = df['label']
    
    # Split data (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Naive Bayes
    nb_model = MultinomialNB(alpha=0.5)
    nb_model.fit(X_train, y_train)
    
    # KNN
    knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn_model.fit(X_train, y_train)
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Prediksi
    y_pred_nb = nb_model.predict(X_test)
    y_pred_knn = knn_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)
    
    # Hasil evaluasi
    results = pd.DataFrame([
        {
            'Model': 'Naive Bayes',
            'Accuracy': accuracy_score(y_test, y_pred_nb),
            'Precision': precision_score(y_test, y_pred_nb, average='weighted'),
            'Recall': recall_score(y_test, y_pred_nb, average='weighted'),
            'F1-Score': f1_score(y_test, y_pred_nb, average='weighted')
        },
        {
            'Model': 'K-Nearest Neighbor (KNN)',
            'Accuracy': accuracy_score(y_test, y_pred_knn),
            'Precision': precision_score(y_test, y_pred_knn, average='weighted'),
            'Recall': recall_score(y_test, y_pred_knn, average='weighted'),
            'F1-Score': f1_score(y_test, y_pred_knn, average='weighted')
        },
        {
            'Model': 'Random Forest',
            'Accuracy': accuracy_score(y_test, y_pred_rf),
            'Precision': precision_score(y_test, y_pred_rf, average='weighted'),
            'Recall': recall_score(y_test, y_pred_rf, average='weighted'),
            'F1-Score': f1_score(y_test, y_pred_rf, average='weighted')
        }
    ])
    
    models = {
        'Naive Bayes': nb_model,
        'K-Nearest Neighbor (KNN)': knn_model,
        'Random Forest': rf_model
    }
    
    # Cross Validation (10-Fold)
    cv_results = {}
    for name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')
        cv_results[name] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores.tolist()
        }
    
    return df, vectorizer, models, results, cv_results, X_test, y_test

# Load data and train
with st.spinner("Memuat dataset dan melatih model..."):
    df, vectorizer, models, results_df, cv_results, X_test, y_test = train_models()

if df is None:
    st.stop()

# Hitung statistik
neg_count = len(df[df['label'] == 1])
neu_count = len(df[df['label'] == 0])

# ============ HALAMAN DASHBOARD UTAMA ============
if menu == "📋 Dashboard Utama":
    st.markdown("### 📊 Ringkasan Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📝 Total Tweet", f"{len(df):,}")
    with col2:
        st.metric("😊 Sentimen Netral", f"{neu_count} ({neu_count/len(df)*100:.1f}%)")
    with col3:
        st.metric("😰 Sentimen Negatif", f"{neg_count} ({neg_count/len(df)*100:.1f}%)")
    with col4:
        best_acc = results_df[results_df['Model'] == 'K-Nearest Neighbor (KNN)']['Accuracy'].values[0]
        st.metric("🏆 Model Terbaik", "KNN", delta=f"{best_acc*100:.2f}% Accuracy")
    
    # Distribusi Sentimen
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=[neu_count, neg_count],
            names=['Netral (0)', 'Negatif (1)'],
            title='Distribusi Sentimen Dataset',
            color_discrete_sequence=['#2a5298', '#ff6b6b'],
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Statistik Dataset")
        st.markdown(f"""
        | Keterangan | Nilai |
        |------------|-------|
        | Total Data | {len(df):,} |
        | Data Training (80%) | {int(len(df)*0.8):,} |
        | Data Testing (20%) | {int(len(df)*0.2):,} |
        | Fitur TF-IDF | 5,000 |
        | Kelas | 2 (Netral, Negatif) |
        """)
    
    # Performance Summary - KNN Terbaik
    st.markdown("---")
    st.markdown("### 🏆 Ringkasan Performa Model")
    
    knn_result = results_df[results_df['Model'] == 'K-Nearest Neighbor (KNN)'].iloc[0]
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0;">
        <h2 style="margin: 0; text-align: center;">🏆 K-Nearest Neighbor (KNN) - Model Terbaik</h2>
        <p style="text-align: center; margin: 0.5rem 0;">Berdasarkan evaluasi pada {len(df):,} tweet, KNN menunjukkan performa terbaik</p>
        <div style="display: flex; justify-content: space-around; text-align: center; margin-top: 1rem;">
            <div><h3>{knn_result['Accuracy']*100:.2f}%</h3><p>Accuracy</p></div>
            <div><h3>{knn_result['Precision']*100:.2f}%</h3><p>Precision</p></div>
            <div><h3>{knn_result['Recall']*100:.2f}%</h3><p>Recall</p></div>
            <div><h3>{knn_result['F1-Score']*100:.2f}%</h3><p>F1-Score</p></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample data
    with st.expander("📋 Lihat Contoh Data"):
        st.dataframe(df[['text', 'label']].head(10), use_container_width=True)

# ============ HALAMAN PERBANDINGAN MODEL ============
elif menu == "📊 Perbandingan Model":
    st.markdown("### 📊 Perbandingan Performa Model")
    
    # Bar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig = go.Figure()
    colors = ['#4a6fa5', '#1e3c72', '#2a5298']
    
    for i, model in enumerate(results_df['Model'].unique()):
        vals = results_df[results_df['Model'] == model][metrics].values[0]
        fig.add_trace(go.Bar(
            name=model,
            x=metrics,
            y=vals,
            text=[f"{v:.4f}" for v in vals],
            textposition='auto',
            marker_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        title="Perbandingan Performa Naive Bayes, KNN, dan Random Forest",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        barmode='group',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabel hasil
    st.markdown("#### Tabel Hasil Evaluasi")
    
    def highlight_best(val):
        if isinstance(val, (int, float)):
            if val > 0.94:
                return 'background-color: #d4efdf; color: #1e3c72; font-weight: bold'
        return ''
    
    st.dataframe(
        results_df.style.format({m: '{:.4f}' for m in metrics}).map(highlight_best, subset=metrics),
        use_container_width=True
    )
    
    # Radar Chart
    st.markdown("#### Radar Chart Perbandingan Komprehensif")
    
    fig2 = go.Figure()
    for model in results_df['Model'].unique():
        vals = results_df[results_df['Model'] == model][metrics].values[0].tolist()
        vals += vals[:1]
        fig2.add_trace(go.Scatterpolar(
            r=vals,
            theta=metrics + [metrics[0]],
            fill='toself',
            name=model
        ))
    
    fig2.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=500,
        title="Radar Chart Perbandingan Performa"
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Confusion Matrices
    st.markdown("#### Confusion Matrix")
    
    knn_model = models['K-Nearest Neighbor (KNN)']
    nb_model = models['Naive Bayes']
    rf_model = models['Random Forest']
    
    y_pred_knn = knn_model.predict(X_test)
    y_pred_nb = nb_model.predict(X_test)
    y_pred_rf = rf_model.predict(X_test)
    
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Naive Bayes**")
        fig_cm_nb = px.imshow(
            cm_nb, text_auto=True, 
            x=['Netral', 'Negatif'], y=['Netral', 'Negatif'],
            color_continuous_scale='Blues', title="Naive Bayes"
        )
        st.plotly_chart(fig_cm_nb, use_container_width=True)
    
    with col2:
        st.markdown("**KNN (Terbaik)**")
        fig_cm_knn = px.imshow(
            cm_knn, text_auto=True,
            x=['Netral', 'Negatif'], y=['Netral', 'Negatif'],
            color_continuous_scale='Greens', title="KNN - Model Terbaik"
        )
        st.plotly_chart(fig_cm_knn, use_container_width=True)
    
    with col3:
        st.markdown("**Random Forest**")
        fig_cm_rf = px.imshow(
            cm_rf, text_auto=True,
            x=['Netral', 'Negatif'], y=['Netral', 'Negatif'],
            color_continuous_scale='Blues', title="Random Forest"
        )
        st.plotly_chart(fig_cm_rf, use_container_width=True)
    
    # Cross Validation Results
    st.markdown("---")
    st.markdown("### 📉 10-Fold Cross Validation")
    
    fig_cv = go.Figure()
    for model_name, result in cv_results.items():
        fig_cv.add_trace(go.Box(
            y=result['scores'],
            name=model_name,
            boxmean='sd'
        ))
    
    fig_cv.update_layout(
        title="10-Fold Cross Validation Accuracy per Model",
        yaxis_title="Accuracy",
        yaxis_range=[0.5, 1],
        height=450
    )
    st.plotly_chart(fig_cv, use_container_width=True)
    
    # CV Statistics Table
    cv_df = pd.DataFrame([
        {
            'Model': model,
            'Mean Accuracy': f"{result['mean']:.4f}",
            'Std Deviation': f"{result['std']:.4f}",
            'Min': f"{min(result['scores']):.4f}",
            'Max': f"{max(result['scores']):.4f}"
        }
        for model, result in cv_results.items()
    ])
    st.dataframe(cv_df, use_container_width=True)
    
    st.success("""
    ### 📝 Kesimpulan Evaluasi
    
    **K-Nearest Neighbor (KNN) merupakan model terbaik** untuk analisis sentimen kesehatan mental pada data Twitter.
    """)

# ============ HALAMAN PREDIKSI SENTIMEN ============
elif menu == "🔮 Prediksi Sentimen":
    st.markdown("### 🔮 Prediksi Sentimen")
    st.markdown("Masukkan teks untuk diprediksi sentimennya menggunakan **KNN (Model Terbaik)**")
    
    model_choice = st.selectbox(
        "Pilih Model:",
        list(models.keys()),
        index=1
    )
    
    text_input = st.text_area(
        "Masukkan Teks:",
        height=120,
        placeholder="Contoh: Aku merasa sangat gelisah dan cemas hari ini..."
    )
    
    if st.button("🔍 Prediksi Sentimen", use_container_width=True):
        if text_input.strip():
            processed = preprocess_text(text_input)
            vec = vectorizer.transform([processed])
            
            model = models[model_choice]
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]
            
            if pred == 1:
                st.markdown(f"""
                <div style="background: #ff6b6b; padding: 1.5rem; border-radius: 15px; color: white; text-align: center;">
                    <h2>😰 NEGATIF</h2>
                    <p>Teks ini mengandung sentimen negatif terkait kesehatan mental</p>
                    <hr>
                    <p>Probabilitas Negatif: {proba[1]*100:.2f}%</p>
                    <p>Probabilitas Netral: {proba[0]*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #4CAF50; padding: 1.5rem; border-radius: 15px; color: white; text-align: center;">
                    <h2>😊 NETRAL</h2>
                    <p>Teks ini memiliki sentimen netral</p>
                    <hr>
                    <p>Probabilitas Netral: {proba[0]*100:.2f}%</p>
                    <p>Probabilitas Negatif: {proba[1]*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Silakan masukkan teks terlebih dahulu!")
    
    st.markdown("---")
    st.markdown("#### 💡 Contoh Tweet dari Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Tweet Negatif:**")
        neg_samples = df[df['label']==1]['text'].head(3).tolist()
        for i, tweet in enumerate(neg_samples):
            tweet_short = tweet[:80] + "..." if len(tweet) > 80 else tweet
            if st.button(f"📝 {i+1}. {tweet_short}", key=f"neg_{i}"):
                st.session_state['contoh'] = tweet
    
    with col2:
        st.markdown("**Tweet Netral:**")
        neu_samples = df[df['label']==0]['text'].head(3).tolist()
        for i, tweet in enumerate(neu_samples):
            tweet_short = tweet[:80] + "..." if len(tweet) > 80 else tweet
            if st.button(f"📝 {i+1}. {tweet_short}", key=f"neu_{i}"):
                st.session_state['contoh'] = tweet
    
    if 'contoh' in st.session_state:
        st.info(f"💡 Teks dipilih: {st.session_state['contoh'][:150]}")
        if st.button("✨ Gunakan teks ini untuk prediksi"):
            st.rerun()

# ============ HALAMAN TENTANG PROYEK ============
else:
    st.markdown("### 📖 Tentang Proyek")
    
    st.markdown("""
    #### 🎯 Latar Belakang
    Proyek ini menganalisis sentimen tweet tentang kesehatan mental menggunakan tiga algoritma machine learning:
    - **Naive Bayes** 
    - **K-Nearest Neighbor (KNN)**
    - **Random Forest**
    
    #### 🤖 Hasil Evaluasi Model
    | Algoritma | Accuracy | Precision | Recall | F1-Score |
    |-----------|----------|-----------|--------|----------|
    | Naive Bayes | 91.68% | 91.35% | 91.68% | 91.43% |
    | **KNN (Terbaik)** | **94.30%** | **94.30%** | **94.30%** | **93.95%** |
    | Random Forest | 94.25% | 94.08% | 94.25% | 94.07% |
    
    #### 👥 Anggota Kelompok
    | Nama | NPM |
    |------|-----|
    | Febriana Afiyah | 237006023 |
    | Sifah Nur Rizkiyah | 237006035 |
    | Linda Yulia Sudrajat | 237006040 |
    
    #### 📅 Mata Kuliah & Tahun
    - **Mata Kuliah:** Rekayasa Data
    - **Tahun:** 2026
    - **Program Studi:** Informatika
    - **Fakultas:** Teknik
    - **Universitas:** Siliwangi
    """)

# ============ FOOTER ============
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>© 2026 Kelompok 11 | Rekayasa Data | Program Studi Informatika</p>
    <p>Fakultas Teknik - Universitas Siliwangi | Tasikmalaya</p>
</div>
""", unsafe_allow_html=True)
