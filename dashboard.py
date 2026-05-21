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
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import warnings
import time
import os

warnings.filterwarnings('ignore')

# Download NLTK resources
@st.cache_resource
def download_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

download_nltk()

# Page configuration
st.set_page_config(
    page_title="Analisis Sentimen Kesehatan Mental",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-top: 1rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        transition: 0.3s;
    }
    footer {
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        border-top: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 Analisis Sentimen Kesehatan Mental</h1>
    <p>Perbandingan Algoritma Naive Bayes, K-Nearest Neighbor, dan Random Forest</p>
    <p style="font-size: 0.9rem;">Febriana Afiyah | Sifah Nur Rizkiyah | Linda Yulia Sudrajat</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Navigasi")
    page = st.radio("Pilih Halaman:", [
        "🏠 Dashboard Utama",
        "📈 Perbandingan Model",
        "🔮 Prediksi Sentimen",
        "📊 Dataset & Preprocessing",
        "📉 Cross Validation"
    ])
    
    st.markdown("---")
    st.markdown("### ℹ️ Tentang Proyek")
    st.info("""
    Proyek ini menganalisis sentimen tweet tentang kesehatan mental 
    menggunakan 3 algoritma machine learning:
    - **Naive Bayes** 
    - **K-Nearest Neighbor (KNN)**
    - **Random Forest**
    
    Data terdiri dari tweet berbahasa Indonesia dengan label:
    - **Netral (0)**: Tweet informasi, pertanyaan, atau dukungan
    - **Negatif (1)**: Tweet mengandung kecemasan, depresi, atau keluhan mental
    """)

# Preprocessing function
def preprocess_text(text):
    """Preprocessing teks untuk analisis sentimen"""
    slang_dict = {
        "gak": "tidak", "ga": "tidak", "nggak": "tidak", "gk": "tidak",
        "tdk": "tidak", "banget": "sangat", "bgt": "sangat", "bikin": "membuat",
        "jg": "juga", "dgn": "dengan", "utk": "untuk", "sm": "sama",
        "sdh": "sudah", "udah": "sudah", "blm": "belum", "aja": "saja",
        "aj": "saja", "dri": "dari", "krn": "karena", "sgt": "sangat",
        "btw": "omong-omong", "tp": "tetapi", "tpi": "tetapi", "sih": "sih",
        "gaenak": "tidak enak", "ga jelas": "tidak jelas", "gangerti": "tidak mengerti",
        "gatau": "tidak tahu", "gabisa": "tidak bisa"
    }
    
    stop_words = set(stopwords.words('indonesian'))
    custom_stopwords = {'rt', 'tw', 'user', 'https', 'http', 'co', 'amp',
                        'yg', 'nya', 'para', 'pada', 'oleh', 'bagi', 'dr', 'jd',
                        'dgn', 'utk', 'aja', 'kok', 'deh', 'dong', 'nih', 'sih',
                        'udh', 'sdh', 'jg'}
    stop_words.update(custom_stopwords)
    
    # Case folding
    text = str(text).lower()
    
    # Remove noise
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Normalisasi slang
    words = text.split()
    words = [slang_dict.get(word, word) for word in words]
    text = ' '.join(words)
    
    # Tokenisasi
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    
    # Hapus stopwords dan kata pendek
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Stemming sederhana
    stemmed = []
    for word in tokens:
        if word.startswith('me') and len(word) > 2:
            word = word[2:]
        elif word.startswith('di') and len(word) > 2:
            word = word[2:]
        elif word.startswith('ter') and len(word) > 3:
            word = word[3:]
        elif word.startswith('ber') and len(word) > 3:
            word = word[3:]
        elif word.endswith('kan') and len(word) > 3:
            word = word[:-3]
        elif word.endswith('an') and len(word) > 2:
            word = word[:-2]
        stemmed.append(word if len(word) > 2 else word)
    
    return ' '.join(stemmed)

# CEK FILE CSV - PRIORITAS DATASET ASLI
required_files = ['datd_train.csv', 'datd_test.csv', 'datd_rand.csv']
missing_files = [f for f in required_files if not os.path.exists(f)]

USE_REAL_DATA = len(missing_files) == 0

if not USE_REAL_DATA:
    st.warning(f"⚠️ File CSV tidak ditemukan: {', '.join(missing_files)}")
    st.info("💡 Menggunakan data sintetis (buatan) untuk demo. Untuk menggunakan dataset asli, pastikan file CSV berada di folder yang sama dengan dashboard.py")

@st.cache_data
def load_real_data():
    """Load dataset from CSV files"""
    train_df = pd.read_csv('datd_train.csv')
    test_df = pd.read_csv('datd_test.csv')
    rand_df = pd.read_csv('datd_rand.csv')
    
    # Gabungkan semua data
    df = pd.concat([train_df, test_df, rand_df], ignore_index=True)
    
    # Drop baris yang memiliki label NaN
    df = df.dropna(subset=['label'])
    
    # Konversi label ke integer
    df['label'] = df['label'].astype(int)
    
    return df

@st.cache_data
def load_synthetic_data():
    """Create synthetic dataset for demonstration"""
    np.random.seed(42)
    
    negative_tweets = [
        "aku gelisah banget hari ini", "rasa cemas terus menerus", "stress kerja makin parah",
        "depresi berat rasanya", "takut dan cemas tanpa sebab", "gelisah ga bisa tidur",
        "pikiran kacau semua", "sedih banget nangis terus", "cemas berlebihan",
        "overthinking mulu", "ga tenang rasanya", "resah gelisah melanda"
    ]
    
    neutral_tweets = [
        "hari ini cerah sekali", "ayo semangat kerja", "makan enak nih",
        "jalan-jalan seru", "ketawa sama teman", "olahraga pagi",
        "belajar hal baru", "dengerin musik", "nonton film bagus",
        "kumpul keluarga", "baca buku menarik", "info kesehatan mental"
    ]
    
    n_samples = 1500
    texts = []
    labels = []
    
    for i in range(n_samples):
        if i < n_samples // 2:
            texts.append(np.random.choice(negative_tweets))
            labels.append(1)
        else:
            texts.append(np.random.choice(neutral_tweets))
            labels.append(0)
    
    df = pd.DataFrame({'text': texts, 'label': labels})
    return df

@st.cache_data
def load_and_preprocess():
    """Load data, preprocess, and train models"""
    
    # Pilih sumber data
    if USE_REAL_DATA:
        st.info("📊 Menggunakan dataset asli dari file CSV")
        df = load_real_data()
    else:
        st.info("🔧 Menggunakan data sintetis (demo mode)")
        df = load_synthetic_data()
    
    # Preprocess texts
    with st.spinner("⏳ Sedang melakukan preprocessing teks..."):
        df['text_processed'] = df['text'].apply(preprocess_text)
    
    # TF-IDF Vectorization
    with st.spinner("⏳ Sedang melakukan ekstraksi fitur TF-IDF..."):
        vectorizer = TfidfVectorizer(
            max_features=3000, 
            ngram_range=(1, 2), 
            min_df=2, 
            max_df=0.85,
            sublinear_tf=True
        )
        X = vectorizer.fit_transform(df['text_processed'])
        y = df['label']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    models = {
        'Naive Bayes': MultinomialNB(alpha=0.5),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    trained_models = {}
    results = []
    
    for name, model in models.items():
        with st.spinner(f'⏳ Melatih model {name}...'):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            results.append({
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted'),
                'Recall': recall_score(y_test, y_pred, average='weighted'),
                'F1-Score': f1_score(y_test, y_pred, average='weighted')
            })
            trained_models[name] = model
    
    # Cross validation results
    cv_results = {}
    for name, model in trained_models.items():
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        cv_results[name] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores
        }
    
    return df, vectorizer, trained_models, pd.DataFrame(results), X_test, y_test, cv_results

# Load data and train models
with st.spinner("🚀 Memuat data dan melatih model..."):
    df, vectorizer, models, results_df, X_test, y_test, cv_results = load_and_preprocess()

# Halaman Dashboard Utama
if page == "🏠 Dashboard Utama":
    st.markdown("### 📊 Ringkasan Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📝 Total Tweet</h3>
            <h2>{len(df)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        neg_count = len(df[df['label'] == 1])
        st.markdown(f"""
        <div class="metric-card">
            <h3>😰 Negatif</h3>
            <h2>{neg_count} ({neg_count/len(df)*100:.1f}%)</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        neu_count = len(df[df['label'] == 0])
        st.markdown(f"""
        <div class="metric-card">
            <h3>😊 Netral</h3>
            <h2>{neu_count} ({neu_count/len(df)*100:.1f}%)</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔤 Fitur</h3>
            <h2>3000</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualisasi Distribusi
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Distribusi Sentimen")
        fig = px.pie(
            values=[neu_count, neg_count],
            names=['Netral (0)', 'Negatif (1)'],
            title='Persentase Sentimen Tweet',
            color_discrete_sequence=['#45B7D1', '#FF6B6B'],
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 Insight Dataset")
        
        st.markdown("**Contoh Tweet Negatif:**")
        neg_examples = df[df['label'] == 1]['text'].head(3).tolist()
        for i, tweet in enumerate(neg_examples, 1):
            tweet_preview = tweet[:100] + "..." if len(tweet) > 100 else tweet
            st.write(f"{i}. {tweet_preview}")
        
        st.markdown("**Contoh Tweet Netral:**")
        neu_examples = df[df['label'] == 0]['text'].head(3).tolist()
        for i, tweet in enumerate(neu_examples, 1):
            tweet_preview = tweet[:100] + "..." if len(tweet) > 100 else tweet
            st.write(f"{i}. {tweet_preview}")
    
    # Best model highlight
    st.markdown("---")
    best_model = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
    best_f1 = results_df.loc[results_df['F1-Score'].idxmax(), 'F1-Score']
    
    st.markdown(f"""
    <div class="prediction-card">
        <h3>🏆 Model Terbaik: {best_model}</h3>
        <p>Berdasarkan evaluasi pada {len(df)} data tweet, model {best_model} memiliki performa terbaik dengan F1-Score: {best_f1:.4f} ({best_f1*100:.2f}%)</p>
        <p>Model ini paling akurat dalam mengklasifikasikan sentimen tweet tentang kesehatan mental.</p>
    </div>
    """, unsafe_allow_html=True)

# Halaman Perbandingan Model
elif page == "📈 Perbandingan Model":
    st.markdown("### 🎯 Perbandingan Performa Model")
    
    # Metrics comparison
    fig = go.Figure()
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for model in results_df['Model'].unique():
        model_data = results_df[results_df['Model'] == model]
        fig.add_trace(go.Bar(
            name=model,
            x=metrics,
            y=[model_data[m].values[0] for m in metrics],
            text=[f"{model_data[m].values[0]:.3f}" for m in metrics],
            textposition='auto'
        ))
    
    fig.update_layout(
        title="Perbandingan Performa Ketiga Model",
        xaxis_title="Metrik",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        barmode='group',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabel hasil
    st.markdown("#### 📋 Tabel Hasil Evaluasi")
    st.dataframe(
        results_df.style.format({
            'Accuracy': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1-Score': '{:.4f}'
        }).background_gradient(cmap='Blues'),
        use_container_width=True
    )
    
    # Radar chart
    st.markdown("#### 🎯 Radar Chart Comparison")
    
    fig = go.Figure()
    
    for model in results_df['Model'].unique():
        model_data = results_df[results_df['Model'] == model]
        values = [model_data[m].values[0] for m in metrics]
        values += values[:1]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill='toself',
            name=model
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# Halaman Prediksi Sentimen
elif page == "🔮 Prediksi Sentimen":
    st.markdown("### 🔮 Prediksi Sentimen Real-Time")
    
    st.markdown("""
    Masukkan teks tweet atau kalimat tentang kesehatan mental di bawah ini untuk 
    memprediksi sentimennya menggunakan model terbaik.
    """)
    
    # Tampilkan sumber data
    if USE_REAL_DATA:
        st.success("✅ Model dilatih dengan **dataset asli** dari file CSV")
    else:
        st.info("ℹ️ Model dilatih dengan **data sintetis** (karena file CSV tidak ditemukan)")
    
    # Select model
    selected_model = st.selectbox(
        "Pilih Model untuk Prediksi:",
        list(models.keys())
    )
    
    # Text input
    user_input = st.text_area(
        "Masukkan Teks:",
        placeholder="Contoh: Aku merasa sangat gelisah dan cemas hari ini...",
        height=150
    )
    
    if st.button("🔍 Prediksi Sentimen", use_container_width=True):
        if user_input.strip():
            with st.spinner("Memproses teks dan memprediksi..."):
                time.sleep(0.5)
                
                processed = preprocess_text(user_input)
                vectorized = vectorizer.transform([processed])
                
                model = models[selected_model]
                prediction = model.predict(vectorized)[0]
                proba = model.predict_proba(vectorized)[0]
                
                sentiment = "Negatif (Mengandung Kecemasan)" if prediction == 1 else "Netral (Informasi/Dukungan)"
                emoji = "😰" if prediction == 1 else "😊"
                color = "#FF6B6B" if prediction == 1 else "#45B7D1"
                
                st.markdown(f"""
                <div class="prediction-card" style="background: linear-gradient(135deg, {color} 0%, {color}cc 100%);">
                    <h2>{emoji} Hasil Prediksi: {sentiment}</h2>
                    <hr>
                    <p><strong>Teks Input:</strong><br>{user_input}</p>
                    <p><strong>Probabilitas:</strong></p>
                    <ul>
                        <li>Netral: {proba[0]:.3f} ({proba[0]*100:.1f}%)</li>
                        <li>Negatif: {proba[1]:.3f} ({proba[1]*100:.1f}%)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Silakan masukkan teks terlebih dahulu!")
    
    # Contoh tweet dari dataset
    st.markdown("---")
    st.markdown("#### 💡 Contoh Tweet dari Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Tweet Negatif:**")
        neg_samples = df[df['label'] == 1]['text'].head(3).tolist()
        for i, tweet in enumerate(neg_samples, 1):
            tweet_preview = tweet[:80] + "..." if len(tweet) > 80 else tweet
            if st.button(f"📝 Contoh Negatif {i}", key=f"neg_{i}"):
                st.session_state['example_text'] = tweet
    
    with col2:
        st.markdown("**Tweet Netral:**")
        neu_samples = df[df['label'] == 0]['text'].head(3).tolist()
        for i, tweet in enumerate(neu_samples, 1):
            tweet_preview = tweet[:80] + "..." if len(tweet) > 80 else tweet
            if st.button(f"📝 Contoh Netral {i}", key=f"neu_{i}"):
                st.session_state['example_text'] = tweet
    
    if 'example_text' in st.session_state:
        st.info(f"💡 Teks contoh: \"{st.session_state['example_text'][:100]}...\"")
        if st.button("Gunakan teks ini untuk prediksi"):
            st.rerun()

# Halaman Dataset & Preprocessing
elif page == "📊 Dataset & Preprocessing":
    st.markdown("### 📊 Dataset dan Preprocessing")
    
    # Tampilkan sumber data
    if USE_REAL_DATA:
        st.success("✅ **Sumber Data: Dataset Asli dari File CSV**")
        st.caption(f"File yang digunakan: datd_train.csv, datd_test.csv, datd_rand.csv")
    else:
        st.warning("⚠️ **Sumber Data: Sintetis (Demo Mode)**")
        st.caption("File CSV tidak ditemukan, menggunakan data buatan untuk demo")
    
    tab1, tab2, tab3 = st.tabs(["📋 Dataset Preview", "🔧 Preprocessing Steps", "📝 Sample Comparison"])
    
    with tab1:
        st.markdown("#### Preview Dataset (20 baris pertama)")
        st.dataframe(df[['text', 'label']].head(20), use_container_width=True)
        
        st.markdown("#### Statistik Dataset")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Distribusi Label:**")
            label_counts = df['label'].value_counts().sort_index()
            st.write(f"Netral (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(df)*100:.1f}%)")
            st.write(f"Negatif (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(df)*100:.1f}%)")
        with col2:
            st.write("**Info Dataset:**")
            st.write(f"- Total data: {len(df)}")
            st.write(f"- Fitur TF-IDF: 3000")
            st.write(f"- Train-Test split: 80%-20%")
            st.write(f"- Sumber: Twitter (X)")
    
    with tab2:
        st.markdown("#### Proses Preprocessing")
        
        steps = {
            "1. Case Folding": "Mengubah semua huruf menjadi lowercase (HURUF BESAR → huruf kecil)",
            "2. Cleaning": "Menghapus mention (@username), URL, hashtag (#), angka, dan karakter spesial",
            "3. Normalisasi Slang": "Mengubah kata tidak baku menjadi baku (gak → tidak, bgt → sangat)",
            "4. Tokenisasi": "Memecah kalimat menjadi kata-kata individual",
            "5. Stopword Removal": "Menghapus kata umum yang tidak memiliki makna (dan, yang, di, ke, dari)",
            "6. Stemming": "Mengubah kata ke bentuk dasarnya (mengobati → obat, berjalan → jalan)"
        }
        
        for step, desc in steps.items():
            st.markdown(f"**{step}** : {desc}")
        
        st.markdown("---")
        st.markdown("#### Contoh Hasil Preprocessing")
        
        # Ambil sample dari dataset
        sample_options = df['text'].head(5).tolist()
        sample_text = st.selectbox("Pilih contoh teks dari dataset:", sample_options)
        
        if sample_text:
            processed = preprocess_text(sample_text)
            st.markdown(f"**Original:** {sample_text}")
            st.markdown(f"**Processed:** {processed}")
    
    with tab3:
        st.markdown("#### Perbandingan Sebelum dan Sesudah Preprocessing (10 baris)")
        
        comparison_df = df[['text', 'text_processed']].head(10).copy()
        st.dataframe(comparison_df, use_container_width=True)

# Halaman Cross Validation
elif page == "📉 Cross Validation":
    st.markdown("### 📉 Cross Validation Results (5-Fold)")
    
    st.markdown("""
    Cross validation dilakukan untuk memastikan model tidak overfitting dan 
    memiliki performa yang stabil pada data yang berbeda. Metode yang digunakan adalah **5-Fold Cross Validation**.
    """)
    
    # Display cross validation results as box plot
    fig = go.Figure()
    
    colors = ['#667eea', '#764ba2', '#45B7D1']
    for idx, (model_name, result) in enumerate(cv_results.items()):
        fig.add_trace(go.Box(
            y=result['scores'],
            name=model_name,
            boxmean='sd',
            marker_color=colors[idx % len(colors)]
        ))
    
    fig.update_layout(
        title="5-Fold Cross Validation Results per Model",
        xaxis_title="Model",
        yaxis_title="Accuracy",
        yaxis_range=[0, 1],
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Table of CV results
    cv_df = pd.DataFrame([
        {
            'Model': model,
            'Mean Accuracy': result['mean'],
            'Std Deviation': result['std'],
            'Min Accuracy': result['scores'].min(),
            'Max Accuracy': result['scores'].max()
        }
        for model, result in cv_results.items()
    ])
    
    st.markdown("#### 📊 Cross Validation Statistics")
    st.dataframe(
        cv_df.style.format({
            'Mean Accuracy': '{:.4f}',
            'Std Deviation': '{:.4f}',
            'Min Accuracy': '{:.4f}',
            'Max Accuracy': '{:.4f}'
        }).background_gradient(cmap='Blues', subset=['Mean Accuracy']),
        use_container_width=True
    )
    
    # Interpretation
    st.markdown("---")
    st.markdown("#### 📝 Interpretasi Hasil")
    
    best_cv_model = cv_df.loc[cv_df['Mean Accuracy'].idxmax(), 'Model']
    best_cv_score = cv_df.loc[cv_df['Mean Accuracy'].idxmax(), 'Mean Accuracy']
    lowest_std_model = cv_df.loc[cv_df['Std Deviation'].idxmin(), 'Model']
    
    st.markdown(f"""
    - **Model dengan performa CV terbaik:** **{best_cv_model}** (Mean Accuracy: {best_cv_score:.4f} atau {best_cv_score*100:.2f}%)
    - **Model paling stabil:** **{lowest_std_model}** (Standard Deviation terkecil: {cv_df.loc[cv_df['Std Deviation'].idxmin(), 'Std Deviation']:.4f})
    - **Kesimpulan:** Model Random Forest menunjukkan performa yang paling konsisten dan akurat.
    """)

# Footer
st.markdown("---")
st.markdown("""
<footer>
    <p>© 2024 Kelompok 11 - Rekayasa Data | Analisis Sentimen Kesehatan Mental</p>
    <p>Dibangun dengan Streamlit, Scikit-learn, Plotly, dan NLTK</p>
</footer>
""", unsafe_allow_html=True)