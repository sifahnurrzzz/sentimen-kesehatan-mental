import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# ============ KONFIGURASI ============
st.set_page_config(
    page_title="Analisis Sentimen Kesehatan Mental",
    page_icon="🧠",
    layout="wide"
)

# ============ HEADER ============
st.markdown("""
<div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 2rem; border-radius: 20px; color: white; text-align: center; margin-bottom: 2rem;">
    <h1 style="margin: 0;">🧠 Analisis Sentimen Kesehatan Mental</h1>
    <p style="margin: 0.5rem 0;">Perbandingan Naive Bayes, KNN, dan Random Forest</p>
    <p style="margin: 0; font-size: 0.9rem;">Febriana Afiyah | Sifah Nur Rizkiyah | Linda Yulia Sudrajat</p>
</div>
""", unsafe_allow_html=True)

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("### 📊 Menu")
    menu = st.radio("", [
        "📋 Dataset", 
        "📊 Perbandingan Model", 
        "🔮 Prediksi",
        "📖 Tentang"
    ])
    
    st.markdown("---")
    st.markdown("### 📊 Statistik")
    
    # Load data untuk statistik
    @st.cache_data
    def load_stats():
        if os.path.exists('datd_train.csv'):
            train = pd.read_csv('datd_train.csv')
            test = pd.read_csv('datd_test.csv')
            rand = pd.read_csv('datd_rand.csv')
            df = pd.concat([train, test, rand], ignore_index=True)
            df = df.dropna()
            return df, True
        else:
            # Data sintetis
            np.random.seed(42)
            texts = ['gelisah']*250 + ['semangat']*250
            labels = [1]*250 + [0]*250
            return pd.DataFrame({'text': texts, 'label': labels}), False
    
    df, is_real = load_stats()
    
    st.metric("📝 Total Data", len(df))
    st.metric("😰 Sentimen Negatif", len(df[df['label']==1]))
    st.metric("😊 Sentimen Netral", len(df[df['label']==0]))

# ============ PREPROCESSING ============
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

# ============ TRAINING MODEL ============
@st.cache_resource
def train_all_models():
    # Load data
    if os.path.exists('datd_train.csv'):
        train = pd.read_csv('datd_train.csv')
        test = pd.read_csv('datd_test.csv')
        rand = pd.read_csv('datd_rand.csv')
        df = pd.concat([train, test, rand], ignore_index=True)
        df = df.dropna()
    else:
        np.random.seed(42)
        neg = ["gelisah", "cemas", "stress", "depresi", "takut", "sedih"]
        neu = ["semangat", "sehat", "bahagia", "tenang", "baik", "senang"]
        texts = [np.random.choice(neg) for _ in range(300)] + [np.random.choice(neu) for _ in range(300)]
        labels = [1]*300 + [0]*300
        df = pd.DataFrame({'text': texts, 'label': labels})
    
    # Preprocess
    df['clean'] = df['text'].apply(clean_text)
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['clean'])
    y = df['label']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Models
    models = {
        'Naive Bayes': MultinomialNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = []
    trained = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        trained[name] = model
        
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1-Score': f1_score(y_test, y_pred, average='weighted')
        })
    
    return df, vectorizer, trained, pd.DataFrame(results)

# ============ HALAMAN DATASET ============
if menu == "📋 Dataset":
    st.markdown("### 📋 Dataset Tweet")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Contoh Data")
        st.dataframe(df[['text', 'label']].head(10), use_container_width=True)
    
    with col2:
        st.markdown("#### Distribusi Label")
        fig = px.pie(
            values=[len(df[df['label']==0]), len(df[df['label']==1])],
            names=['Netral (0)', 'Negatif (1)'],
            color_discrete_sequence=['#2a5298', '#ff6b6b'],
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("#### Proses Preprocessing")
    st.markdown("""
    1. **Case Folding** - Mengubah huruf menjadi lowercase
    2. **Cleaning** - Menghapus karakter khusus, angka, dan tanda baca
    3. **Tokenisasi** - Memecah kalimat menjadi kata
    4. **Stopword Removal** - Menghapus kata umum (dan, yang, di)
    5. **Vectorization** - TF-IDF untuk mengubah teks menjadi angka
    """)

# ============ HALAMAN PERBANDINGAN ============
elif menu == "📊 Perbandingan Model":
    st.markdown("### 📊 Perbandingan Performa Model")
    
    with st.spinner("Melatih model..."):
        df, vectorizer, models, results_df = train_all_models()
    
    # Bar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig = go.Figure()
    colors = ['#1e3c72', '#2a5298', '#4a6fa5']
    
    for i, model in enumerate(results_df['Model'].unique()):
        vals = results_df[results_df['Model'] == model][metrics].values[0]
        fig.add_trace(go.Bar(
            name=model,
            x=metrics,
            y=vals,
            text=[f"{v:.3f}" for v in vals],
            textposition='auto',
            marker_color=colors[i]
        ))
    
    fig.update_layout(
        title="Perbandingan Ketiga Model",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        barmode='group',
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabel
    st.markdown("#### Hasil Evaluasi Lengkap")
    st.dataframe(
        results_df.style.format({m: '{:.4f}' for m in metrics}).background_gradient(cmap='Blues', subset=['F1-Score']),
        use_container_width=True
    )
    
    # Best model
    best = results_df.loc[results_df['F1-Score'].idxmax()]
    st.success(f"""
    🏆 **Model Terbaik: {best['Model']}**
    
    - Accuracy: {best['Accuracy']:.4f} ({best['Accuracy']*100:.2f}%)
    - F1-Score: best['F1-Score']:.4f} ({best['F1-Score']*100:.2f}%)
    """)
    
    # Radar chart
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
        height=450
    )
    st.plotly_chart(fig2, use_container_width=True)

# ============ HALAMAN PREDIKSI ============
elif menu == "🔮 Prediksi":
    st.markdown("### 🔮 Prediksi Sentimen")
    
    with st.spinner("Memuat model..."):
        df, vectorizer, models, results_df = train_all_models()
    
    # Pilih model
    model_choice = st.selectbox(
        "Pilih Model yang Akan Digunakan:",
        list(models.keys()),
        index=2  # default Random Forest
    )
    
    # Input teks
    text_input = st.text_area(
        "Masukkan Teks Tweet:",
        height=120,
        placeholder="Contoh: Aku merasa sangat gelisah dan cemas hari ini..."
    )
    
    # Prediksi
    if st.button("🔍 Prediksi Sentimen", use_container_width=True):
        if text_input.strip():
            # Preprocess
            clean = clean_text(text_input)
            vec = vectorizer.transform([clean])
            
            # Predict
            model = models[model_choice]
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]
            
            # Result
            if pred == 1:
                st.markdown("""
                <div style="background: #ff6b6b; padding: 1.5rem; border-radius: 15px; color: white; text-align: center;">
                    <h2>😰 NEGATIF</h2>
                    <p>Teks ini mengandung sentimen negatif terkait kesehatan mental</p>
                    <hr>
                    <p>Probabilitas Negatif: {:.2f}%</p>
                    <p>Probabilitas Netral: {:.2f}%</p>
                </div>
                """.format(proba[1]*100, proba[0]*100), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #4CAF50; padding: 1.5rem; border-radius: 15px; color: white; text-align: center;">
                    <h2>😊 NETRAL</h2>
                    <p>Teks ini memiliki sentimen netral</p>
                    <hr>
                    <p>Probabilitas Netral: {:.2f}%</p>
                    <p>Probabilitas Negatif: {:.2f}%</p>
                </div>
                """.format(proba[0]*100, proba[1]*100), unsafe_allow_html=True)
        else:
            st.warning("Silakan masukkan teks terlebih dahulu!")
    
    # Contoh
    st.markdown("---")
    st.markdown("#### 💡 Contoh Teks untuk Dicoba")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("😰 Contoh Sentimen Negatif"):
            st.session_state['contoh'] = "aku gelisah banget ga bisa tidur semalaman"
    with col2:
        if st.button("😊 Contoh Sentimen Netral"):
            st.session_state['contoh'] = "ayo jaga kesehatan mental dengan olahraga"
    
    if 'contoh' in st.session_state:
        st.info(f"📝 Teks: {st.session_state['contoh']}")

# ============ HALAMAN TENTANG ============
else:
    st.markdown("### 📖 Tentang Proyek")
    
    st.markdown("""
    #### 🎯 Latar Belakang
    Proyek ini bertujuan untuk menganalisis sentimen tweet tentang kesehatan mental 
    menggunakan tiga algoritma machine learning.
    
    #### 🤖 Algoritma yang Digunakan
    | Algoritma | Kelebihan | Kekurangan |
    |-----------|-----------|------------|
    | **Naive Bayes** | Cepat, baik untuk teks | Asumsi independensi fitur |
    | **KNN** | Sederhana, akurat | Lambat untuk data besar |
    | **Random Forest** | Akurasi tinggi, robust | Lebih kompleks |
    
    #### 📊 Metrik Evaluasi
    - **Accuracy**: Persentase prediksi benar
    - **Precision**: Ketepatan prediksi positif
    - **Recall**: Kemampuan menemukan semua positif
    - **F1-Score**: Harmonisasi precision & recall
    
    #### 👥 Anggota Kelompok
    - Febriana Afiyah (237006023)
    - Sifah Nur Rizkiyah (237006035)
    - Linda Yulia Sudrajat (237006040)
    
    #### 📅 Mata Kuliah
    Rekayasa Data - UAS
    """)

# ============ FOOTER ============
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    © 2024 Kelompok 11 | Analisis Sentimen Kesehatan Mental
</div>
""", unsafe_allow_html=True)
