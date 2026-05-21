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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

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

# ============ LOAD DATASET ASLI ============
@st.cache_data
def load_dataset():
    """Load dataset asli dari file CSV"""
    
    # Cek apakah file ada
    files_exist = {
        'train': os.path.exists('datd_train.csv'),
        'test': os.path.exists('datd_test.csv'),
        'rand': os.path.exists('datd_rand.csv')
    }
    
    if all(files_exist.values()):
        # Load semua file CSV asli
        train_df = pd.read_csv('datd_train.csv')
        test_df = pd.read_csv('datd_test.csv')
        rand_df = pd.read_csv('datd_rand.csv')
        
        # Gabungkan semua data
        df = pd.concat([train_df, test_df, rand_df], ignore_index=True)
        
        # Bersihkan data
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
        
        st.success(f"✅ Dataset asli berhasil dimuat! Total {len(df)} tweet")
        return df, True
    else:
        st.error("❌ File CSV tidak ditemukan!")
        st.info("Pastikan file berikut ada di folder yang sama:")
        for name, exists in files_exist.items():
            status = "✅" if exists else "❌"
            st.write(f"{status} datd_{name}.csv")
        
        # Fallback ke data kecil untuk demo
        st.warning("⚠️ Menggunakan data demo kecil...")
        demo_data = pd.DataFrame({
            'text': ['aku gelisah', 'semangat pagi', 'stress berat', 'bahagia selalu', 'cemas mulu', 'sehat selalu'],
            'label': [1, 0, 1, 0, 1, 0]
        })
        return demo_data, False

# ============ PREPROCESSING ============
def clean_text(text):
    """Bersihkan teks"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ============ TRAINING MODEL ============
@st.cache_resource
def train_models():
    """Latih model dengan dataset asli"""
    
    df, is_real = load_dataset()
    
    # Preprocessing
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Hapus teks kosong
    df = df[df['clean_text'].str.len() > 0]
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Models
    models = {
        'Naive Bayes': MultinomialNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42)
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        trained_models[name] = model
        
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1-Score': f1_score(y_test, y_pred, average='weighted')
        })
    
    return df, vectorizer, trained_models, pd.DataFrame(results), y_test

# Load data and train
with st.spinner("Memuat dataset dan melatih model..."):
    df, vectorizer, models, results_df, y_test = train_models()

# ============ HALAMAN DATASET ============
if menu == "📋 Dataset":
    st.markdown("### 📋 Dataset Tweet")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("📝 Total Data", len(df))
        st.metric("😰 Sentimen Negatif (1)", len(df[df['label']==1]))
        st.metric("😊 Sentimen Netral (0)", len(df[df['label']==0]))
    
    with col2:
        fig = px.pie(
            values=[len(df[df['label']==0]), len(df[df['label']==1])],
            names=['Netral (0)', 'Negatif (1)'],
            title='Distribusi Sentimen',
            color_discrete_sequence=['#2a5298', '#ff6b6b']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### Contoh Data (20 baris pertama)")
    st.dataframe(df[['text', 'label']].head(20), use_container_width=True)
    
    # Statistik tambahan
    with st.expander("📊 Statistik Dataset"):
        st.write("**Jumlah per label:**")
        st.write(df['label'].value_counts())
        st.write("**Panjang teks (karakter):**")
        st.write(f"Min: {df['text'].str.len().min()}")
        st.write(f"Max: {df['text'].str.len().max()}")
        st.write(f"Rata-rata: {df['text'].str.len().mean():.0f}")

# ============ HALAMAN PERBANDINGAN MODEL ============
elif menu == "📊 Perbandingan Model":
    st.markdown("### 📊 Perbandingan Performa Model")
    
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
        title="Perbandingan Performa Model",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        barmode='group',
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabel hasil
    st.markdown("#### Tabel Hasil Evaluasi")
    st.dataframe(
        results_df.style.format({m: '{:.4f}' for m in metrics}),
        use_container_width=True
    )
    
    # Best model
    best = results_df.loc[results_df['F1-Score'].idxmax()]
    st.success(f"""
    🏆 **Model Terbaik: {best['Model']}**  
    - Accuracy: {best['Accuracy']:.4f} ({best['Accuracy']*100:.2f}%)  
    - F1-Score: {best['F1-Score']:.4f} ({best['F1-Score']*100:.2f}%)
    """)
    
    # Confusion Matrix untuk model terbaik
    st.markdown("#### Confusion Matrix (Model Terbaik)")
    
    best_model = models[best['Model']]
    # Perlu re-predict untuk confusion matrix
    df_temp, vec_temp, _, _, y_test_temp = train_models()
    X_temp = vec_temp.transform(df_temp['clean_text'])
    _, X_test_temp, _, y_test_temp = train_test_split(X_temp, df_temp['label'], test_size=0.2, random_state=42, stratify=df_temp['label'])
    y_pred_best = best_model.predict(X_test_temp)
    
    cm = confusion_matrix(y_test_temp, y_pred_best)
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        x=['Netral (0)', 'Negatif (1)'],
        y=['Netral (0)', 'Negatif (1)'],
        color_continuous_scale='Blues',
        title=f"Confusion Matrix - {best['Model']}"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

# ============ HALAMAN PREDIKSI ============
elif menu == "🔮 Prediksi":
    st.markdown("### 🔮 Prediksi Sentimen")
    
    # Pilih model
    model_choice = st.selectbox(
        "Pilih Model:",
        list(models.keys()),
        index=2  # Random Forest sebagai default
    )
    
    # Input teks
    text_input = st.text_area(
        "Masukkan Teks:",
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
            
            # Hasil
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
    
    # Contoh tweet dari dataset asli
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

# ============ HALAMAN TENTANG ============
else:
    st.markdown("### 📖 Tentang Proyek")
    
    st.markdown("""
    #### 🎯 Latar Belakang
    Proyek ini menganalisis sentimen tweet tentang kesehatan mental menggunakan tiga algoritma machine learning.
    
    #### 🤖 Algoritma
    | Algoritma | Kelebihan |
    |-----------|-----------|
    | **Naive Bayes** | Cepat, cocok untuk klasifikasi teks |
    | **KNN** | Sederhana, akurat untuk data kecil |
    | **Random Forest** | Akurasi tinggi, robust |
    
    #### 📊 Metrik Evaluasi
    - **Accuracy**: Persentase prediksi yang benar
    - **Precision**: Ketepatan prediksi positif
    - **Recall**: Kemampuan menemukan data positif
    - **F1-Score**: Rata-rata harmonis precision & recall
    
    #### 👥 Anggota Kelompok
    - **Febriana Afiyah** (237006023)
    - **Sifah Nur Rizkiyah** (237006035)
    - **Linda Yulia Sudrajat** (237006040)
    
    #### 📅 Mata Kuliah
    Rekayasa Data - UAS
    
    #### 📂 Sumber Data
    Twitter (X) - Tweet tentang kesehatan mental
    """)

# ============ FOOTER ============
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    © 2024 Kelompok 11 | Analisis Sentimen Kesehatan Mental
</div>
""", unsafe_allow_html=True)
