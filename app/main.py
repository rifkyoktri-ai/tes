import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="LDA Topic Modeling Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    base_path = Path(__file__).parent.parent
    
    try:
        # Load evaluation metrics
        metrics = pd.read_csv(base_path / "model" / "evaluation_metrics.csv")
        
        # Load topic distribution
        topic_dist = pd.read_csv(base_path / "model" / "topic_distribution.csv")
        
        # Load preprocessed dataset
        dataset = pd.read_csv(base_path / "data" / "intermediate" / "dataset_preprocessed.csv")
        
        return metrics, topic_dist, dataset
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Load topic labels
@st.cache_data
def load_topic_labels():
    base_path = Path(__file__).parent.parent
    labels_path = base_path / "model" / "topic_labels.csv"
    try:
        if labels_path.exists():
            labels_df = pd.read_csv(labels_path)
            # Ensure new columns exist with defaults
            for col in ['label_score', 'quality_coherence']:
                if col not in labels_df.columns:
                    labels_df[col] = ''
            return labels_df
        else:
            return pd.DataFrame({
                'topic_id': range(5),
                'label': [f'Topic {i}' for i in range(5)],
                'description': [''] * 5,
                'keywords': [''] * 5,
                'label_score': [''] * 5,
                'quality_coherence': [''] * 5
            })
    except Exception as e:
        st.warning(f"Could not load topic labels: {e}")
        return pd.DataFrame({
            'topic_id': range(5),
            'label': [f'Topic {i}' for i in range(5)],
            'description': [''] * 5,
            'keywords': [''] * 5,
            'label_score': [''] * 5,
            'quality_coherence': [''] * 5
        })

# Load LDA visualization HTML
@st.cache_data
def load_lda_viz():
    base_path = Path(__file__).parent.parent
    viz_path = base_path / "model" / "lda_visualization.html"
    if viz_path.exists():
        with open(viz_path, 'r', encoding='utf-8') as f:
            return f.read()
    return None

# Display HTML with proper rendering
def display_html(html_content):
    """Display HTML content safely"""
    try:
        st.html(html_content)
    except Exception as e:
        st.error(f"Error: {e}")
        st.write("💡 Untuk melihat visualisasi, silakan buka file HTML langsung: `model/lda_visualization.html`")

# Title and Header
st.title("📊 LDA Topic Modeling Dashboard")
st.markdown("---")

# Load all data
metrics_df, topic_dist_df, dataset_df = load_data()
lda_viz_html = load_lda_viz()
topic_labels_df = load_topic_labels()

# Check if data loaded successfully
if metrics_df is None or topic_dist_df is None:
    st.error("❌ Gagal memuat data. Pastikan file CSV ada di folder yang benar.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("🎛️ Navigation")
    page = st.radio(
        "Pilih Halaman:",
        ["📈 Overview", "🔵 Visualisasi LDA", "📊 Model Metrics", "🏷️ Topic Analysis", "🔍 Document Search", "📖 Data Info", "⚙️ Manage Labels"]
    )

# PAGE 1: OVERVIEW
if page == "📈 Overview":
    st.header("Overview Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📚 Total Documents", len(topic_dist_df))
    
    with col2:
        metrics_dict = dict(zip(metrics_df['Metrik'], metrics_df['Nilai']))
        st.metric("🎯 Jumlah Topics", int(metrics_dict.get('Jumlah Topik (K)', 0)))
    
    with col3:
        coherence = metrics_dict.get('Coherence Score (CV)', 0)
        st.metric("✅ Coherence Score", f"{coherence:.4f}")
    
    with col4:
        perplexity = metrics_dict.get('Log Perplexity', 0)
        st.metric("📉 Log Perplexity", f"{perplexity:.4f}")
    
    st.markdown("---")
    
    # Dominant topic distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribusi Topik Dominan")
        topic_counts = topic_dist_df['topik_dominan'].value_counts().sort_index()
        fig = px.bar(
            x=topic_counts.index,
            y=topic_counts.values,
            labels={'x': 'Topic ID', 'y': 'Jumlah Documents'},
            title="Jumlah Documents per Topic",
            color=topic_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("Distribusi Probabilitas Topik Dominan")
        fig = px.histogram(
            topic_dist_df['prob_dominan'],
            nbins=30,
            title="Distribution Probability Topik Dominan",
            labels={'x': 'Probability', 'count': 'Frequency'},
            color_discrete_sequence=['#636EFA']
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # Year distribution
    st.subheader("📅 Distribusi Skripsi per Tahun")
    year_dist = topic_dist_df['Tahun'].value_counts().sort_index()
    fig = px.line(
        x=year_dist.index,
        y=year_dist.values,
        title="Timeline Skripsi",
        labels={'x': 'Tahun', 'y': 'Jumlah Skripsi'},
        markers=True
    )
    fig.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig, width='stretch')


# PAGE 2: VISUALISASI LDA
elif page == "🔵 Visualisasi LDA":
    st.header("🔵 Visualisasi LDA Model - Topic Distribution")
    
    st.markdown("""
    ### Visualisasi Interaktif Topic LDA
    Menampilkan distribusi dan karakteristik 14 topik yang diekstrak dari model LDA.
    """)
    
    # Create topic visualization data
    topic_stats = topic_dist_df.groupby('topik_dominan').agg({
        'prob_dominan': ['mean', 'count'],
        'Tahun': ['min', 'max']
    }).reset_index()
    
    topic_stats.columns = ['topic_id', 'avg_prob', 'doc_count', 'year_min', 'year_max']
    topic_stats['topic_id'] = topic_stats['topic_id'].astype(int)
    topic_stats['year_range'] = topic_stats['year_max'] - topic_stats['year_min']
    
    # Merge dengan topic labels
    topic_stats = topic_stats.merge(
        topic_labels_df[['topic_id', 'label', 'description']],
        on='topic_id',
        how='left'
    )
    
    # Scatter plot - topics as circles
    fig_bubble = px.scatter(
        topic_stats,
        x='topic_id',
        y='avg_prob',
        size='doc_count',
        color='avg_prob',
        hover_name='label',
        hover_data={
            'topic_id': True,
            'doc_count': True,
            'avg_prob': ':.4f',
            'year_min': True,
            'year_max': True,
            'year_range': False,
            'description': True,
            'label': False
        },
        title="LDA Topics Distribution - Labeled Bubble Chart",
        labels={
            'topic_id': 'Topic ID',
            'avg_prob': 'Average Probability',
            'doc_count': 'Number of Documents'
        },
        color_continuous_scale='Viridis',
        size_max=60
    )
    
    fig_bubble.update_layout(
        height=500,
        hovermode='closest',
        font=dict(size=12)
    )
    fig_bubble.update_traces(marker=dict(line=dict(width=2, color='white')))
    
    st.plotly_chart(fig_bubble, width='stretch')
    
    st.markdown("---")
    
    # Topic statistics table
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Topic Statistics with Labels")
        display_table = topic_stats[['topic_id', 'label', 'doc_count', 'avg_prob', 'year_min', 'year_max']].copy()
        display_table.columns = ['Topic ID', 'Label', 'Documents', 'Avg Probability', 'First Year', 'Last Year']
        display_table = display_table.sort_values('Documents', ascending=False)
        st.dataframe(display_table, width='stretch', hide_index=True)
    
    with col2:
        st.subheader("📈 Summary")
        st.metric("Total Topics", len(topic_stats))
        st.metric("Avg Documents/Topic", f"{topic_stats['doc_count'].mean():.1f}")
        st.metric("Total Documents", topic_stats['doc_count'].sum())
    
    st.markdown("---")
    
    # PyLDAvis HTML visualization (if available)
    if lda_viz_html:
        st.subheader("📍 PyLDAvis Interactive Visualization")
        st.write("*Klik pada topik di visualisasi untuk melihat top terms*")
        try:
            with st.container(height=900):
                st.html(lda_viz_html)
        except Exception as e:
            st.info("💡 Untuk melihat visualisasi PyLDAvis interaktif, silakan buka file: `model/lda_visualization.html`")
            with st.expander("📚 Tentang PyLDAvis"):
                st.markdown("""
                **PyLDAvis** menampilkan:
                - **Lingkaran**: Setiap topik (ukuran = jumlah dokumen)
                - **Jarak antar lingkaran**: Similarity antar topik
                - **Top Terms**: Terms paling relevan untuk setiap topik
                
                Visualisasi ini membantu memahami struktur topik dan hubungan antar topik.
                """)
    
    st.markdown("---")
    
    with st.expander("📚 Penjelasan Visualisasi"):
        st.markdown("""
        ### Bubble Chart Visualization
        - **X-Axis**: Topic ID (1-14)
        - **Y-Axis**: Average Probability (semakin tinggi = topik lebih dominan)
        - **Bubble Size**: Jumlah dokumen dalam topik (semakin besar = topik lebih sering muncul)
        - **Color**: Probability distribution (warna terang = probability tinggi)
        
        ### Cara Membaca:
        1. Hover di atas bubble untuk melihat detail topik
        2. Ukuran bubble menunjukkan popularitas topik
        3. Posisi vertikal menunjukkan kepercayaan model terhadap topik
        4. Warna menunjukkan intensitas probability
        """)


# PAGE 3: MODEL METRICS
elif page == "📊 Model Metrics":
    st.header("Model Evaluation Metrics")
    
    st.subheader("Metrics Summary")
    metrics_dict = dict(zip(metrics_df['Metrik'], metrics_df['Nilai']))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Number of Topics")
        st.info(f"**{int(metrics_dict['Jumlah Topik (K)'])}** Topics")
        st.caption("Jumlah topik yang digunakan dalam model LDA")
    
    with col2:
        st.markdown("### Coherence Score")
        coherence = metrics_dict['Coherence Score (CV)']
        st.info(f"**{coherence:.4f}**")
        st.caption("Semakin tinggi semakin baik (0-1)")
        if coherence > 0.5:
            st.success("✅ Good coherence score")
        else:
            st.warning("⚠️ Moderate coherence score")
    
    with col3:
        st.markdown("### Log Perplexity")
        perplexity = metrics_dict['Log Perplexity']
        st.info(f"**{perplexity:.4f}**")
        st.caption("Semakin negatif semakin baik")
    
    st.markdown("---")
    
    st.subheader("📋 Detailed Metrics Table")
    st.dataframe(metrics_df, width='stretch', hide_index=True)
    
    st.markdown("---")
    
    # Explanation
    with st.expander("📚 Penjelasan Metrics"):
        st.markdown("""
        - **Coherence Score**: Mengukur seberapa terkoheresi topik-topik yang dihasilkan.
          Nilai mendekati 1 berarti topik lebih meaningful.
        
        - **Log Perplexity**: Mengukur performa model dalam memprediksi data baru.
          Nilai yang lebih rendah (lebih negatif) menunjukkan model lebih baik.
        
        - **Jumlah Topik**: Jumlah topik yang diekstrak dari corpus dokumen.
        """)


# PAGE 3: TOPIC ANALYSIS
elif page == "🏷️ Topic Analysis":
    st.header("Topic Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Top Topics by Document Count (with Labels)")
        topic_counts = topic_dist_df['topik_dominan'].value_counts().head(10).sort_values()
        
        # Merge dengan labels
        topic_counts_df = pd.DataFrame({
            'topic_id': topic_counts.index,
            'count': topic_counts.values
        }).merge(topic_labels_df[['topic_id', 'label']], on='topic_id', how='left')
        
        fig = px.bar(
            topic_counts_df,
            y='label',
            x='count',
            orientation='h',
            title="Top 10 Topics (Most Documents)",
            labels={'count': 'Number of Documents', 'label': 'Topic Label'},
            color='count',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400, showlegend=False, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("Quick Stats")
        st.metric("Total Topics", len(topic_dist_df['topik_dominan'].unique()))
        st.metric("Avg Prob", f"{topic_dist_df['prob_dominan'].mean():.4f}")
        st.metric("Max Prob", f"{topic_dist_df['prob_dominan'].max():.4f}")
        st.metric("Min Prob", f"{topic_dist_df['prob_dominan'].min():.4f}")
    
    st.markdown("---")

    # Coherence quality per topic
    st.subheader("Kualitas Label per Topik")
    if 'quality_coherence' in topic_labels_df.columns and topic_labels_df['quality_coherence'].astype(str).str.strip().str.len().sum() > 0:
        quality_df = topic_labels_df[['topic_id', 'label', 'label_score', 'quality_coherence']].copy()
        quality_df['quality_coherence'] = pd.to_numeric(quality_df['quality_coherence'], errors='coerce')
        quality_df['label_score'] = pd.to_numeric(quality_df['label_score'], errors='coerce')
        quality_df = quality_df.sort_values('quality_coherence', ascending=False)

        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig_q = px.bar(
                quality_df.dropna(subset=['quality_coherence']),
                x='label', y='quality_coherence',
                title="Coherence Quality per Topik",
                labels={'label': 'Topik', 'quality_coherence': 'Quality Coherence'},
                color='quality_coherence',
                color_continuous_scale='RdYlGn',
                range_color=[0, 1]
            )
            fig_q.add_hline(y=0.4, line_dash="dash", line_color="red",
                          annotation_text="Threshold (0.4)")
            fig_q.update_layout(height=400)
            st.plotly_chart(fig_q, width='stretch')

        with col_b:
            st.metric("Rata-rata Quality", f"{quality_df['quality_coherence'].mean():.3f}")
            low_q = len(quality_df[quality_df['quality_coherence'] < 0.4])
            st.metric("Perlu Review (<0.4)", str(low_q))
            if low_q > 0:
                st.warning(f"⚠️ {low_q} topik punya coherence rendah")
            else:
                st.success("✅ Semua topik di atas threshold")

        st.markdown("---")
        with st.expander("📋 Detail Tabel Kualitas Label"):
            st.dataframe(quality_df, width='stretch', hide_index=True)
    else:
        st.info("Data kualitas label belum tersedia. Jalankan pipeline ulang dengan `--label-method tfidf` untuk mendapatkannya.")

    st.markdown("---")

    # Topic probability analysis
    st.subheader("Average Probability per Topic (Labeled)")
    avg_prob = topic_dist_df.groupby('topik_dominan')['prob_dominan'].mean().sort_values(ascending=False)
    
    # Merge dengan labels
    avg_prob_df = pd.DataFrame({
        'topic_id': avg_prob.index,
        'avg_probability': avg_prob.values
    }).merge(topic_labels_df[['topic_id', 'label']], on='topic_id', how='left')
    fig = px.bar(
        avg_prob_df,
        x='label',
        y='avg_probability',
        title="Average Dominant Topic Probability (Labeled)",
        labels={'label': 'Topic Label', 'avg_probability': 'Average Probability'},
        color='avg_probability',
        color_continuous_scale='Greens'
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    
    # Topic selection for detail
    st.subheader("Analisis Detail Topic")
    
    try:
        topic_list = sorted(topic_dist_df['topik_dominan'].unique().astype(int))
        selected_topic = st.selectbox(
            "Pilih Topic untuk detail:",
            topic_list,
            key="topic_select"
        )
        
        # Get topic label
        topic_label_row = topic_labels_df[topic_labels_df['topic_id'] == selected_topic]
        if len(topic_label_row) > 0:
            topic_label = topic_label_row.iloc[0]
            st.success(f"🏷️ **{topic_label['label']}**")
            if topic_label['description']:
                st.write(f"*{topic_label['description']}*")
            if topic_label['keywords']:
                st.caption(f"Keywords: {topic_label['keywords']}")
            # Quality indicators
            qc = topic_label.get('quality_coherence', '')
            ls = topic_label.get('label_score', '')
            if qc != '' and pd.notna(qc) and str(qc).strip():
                qc_val = float(qc)
                if qc_val < 0.4:
                    st.warning(f"⚠️ Kualitas label rendah (coherence={qc_val:.3f}) — perlu review manual")
                else:
                    st.success(f"✅ Kualitas label baik (coherence={qc_val:.3f})")
            if ls != '' and pd.notna(ls) and str(ls).strip():
                st.caption(f"Label Score: {float(ls):.4f}")
            st.markdown("---")
        
        topic_docs = topic_dist_df[topic_dist_df['topik_dominan'] == selected_topic]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents", len(topic_docs))
        with col2:
            st.metric("Avg Probability", f"{topic_docs['prob_dominan'].mean():.4f}")
        with col3:
            if len(topic_docs) > 0:
                st.metric("Year Range", f"{int(topic_docs['Tahun'].min())} - {int(topic_docs['Tahun'].max())}")
        
        # Display data with error handling
        if len(topic_docs) > 0:
            display_df = topic_docs[['Nama', 'Judul', 'Tahun', 'prob_dominan']].head(15).reset_index(drop=True)
            st.dataframe(display_df, width='stretch')
        else:
            st.info("No documents found for this topic.")
            
    except Exception as e:
        st.error(f"Error in topic analysis: {e}")


# PAGE 4: DOCUMENT SEARCH
elif page == "🔍 Document Search":
    st.header("Document Search & Filter")
    
    # Search options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_type = st.selectbox(
            "Cari berdasarkan:",
            ["Nama Penulis", "Judul", "Tahun", "Topic ID"],
            key="search_type"
        )
    
    # Initialize results as empty dataframe
    results = pd.DataFrame()
    
    with col2:
        try:
            if search_type == "Nama Penulis":
                search_term = st.text_input("Masukkan nama penulis:", key="nama_search")
                if search_term:
                    results = topic_dist_df[topic_dist_df['Nama'].str.contains(search_term, case=False, na=False)]
            
            elif search_type == "Judul":
                search_term = st.text_input("Masukkan keyword judul:", key="judul_search")
                if search_term:
                    results = topic_dist_df[topic_dist_df['Judul'].str.contains(search_term, case=False, na=False)]
            
            elif search_type == "Tahun":
                tahun_list = sorted(topic_dist_df['Tahun'].unique().astype(int))
                tahun = st.selectbox("Pilih Tahun:", tahun_list, key="tahun_search")
                results = topic_dist_df[topic_dist_df['Tahun'] == tahun]
            
            else:  # Topic ID
                topic_list = sorted(topic_dist_df['topik_dominan'].unique().astype(int))
                topic_id = st.selectbox("Pilih Topic ID:", topic_list, key="topic_search")
                results = topic_dist_df[topic_dist_df['topik_dominan'] == topic_id]
        
        except Exception as e:
            st.error(f"Error in search: {e}")
            results = pd.DataFrame()
    
    with col3:
        st.info(f"Found: **{len(results)}** documents")
    
    st.markdown("---")
    
    if len(results) > 0:
        st.subheader("Search Results")
        
        # Display options
        display_cols = ['Nama', 'Judul', 'Tahun', 'topik_dominan', 'prob_dominan']
        st.dataframe(
            results[display_cols].reset_index(drop=True),
            width='stretch',
            hide_index=True
        )
        
        # Export option
        if st.button("📥 Download Results as CSV"):
            csv = results.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="search_results.csv",
                mime="text/csv"
            )
    else:
        st.warning("❌ No documents found. Try a different search term.")


# PAGE 5: DATA INFO
elif page == "📖 Data Info":
    st.header("Data Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Dataset Statistics")
        st.metric("Total Documents", len(topic_dist_df))
        st.metric("Unique Authors", len(topic_dist_df['Nama'].unique()))
        st.metric("Year Range", f"{topic_dist_df['Tahun'].min():.0f} - {topic_dist_df['Tahun'].max():.0f}")
        st.metric("Unique Topics", len(topic_dist_df['topik_dominan'].unique()))
    
    with col2:
        st.subheader("📈 Probability Statistics")
        st.metric("Mean Probability", f"{topic_dist_df['prob_dominan'].mean():.4f}")
        st.metric("Median Probability", f"{topic_dist_df['prob_dominan'].median():.4f}")
        st.metric("Max Probability", f"{topic_dist_df['prob_dominan'].max():.4f}")
        st.metric("Min Probability", f"{topic_dist_df['prob_dominan'].min():.4f}")
    
    st.markdown("---")
    
    st.subheader("📋 Raw Data Preview")
    st.dataframe(topic_dist_df.head(20), width='stretch', hide_index=True)
    
    st.markdown("---")
    
    # Data description
    with st.expander("📚 Data Description"):
        st.markdown("""
        ### Dataset Columns:
        - **Nama**: Nama penulis skripsi
        - **Judul**: Judul lengkap skripsi
        - **Tahun**: Tahun penulisan skripsi
        - **topik_dominan**: ID topic dominan dari model LDA
        - **prob_dominan**: Probabilitas topic dominan
        
        ### Sumber Data:
        Dataset ini merupakan kumpulan skripsi Sistem Informasi yang telah dianalisis 
        menggunakan teknik LDA (Latent Dirichlet Allocation) untuk ekstraksi topik.
        """)


# PAGE 6: MANAGE LABELS
elif page == "⚙️ Manage Labels":
    st.header("⚙️ Kelola Label Topik")
    
    st.markdown("""
    Halaman ini memungkinkan Anda untuk mengedit dan menyimpan label interpretasi untuk setiap topik.
    Label akan ditampilkan di seluruh dashboard untuk memberikan konteks yang lebih jelas.
    """)
    
    st.markdown("---")
    
    # Initialize session state for labels
    if 'labels_edited' not in st.session_state:
        st.session_state.labels_edited = topic_labels_df.copy()
        # Ensure new columns exist
        for col in ['label_score', 'quality_coherence']:
            if col not in st.session_state.labels_edited.columns:
                st.session_state.labels_edited[col] = ''
    
    st.subheader("📝 Edit Topic Labels")
    
    # Ambil topic_id aktual dari DataFrame
    topic_ids = sorted(topic_labels_df['topic_id'].unique())
    num_topics = len(topic_ids)
    num_cols = 4
    num_tabs = (num_topics + num_cols - 1) // num_cols
    
    tab_list = [f"Topics {topic_ids[i*num_cols]}-{topic_ids[min((i+1)*num_cols-1, num_topics-1)]}" for i in range(num_tabs)]
    tabs = st.tabs(tab_list)
    
    for tab_idx, tab in enumerate(tabs):
        with tab:
            start_idx = tab_idx * num_cols
            end_idx = min((tab_idx + 1) * num_cols, num_topics)
            
            for idx in range(start_idx, end_idx):
                actual_topic_id = topic_ids[idx]
                topic_row = topic_labels_df[topic_labels_df['topic_id'] == actual_topic_id]
                
                if len(topic_row) == 0:
                    continue
                
                col1, col2 = st.columns([0.3, 0.7])
                
                with col1:
                    st.write(f"### Topic {actual_topic_id}")
                    topic_doc_count = len(topic_dist_df[topic_dist_df['topik_dominan'] == actual_topic_id])
                    st.caption(f"📊 {topic_doc_count} documents")
                
                with col2:
                    label = st.text_input(
                        "Label:",
                        value=topic_row['label'].values[0],
                        key=f"label_{actual_topic_id}"
                    )
                    
                    description = st.text_area(
                        "Deskripsi:",
                        value=topic_row['description'].values[0],
                        height=60,
                        key=f"desc_{actual_topic_id}"
                    )
                    
                    keywords = st.text_input(
                        "Keywords (pisahkan dengan `;`):",
                        value=topic_row['keywords'].values[0],
                        key=f"keywords_{actual_topic_id}"
                    )

                    # Show quality metrics if available
                    qc_val = topic_row['quality_coherence'].values[0] if 'quality_coherence' in topic_row.columns else ''
                    ls_val = topic_row['label_score'].values[0] if 'label_score' in topic_row.columns else ''
                    if qc_val != '' and pd.notna(qc_val) and str(qc_val).strip():
                        qc_float = float(qc_val)
                        if qc_float < 0.4:
                            st.warning(f"⚠️ Quality Coherence: {qc_float:.3f}")
                        else:
                            st.info(f"✅ Quality Coherence: {qc_float:.3f}")
                    if ls_val != '' and pd.notna(ls_val) and str(ls_val).strip():
                        st.caption(f"Label Score: {float(ls_val):.4f}")
                    
                    # Update session state
                    st.session_state.labels_edited.loc[
                        st.session_state.labels_edited['topic_id'] == actual_topic_id, 'label'
                    ] = label
                    st.session_state.labels_edited.loc[
                        st.session_state.labels_edited['topic_id'] == actual_topic_id, 'description'
                    ] = description
                    st.session_state.labels_edited.loc[
                        st.session_state.labels_edited['topic_id'] == actual_topic_id, 'keywords'
                    ] = keywords
                
                st.divider()
    
    st.markdown("---")
    
    # Save button
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Simpan Semua Label", use_container_width=True):
            try:
                base_path = Path(__file__).parent.parent
                labels_path = base_path / "model" / "topic_labels.csv"
                st.session_state.labels_edited.to_csv(labels_path, index=False)
                st.success("✅ Label berhasil disimpan!")
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error saat menyimpan: {e}")
    
    with col2:
        if st.button("🔄 Reset ke Label Awal", use_container_width=True):
            st.session_state.labels_edited = topic_labels_df.copy()
            st.info("Label direset ke kondisi awal")
            st.rerun()
    
    with col3:
        csv = st.session_state.labels_edited.to_csv(index=False)
        st.download_button(
            label="📥 Download Labels CSV",
            data=csv,
            file_name="topic_labels.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.markdown("---")
    
    st.subheader("📋 Preview Semua Label")
    st.dataframe(st.session_state.labels_edited, width='stretch', hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p style='color: gray; font-size: 12px;'>
    LDA Topic Modeling Dashboard | Data Science Project
    </p>
</div>
""", unsafe_allow_html=True)
