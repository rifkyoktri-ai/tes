import re
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from indonesian_stopwords import get_all_stopwords

LABEL_MAPPING = {
    "neighbor nearest"       : "K-Nearest Neighbor (KNN)",
    "nearest neighbor"       : "K-Nearest Neighbor (KNN)",
    "naive bayes"            : "Klasifikasi Naive Bayes",
    "support vector"         : "Support Vector Machine (SVM)",
    "decision tree"          : "Decision Tree Classifier",
    "random forest"          : "Random Forest",
    "neural network"         : "Jaringan Syaraf Tiruan (ANN)",
    "deep learning"          : "Deep Learning",
    "klasifikasi"            : "Klasifikasi & Data Mining",
    "prediksi"               : "Prediksi & Forecasting",
    "clustering"             : "Clustering & Pengelompokan",
    "sentimen"               : "Analisis Sentimen",
    "sentiment"              : "Analisis Sentimen",
    "mining"                 : "Data Mining",

    "extreme programming"    : "Extreme Programming (XP)",
    "rapid application"      : "Rapid Application Development (RAD)",
    "prototype"              : "Metode Prototyping",
    "agile"                  : "Agile Development",
    "scrum"                  : "Agile Scrum",

    "delone mclean"          : "Evaluasi Sistem DeLone & McLean",
    "technology acceptance"  : "Technology Acceptance Model (TAM)",
    "end user computing"     : "End User Computing Satisfaction (EUCS)",
    "eucs"                   : "End User Computing Satisfaction (EUCS)",
    "webqual"                : "Analisis Kualitas Website (WebQual)",
    "servqual"               : "Analisis Kualitas Layanan (ServQual)",
    "importance performance" : "Importance Performance Analysis (IPA)",
    "iso 9126"               : "Evaluasi Kualitas ISO 9126",
    "iso 25010"              : "Evaluasi Kualitas ISO 25010",
    "kepuasan"               : "Analisis Kepuasan Pengguna",
    "kualitas"               : "Analisis Kualitas Layanan",
    "usability"              : "Evaluasi Usabilitas Sistem",

    "beasiswa"               : "SPK Pemilihan Beasiswa",
    "spk"                    : "Sistem Pendukung Keputusan",
    "decision support"       : "Sistem Pendukung Keputusan",
    "ahp"                    : "SPK Metode AHP",
    "topsis"                 : "SPK Metode TOPSIS",
    "saw"                    : "SPK Metode SAW",

    "perpustakaan"           : "Sistem Informasi Perpustakaan",
    "library"                : "Sistem Informasi Perpustakaan",
    "akademik"               : "Sistem Informasi Akademik",
    "presensi"               : "Sistem Presensi & Absensi",
    "absensi"                : "Sistem Presensi & Absensi",
    "keuangan"               : "Sistem Informasi Keuangan",
    "inventory"              : "Sistem Manajemen Inventori",
    "stok"                   : "Sistem Manajemen Inventori",
    "kesehatan"              : "Sistem Informasi Kesehatan",
    "rekam medis"            : "Sistem Rekam Medis",
    "penjualan"              : "Sistem Informasi Penjualan",
    "peminjaman"             : "Sistem Informasi Peminjaman",
    "kepegawaian"            : "Sistem Informasi Kepegawaian",
    "pegawai"                : "Sistem Informasi Kepegawaian",
    "surat"                  : "Sistem Manajemen Surat",
    "arsip"                  : "Sistem Manajemen Arsip",
    "monitoring"             : "Sistem Monitoring",
    "audit"                  : "Audit Sistem Informasi",
    "cobit"                  : "Audit Sistem Informasi (COBIT)",

    "keamanan"               : "Keamanan Sistem & Jaringan",
    "enkripsi"               : "Keamanan & Kriptografi",
    "kriptografi"            : "Kriptografi & Enkripsi Data",
    "jaringan"               : "Jaringan Komputer",
    "network"                : "Jaringan Komputer",
    "firewall"               : "Keamanan Jaringan",

    "iot"                    : "Internet of Things (IoT)",
    "smart"                  : "Smart System & IoT",
    "gis"                    : "Geographic Information System (GIS)",
    "geographic"             : "Geographic Information System (GIS)",
    "mobile"                 : "Aplikasi Mobile",
    "android"                : "Pengembangan Aplikasi Android",
    "ecommerce"              : "E-Commerce & Transaksi Online",
    "toko online"            : "E-Commerce & Transaksi Online",
    "marketplace"            : "E-Commerce & Marketplace",
    "hukum"                  : "Sistem Informasi Hukum",
    "syariah"                : "Sistem Informasi Keuangan Syariah",

    # Tambahan pola untuk mencakup lebih banyak judul
    "tracer"                 : "Tracer Study & Alumni",
    "alumni"                 : "Tracer Study & Alumni",
    "simkah"                 : "Sistem Informasi Pelayanan Nikah",
    "nikah"                  : "Sistem Informasi Pelayanan Nikah",
    "octave"                 : "Manajemen Risiko Sistem Informasi",
    "allegro"                : "Manajemen Risiko Sistem Informasi",
    "risiko"                 : "Manajemen Risiko Sistem Informasi",
    "fmea"                   : "Manajemen Risiko (FMEA)",
    "nist"                   : "Audit & Keamanan Sistem Informasi",
    "hot fit"                : "Evaluasi Sistem Informasi (HOT-FIT)",
    "geografis"              : "Sistem Informasi Geografis (GIS)",
    "gis"                    : "Sistem Informasi Geografis (GIS)",
    "pemetaan"               : "Sistem Informasi Geografis (GIS)",
    "koperasi"               : "Sistem Informasi Koperasi",
    "k means"                : "K-Means Clustering",
    "kmeans"                 : "K-Means Clustering",
    "apriori"                : "Data Mining (Apriori)",
    "c45"                    : "Klasifikasi C4.5",
    "c4.5"                   : "Klasifikasi C4.5",
    "sp 800"                 : "Audit & Keamanan Sistem Informasi",
    "iso 27001"              : "Audit & Keamanan Sistem Informasi",
    "utaut"                  : "Technology Acceptance (UTAUT)",
    "tam"                    : "Technology Acceptance Model (TAM)",
    "task technology"        : "Task Technology Fit (TTF)",
    "ttf"                    : "Task Technology Fit (TTF)",
    "knn"                    : "K-Nearest Neighbor (KNN)",
    "expectation confirm"    : "Expectation Confirmation Model (ECM)",
    "ecm"                    : "Expectation Confirmation Model (ECM)",
    "waterfall"              : "Metode Waterfall",
    "unified approach"       : "Unified Approach (UA)",
    "rational unified"       : "Rational Unified Process (RUP)",
    "rup"                    : "Rational Unified Process (RUP)",
    "kualitatif"             : "Penelitian Kualitatif",
    "kuantitatif"            : "Penelitian Kuantitatif",
    "balanced scorecard"     : "IT Balanced Scorecard",
    "failure mode"           : "Manajemen Risiko (FMEA)",
    "octave allegro"         : "Manajemen Risiko (OCTAVE Allegro)",
    "codeigniter"            : "Framework CodeIgniter",
    "laravel"                : "Framework Laravel",
    "framework"              : "Pengembangan Sistem Informasi",
}

def apply_label_mapping(label_auto, top_words):
    combined = (label_auto + " " + " ".join(top_words)).lower()
    sorted_patterns = sorted(LABEL_MAPPING.keys(), key=len, reverse=True)
    for pattern in sorted_patterns:
        if re.search(r'\b' + re.escape(pattern) + r'\b', combined):
            return LABEL_MAPPING[pattern]
    return label_auto.title()

def scan_titles_for_label(titles, mapping, min_pct=0.10):
    """Find the most frequent LABEL_MAPPING label across document titles.
    Only returns label if at least min_pct of documents match it."""
    label_counts = {}
    for title in titles:
        title_lower = title.lower()
        sorted_patterns = sorted(mapping.keys(), key=len, reverse=True)
        for pattern in sorted_patterns:
            if re.search(r'\b' + re.escape(pattern) + r'\b', title_lower):
                label = mapping[pattern]
                label_counts[label] = label_counts.get(label, 0) + 1
                break
    if not label_counts:
        return None
    most_common = max(label_counts, key=label_counts.get)
    most_common_cnt = label_counts[most_common]
    if most_common_cnt / len(titles) >= min_pct:
        return most_common
    return None

def label_topics_with_keybert(lda_model, all_stopwords, df_result=None, text_column='Judul'):
    print("[KeyBERT] Loading model paraphrase-multilingual-MiniLM-L12-v2...")

    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    kw_model = KeyBERT(model=model)

    topic_labels = {}
    num_topics = lda_model.num_topics

    for topic_id in range(num_topics):
        print(f"[KeyBERT] Memproses Topik {topic_id + 1}/{num_topics}...")

        # Step 1: Rule-based scan on actual document titles (most reliable)
        rule_label = None
        if df_result is not None and text_column in df_result.columns:
            topic_docs = df_result[df_result['topik_dominan'] == (topic_id + 1)][text_column]
            rule_label = scan_titles_for_label(topic_docs.tolist(), LABEL_MAPPING)

        raw_words = lda_model.show_topic(topic_id, topn=20)

        filtered_words = [
            word for word, weight in raw_words
            if word.lower() not in all_stopwords
            and len(word) > 2
            and not word.isdigit()
        ]

        if len(filtered_words) < 3:
            label_final = rule_label or f"Topik {topic_id + 1}"
            topic_labels[str(topic_id)] = {
                "label_auto"  : f"Topik {topic_id + 1}",
                "score"       : 0.0,
                "top_words"   : filtered_words,
                "label_final" : label_final
            }
            continue

        # Build input for KeyBERT: prefer actual document titles, fallback to top words
        if df_result is not None and text_column in df_result.columns:
            topic_docs = df_result[df_result['topik_dominan'] == (topic_id + 1)][text_column]
            doc = " ".join(topic_docs.astype(str).tolist())
        else:
            doc = " ".join(filtered_words)

        try:
            keywords = kw_model.extract_keywords(
                doc,
                keyphrase_ngram_range=(2, 4),
                stop_words=list(all_stopwords),
                use_mmr=True,
                diversity=0.5,
                top_n=5
            )

            if keywords:
                best_label = keywords[0][0]
                best_score = keywords[0][1]
            else:
                best_label = " ".join(filtered_words[:3])
                best_score = 0.0

        except Exception as e:
            print(f"[KeyBERT] Warning Topik {topic_id}: {e}")
            best_label = " ".join(filtered_words[:3])
            best_score = 0.0

        label_final = apply_label_mapping(best_label, filtered_words)

        # Override: Use rule-based label if found (already filtered by >10% threshold)
        if rule_label:
            label_final = rule_label

        topic_labels[str(topic_id)] = {
            "label_auto"  : best_label,
            "score"       : round(best_score, 4),
            "top_words"   : filtered_words[:10],
            "label_final" : label_final
        }

        print(f"[KeyBERT] Topik {topic_id}: {label_final}")

    return topic_labels
