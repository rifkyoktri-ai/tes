"""
Auto Labeling Topik LDA menggunakan KeyBERT
Sumber label: top words dari model LDA (bukan dari judul dokumen)
Model: paraphrase-multilingual-MiniLM-L12-v2 (CPU-compatible)
"""

import json
import warnings
import sys
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')

from gensim.models import LdaModel
from indonesian_stopwords import get_all_stopwords

LABEL_MAPPING = {
    # Evaluasi & Pengukuran Sistem
    "end user computing satisfaction" : "End User Computing Satisfaction (EUCS)",
    "technology acceptance model"     : "Technology Acceptance Model (TAM)",
    "importance performance analysis" : "Importance Performance Analysis (IPA)",
    "delone mclean"                   : "Evaluasi Sistem DeLone & McLean",
    "delone dan mclean"               : "Evaluasi Sistem DeLone & McLean",
    "end user computing"              : "End User Computing Satisfaction (EUCS)",
    "webqual"                         : "Analisis Kualitas Website (WebQual)",
    "servqual"                        : "Analisis Kualitas Layanan (ServQual)",
    "iso 25010"                       : "Evaluasi Kualitas ISO 25010",
    "iso 9126"                        : "Evaluasi Kualitas ISO 9126",
    "kepuasan pengguna"               : "Analisis Kepuasan Pengguna",
    "kepuasan"                        : "Analisis Kepuasan Pengguna",
    "kualitas layanan"                : "Analisis Kualitas Layanan",
    "usability"                       : "Evaluasi Usabilitas Sistem",

    # Machine Learning & Data Mining
    "k-nearest neighbor"              : "K-Nearest Neighbor (KNN)",
    "nearest neighbor"                : "K-Nearest Neighbor (KNN)",
    "naive bayes"                     : "Klasifikasi Naive Bayes",
    "support vector machine"          : "Support Vector Machine (SVM)",
    "support vector"                  : "Support Vector Machine (SVM)",
    "decision tree"                   : "Decision Tree Classifier",
    "random forest"                   : "Random Forest Classifier",
    "neural network"                  : "Jaringan Syaraf Tiruan (ANN)",
    "deep learning"                   : "Deep Learning",
    "klasifikasi"                     : "Klasifikasi & Data Mining",
    "prediksi"                        : "Prediksi & Forecasting",
    "clustering"                      : "Clustering & Pengelompokan Data",
    "sentimen"                        : "Analisis Sentimen",
    "sentiment"                       : "Analisis Sentimen",
    "text mining"                     : "Text Mining & NLP",
    "data mining"                     : "Data Mining",

    # Metode Pengembangan Sistem
    "programming extreme"             : "Extreme Programming (XP)",
    "extreme programming"             : "Extreme Programming (XP)",
    "rapid application development"   : "Rapid Application Development (RAD)",
    "rapid application"               : "Rapid Application Development (RAD)",
    "prototyping"                     : "Metode Prototyping",
    "agile scrum"                     : "Agile Scrum",
    "bangun"                          : "Pengembangan Sistem Informasi",
    "mudah"                           : "Pengembangan Sistem Informasi",

    # Sistem Pendukung Keputusan
    "sistem pendukung keputusan"      : "Sistem Pendukung Keputusan (SPK)",
    "decision support"                : "Sistem Pendukung Keputusan (SPK)",
    "beasiswa"                        : "SPK Pemilihan Beasiswa",
    "topsis"                          : "SPK Metode TOPSIS",
    "ahp"                             : "SPK Metode AHP",
    "saw"                             : "SPK Metode SAW",

    # Domain Sistem Informasi Spesifik
    "sistem informasi manajemen"      : "Sistem Informasi Manajemen",
    "sistem informasi akademik"       : "Sistem Informasi Akademik",
    "sistem informasi keuangan"       : "Sistem Informasi Keuangan",
    "sistem informasi kepegawaian"    : "Sistem Informasi Kepegawaian",
    "sistem informasi perpustakaan"   : "Sistem Informasi Perpustakaan",
    "sistem informasi kesehatan"      : "Sistem Informasi Kesehatan",
    "rekam medis"                     : "Sistem Rekam Medis",
    "perpustakaan"                    : "Sistem Informasi Perpustakaan",
    "pustaka"                         : "Sistem Informasi Perpustakaan & Digital Library",
    "presensi"                        : "Sistem Presensi & Absensi",
    "absensi"                         : "Sistem Presensi & Absensi",
    "inventory"                       : "Sistem Manajemen Inventori",
    "persediaan"                      : "Sistem Manajemen Inventori",
    "penjualan"                       : "Sistem Informasi Penjualan",
    "kepegawaian"                     : "Sistem Informasi Kepegawaian",
    "pegawai"                         : "Sistem Informasi Kepegawaian",
    "surat"                           : "Sistem Manajemen Surat",
    "arsip"                           : "Sistem Manajemen Arsip",
    "monitoring"                      : "Sistem Monitoring & Evaluasi",

    # Audit & Keamanan
    "audit sistem"                    : "Audit Sistem Informasi",
    "cobit"                           : "Audit Sistem Informasi (COBIT)",
    "keamanan"                        : "Keamanan Sistem & Jaringan",
    "enkripsi"                        : "Kriptografi & Keamanan Data",
    "kriptografi"                     : "Kriptografi & Keamanan Data",

    # Teknologi Modern
    "internet of things"              : "Internet of Things (IoT)",
    "geographic information"          : "Geographic Information System (GIS)",
    "android"                         : "Pengembangan Aplikasi Android",
    "mobile"                          : "Aplikasi Mobile",
    "e-commerce"                      : "E-Commerce & Transaksi Online",
    "ecommerce"                       : "E-Commerce & Transaksi Online",
    "marketplace"                     : "E-Commerce & Marketplace",
    "hukum"                           : "Sistem Informasi Hukum",
    "syariah"                         : "Sistem Informasi Keuangan Syariah",
    "akademik"                        : "Sistem Informasi Akademik",
    "keuangan"                        : "Sistem Informasi Keuangan",
    "jaringan"                        : "Jaringan Komputer",
}


def apply_label_mapping(label_auto: str, top_words: list) -> str:
    search_text = (label_auto + " " + " ".join(top_words)).lower()
    sorted_patterns = sorted(LABEL_MAPPING.keys(), key=len, reverse=True)
    for pattern in sorted_patterns:
        if pattern.lower() in search_text:
            return LABEL_MAPPING[pattern]
    return label_auto.strip().title()


def label_topics_keybert(lda_model, all_stopwords: set) -> dict:
    print("\n" + "="*60)
    print("AUTO LABELING TOPIK LDA DENGAN KEYBERT")
    print("="*60)
    print(f"Jumlah topik: {lda_model.num_topics}")
    print("Model: paraphrase-multilingual-MiniLM-L12-v2 (CPU)")
    print("Estimasi waktu: 10-20 menit (CPU only)")
    print("="*60)

    print("\n[1/3] Loading KeyBERT model...")
    print("      Download ~500MB jika belum ada di cache...")
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer
    sentence_model = SentenceTransformer(
        'paraphrase-multilingual-MiniLM-L12-v2'
    )
    kw_model = KeyBERT(model=sentence_model)
    print("      KeyBERT model loaded.")

    print("\n[2/3] Memproses setiap topik...")
    topic_labels = {}

    for topic_id in range(lda_model.num_topics):
        print(f"\n  Topik {topic_id + 1}/{lda_model.num_topics}:")

        raw_words = lda_model.show_topic(topic_id, topn=25)

        filtered_words = [
            word for word, weight in raw_words
            if word.lower() not in all_stopwords
            and len(word) > 2
            and not word.isdigit()
            and word.isalpha()
        ]

        print(f"  Raw words (top 10): {[w for w,_ in raw_words[:10]]}")
        print(f"  Filtered words: {filtered_words[:10]}")

        if len(filtered_words) < 3:
            print(f"  Filtered words < 3, gunakan fallback label")
            topic_labels[str(topic_id)] = {
                "label_auto"  : f"Topik {topic_id + 1}",
                "score"       : 0.0,
                "top_words"   : [w for w, _ in raw_words[:10]],
                "top_words_filtered": filtered_words,
                "label_final" : f"Topik Umum {topic_id + 1}"
            }
            continue

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
                best_score = float(keywords[0][1])
                print(f"  KeyBERT candidates: {[(k, round(s,3)) for k,s in keywords[:3]]}")
            else:
                best_label = " ".join(filtered_words[:3])
                best_score = 0.0
                print(f"  KeyBERT tidak menghasilkan kandidat")

        except Exception as e:
            print(f"  KeyBERT error: {e}")
            best_label = " ".join(filtered_words[:3])
            best_score = 0.0

        label_final = apply_label_mapping(best_label, filtered_words)

        print(f"  label_auto  : {best_label} (score: {best_score:.3f})")
        print(f"  label_final : {label_final}")

        topic_labels[str(topic_id)] = {
            "label_auto"        : best_label,
            "score"             : round(best_score, 4),
            "top_words"         : [w for w, _ in raw_words[:10]],
            "top_words_filtered": filtered_words[:10],
            "label_final"       : label_final
        }

    return topic_labels


def save_topic_labels(topic_labels: dict, output_dir: str = "model") -> None:
    output_path = Path(output_dir) / "topic_labels.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(topic_labels, f, ensure_ascii=False, indent=4)
    print(f"\n[3/3] Topic labels disimpan ke: {output_path}")


def run_auto_labeling():
    base_path = Path(__file__).parent

    model_path = base_path / "model" / "lda_model.gensim"
    if not model_path.exists():
        raise FileNotFoundError(
            f"LDA model tidak ditemukan: {model_path}\n"
            "Pastikan sudah menjalankan training LDA terlebih dahulu."
        )

    print(f"Loading LDA model dari: {model_path}")
    lda_model = LdaModel.load(str(model_path))
    print(f"Model loaded. Jumlah topik: {lda_model.num_topics}")

    all_stopwords = get_all_stopwords()
    print(f"Total stopwords: {len(all_stopwords)} kata")

    topic_labels = label_topics_keybert(lda_model, all_stopwords)

    save_topic_labels(topic_labels, str(base_path / "model"))

    print("\n" + "="*60)
    print("RINGKASAN HASIL LABELING")
    print("="*60)
    for tid, data in topic_labels.items():
        print(f"Topik {int(tid)+1}: {data['label_final']}")
        print(f"  Score: {data['score']:.4f}")
        print(f"  Words: {', '.join(data['top_words_filtered'][:5])}")
    print("="*60)

    return topic_labels


if __name__ == "__main__":
    run_auto_labeling()
