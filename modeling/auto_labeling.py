import re
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from indonesian_stopwords import get_all_stopwords

logger = logging.getLogger(__name__)


def get_topic_top_words(
    lda_model,
    topic_id: int,
    topn: int = 10
) -> List[Tuple[str, float]]:
    raw = lda_model.show_topic(topic_id, topn=topn)
    return raw


def label_with_tfidf(
    topic_id: int,
    top_words: List[str],
    corpus_texts: List[str],
    dictionary,
    all_stopwords: set,
    ngram_range: Tuple[int, int] = (1, 2)
) -> Dict:
    tokenized_docs = []
    for text in corpus_texts:
        tokens = [w for w in text.lower().split()
                  if w not in all_stopwords and len(w) > 2]
        tokenized_docs.append(" ".join(tokens))

    extra_stop = [w for w in all_stopwords]
    vectorizer = TfidfVectorizer(
        min_df=2,
        max_df=0.85,
        ngram_range=ngram_range,
        stop_words=extra_stop,
        max_features=2000,
        sublinear_tf=True
    )
    tfidf_matrix = vectorizer.fit_transform(tokenized_docs)
    feature_names = vectorizer.get_feature_names_out()

    query_terms = " ".join(top_words[:8])
    query_vec = vectorizer.transform([query_terms])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_doc_idx = np.argsort(scores)[-5:][::-1]

    candidate_ngrams = []
    for idx in top_doc_idx:
        row = tfidf_matrix[idx].toarray().flatten()
        top_ngram_idx = np.argsort(row)[-3:][::-1]
        for nid in top_ngram_idx:
            if row[nid] > 0:
                candidate_ngrams.append(feature_names[nid])
    candidate_ngrams = list(dict.fromkeys(candidate_ngrams))

    label_score = float(np.mean(scores[top_doc_idx])) if len(scores) > 0 else 0.0

    best_label = candidate_ngrams[0].title() if candidate_ngrams else top_words[0].title()

    return {
        "label_auto": best_label,
        "score": round(label_score, 4),
        "top_ngrams": candidate_ngrams[:5],
        "method": "tfidf"
    }


def evaluate_label_quality(
    top_words: List[str],
    label: str,
    model,
    dictionary
) -> Dict:
    label_tokens = label.lower().split()
    if not label_tokens:
        return {"coherence": 0.0, "word_overlap": 0.0, "coverage": 0.0}

    top_word_set = set(w.lower() for w, _ in top_words)
    label_word_set = set(label_tokens)

    overlap = len(top_word_set & label_word_set)
    word_overlap = overlap / len(label_word_set) if label_word_set else 0.0
    coverage = overlap / len(top_word_set) if top_word_set else 0.0

    lda_words_only = [(w, p) for w, p in top_words if w.lower() in label_word_set]
    coherence_weighted = sum(p for _, p in lda_words_only) / sum(p for _, p in top_words) if top_words else 0.0

    return {
        "coherence": round(coherence_weighted, 4),
        "word_overlap": round(word_overlap, 4),
        "coverage": round(coverage, 4)
    }


def generate_dynamic_label(
    topic_id: int,
    lda_model,
    topn: int = 5
) -> str:
    top_words = lda_model.show_topic(topic_id, topn=topn)
    phrases = [t[0].title() for t in top_words[:2]]

    if top_words:
        w0 = top_words[0][0].lower()
        w1 = top_words[1][0].lower() if len(top_words) > 1 else ""

        domain_map = {
            "sistem": "Sistem Informasi",
            "aplikasi": "Aplikasi",
            "analisis": "Analisis",
            "evaluasi": "Evaluasi",
            "pengaruh": "Pengaruh",
            "kualitas": "Kualitas Layanan",
            "kepuasan": "Kepuasan",
            "sentimen": "Analisis Sentimen",
            "klasifikasi": "Klasifikasi",
            "prediksi": "Prediksi",
            "cluster": "Clustering",
            "deteksi": "Deteksi",
            "rekomendasi": "Sistem Rekomendasi",
            "manajemen": "Manajemen",
            "monitor": "Monitoring",
            "pengelola": "Pengelolaan",
            "peminjam": "Peminjaman",
            "presensi": "Presensi",
            "absensi": "Presensi",
            "penjualan": "Penjualan",
            "pembelian": "Pembelian",
            "inventori": "Inventori",
            "stok": "Inventori",
            "keuangan": "Keuangan",
            "akademik": "Akademik",
            "perpustakaan": "Perpustakaan",
            "pegawai": "Kepegawaian",
            "kepegawaian": "Kepegawaian",
            "surat": "Manajemen Surat",
            "arsip": "Manajemen Arsip",
            "geografis": "Sistem Informasi Geografis",
            "pemetaan": "Sistem Informasi Geografis",
            "iot": "Internet of Things",
            "mobile": "Aplikasi Mobile",
            "android": "Aplikasi Android",
            "jaringan": "Jaringan Komputer",
            "keamanan": "Keamanan Sistem",
            "enkripsi": "Kriptografi",
            "kriptografi": "Kriptografi",
            "knn": "K-Nearest Neighbor",
            "naive": "Naive Bayes",
            "bayes": "Naive Bayes",
            "svm": "Support Vector Machine",
            "decision": "Decision Tree",
            "tree": "Decision Tree",
            "random": "Random Forest",
            "forest": "Random Forest",
            "neural": "Jaringan Syaraf Tiruan",
            "deep": "Deep Learning",
            "mining": "Data Mining",
            "apriori": "Apriori",
            "ahp": "AHP",
            "topsis": "TOPSIS",
            "saw": "SAW",
            "waterfall": "Waterfall",
            "prototype": "Prototyping",
            "agile": "Agile",
            "scrum": "Scrum",
            "extreme": "Extreme Programming",
            "programming": "Extreme Programming",
            "rad": "Rapid Application Development",
            "rup": "Rational Unified Process",
            "tam": "Technology Acceptance Model",
            "utaut": "UTAUT",
            "servqual": "ServQual",
            "webqual": "WebQual",
            "delone": "DeLone & McLean",
            "mclean": "DeLone & McLean",
            "cobit": "COBIT",
            "iso": "ISO",
        }
        matched = [v for k, v in domain_map.items() if k in w0 or k in w1]
        if matched:
            return matched[0]

    label = " & ".join(phrases) if phrases else f"Topik {topic_id + 1}"
    return label


def auto_label_all_topics(
    lda_model,
    corpus_texts: List[str],
    dictionary,
    all_stopwords: Optional[set] = None,
    method: str = "tfidf"
) -> Dict[int, Dict]:
    if all_stopwords is None:
        all_stopwords = get_all_stopwords()

    num_topics = lda_model.num_topics
    results = {}

    for topic_id in range(num_topics):
        top_words = get_topic_top_words(lda_model, topic_id, topn=15)

        if method == "tfidf":
            label_info = label_with_tfidf(
                topic_id,
                [w for w, _ in top_words],
                corpus_texts,
                dictionary,
                all_stopwords
            )
        else:
            label_info = {"label_auto": generate_dynamic_label(topic_id, lda_model),
                          "score": 0.0, "top_ngrams": [], "method": "dynamic"}

        quality = evaluate_label_quality(top_words, label_info["label_auto"],
                                         lda_model, dictionary)

        label_info["quality"] = quality
        label_info["top_words"] = [w for w, _ in top_words[:10]]
        label_info["method"] = method

        results[topic_id] = label_info
        logger.info(f"Topik {topic_id + 1}: {label_info['label_auto']} "
                    f"(score={label_info['score']}, quality={quality['coherence']})")

    return results
