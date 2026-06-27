import argparse
import logging
import sys
import os
import pickle
import ast
import multiprocessing
from datetime import datetime
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

from indonesian_stopwords import get_all_stopwords

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'logs/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

PROSES_DIR = 'data/intermediate'
MODEL_DIR = 'model'


def get_dominant_topic(lda_model, corpus):
    dominant_topics = []
    for bow in corpus:
        topic_probs = lda_model.get_document_topics(bow, minimum_probability=0)
        dominant = max(topic_probs, key=lambda x: x[1])
        dominant_topics.append({
            'topik_dominan': dominant[0] + 1,
            'prob_dominan': round(dominant[1], 4)
        })
    return pd.DataFrame(dominant_topics)


def train_model(
    corpus,
    dictionary,
    num_topics: int,
    passes: int = 20,
    iterations: int = 400,
    alpha=None,
    eta=None
) -> LdaModel:
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        alpha=alpha or 'auto',
        eta=eta or 'auto',
        passes=passes,
        iterations=iterations,
        random_state=42,
        per_word_topics=True
    )
    return model


def auto_tune(
    corpus,
    dictionary,
    tokenized_docs,
    k_range: range = range(4, 13),
    passes: int = 20,
    iterations: int = 400,
    alpha=None,
    eta=None
) -> Tuple[int, float, LdaModel]:
    coherence_scores = []
    models = {}

    for k in k_range:
        logger.info(f"  Auto-tune K={k}...")
        model = train_model(corpus, dictionary, k, passes, iterations, alpha, eta)
        cm = CoherenceModel(
            model=model,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence='c_v',
            processes=1
        )
        cv = cm.get_coherence()
        models[k] = model
        coherence_scores.append((k, cv))
        logger.info(f"  K={k}: coherence={cv:.4f}")

    coherence_scores.sort(key=lambda x: -x[1])
    optimal_k, coherence_final = coherence_scores[0]
    logger.info(f"  K optimal: {optimal_k} (coherence={coherence_final:.4f})")

    return optimal_k, coherence_final, models[optimal_k]


def plot_topic_words(lda_model, optimal_k: int):
    n_cols = min(3, optimal_k)
    n_rows = (optimal_k + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
    axes = axes.flatten()
    COLORS = plt.cm.tab10.colors

    for i in range(optimal_k):
        top_words = lda_model.show_topic(i, topn=10)
        words = [w for w, _ in top_words]
        weights = [p for _, p in top_words]
        ax = axes[i]
        ax.barh(words[::-1], weights[::-1], color=COLORS[i % len(COLORS)])
        ax.set_title(f'Topik {i + 1}', fontweight='bold')
        ax.set_xlabel('Bobot')
        ax.grid(axis='x', linestyle='--', alpha=0.5)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/top_words_per_topic.png', dpi=150)
    plt.close()
    logger.info(f"  Plot top words -> {MODEL_DIR}/top_words_per_topic.png")


def plot_topic_distribution(df_result: pd.DataFrame, optimal_k: int):
    topic_counts = df_result['topik_dominan'].value_counts().sort_index()

    plt.figure(figsize=(10, 5))
    bars = plt.bar(
        [f'Topik {t}' for t in topic_counts.index],
        topic_counts.values,
        color=plt.cm.tab10.colors[:len(topic_counts)],
        edgecolor='white',
        linewidth=0.7
    )
    for bar, val in zip(bars, topic_counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.title(f'Distribusi Jumlah Dokumen per Topik (K={optimal_k})',
              fontsize=13, fontweight='bold')
    plt.xlabel('Topik')
    plt.ylabel('Jumlah Dokumen')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/distribusi_topik.png', dpi=150)
    plt.close()
    logger.info(f"  Plot distribusi topik -> {MODEL_DIR}/distribusi_topik.png")


def plot_topic_trend(df_result: pd.DataFrame):
    pivot = df_result.groupby(['Tahun', 'topik_dominan']).size().unstack(fill_value=0)

    plt.figure(figsize=(12, 6))
    sns.heatmap(
        pivot, annot=True, fmt='d', cmap='YlOrRd',
        linewidths=0.5, linecolor='grey',
        cbar_kws={'label': 'Jumlah Dokumen'}
    )
    plt.title('Distribusi Tren Topik per Tahun', fontsize=13, fontweight='bold')
    plt.xlabel('Topik')
    plt.ylabel('Tahun')
    plt.tight_layout()
    plt.savefig(f'{MODEL_DIR}/tren_topik_per_tahun.png', dpi=150)
    plt.close()
    logger.info(f"  Plot tren tahunan -> {MODEL_DIR}/tren_topik_per_tahun.png")


def main():
    parser = argparse.ArgumentParser(description='Pipeline LDA Topic Modeling')
    parser.add_argument('--num-topics', type=int, default=None,
                        help='Jumlah topik (default: auto-tune dari 4-12)')
    parser.add_argument('--passes', type=int, default=20,
                        help='Jumlah passes LDA (default: 20)')
    parser.add_argument('--iterations', type=int, default=400,
                        help='Jumlah iterasi LDA (default: 400)')
    parser.add_argument('--alpha', type=str, default=None,
                        help='Alpha parameter (default: auto)')
    parser.add_argument('--eta', type=str, default=None,
                        help='Eta parameter (default: auto)')
    parser.add_argument('--label-method', type=str, default='keybert',
                        choices=['keybert', 'tfidf', 'dynamic'],
                        help='Metode pelabelan (default: keybert)')
    parser.add_argument('--no-auto-tune', action='store_true',
                        help='Nonaktifkan auto-tune (wajib set --num-topics)')
    parser.add_argument('--min-k', type=int, default=4,
                        help='K minimum untuk auto-tune (default: 4)')
    parser.add_argument('--max-k', type=int, default=12,
                        help='K maksimum untuk auto-tune (default: 12)')
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    logger.info("=" * 60)
    logger.info("PIPELINE TOPIC MODELING LDA + LABELING")
    logger.info("=" * 60)

    logger.info(f"\n[1/6] Memuat data preprocessed...")
    df = pd.read_csv(f'{PROSES_DIR}/dataset_preprocessed.csv')
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    tokenized_docs = df['tokens'].tolist()
    logger.info(f"  Dokumen: {len(df)}")

    logger.info(f"\n[2/6] Memuat dictionary & corpus...")
    dict_path = f'{PROSES_DIR}/dictionary.gensim'
    corpus_path = f'{PROSES_DIR}/corpus.pkl'

    if os.path.exists(dict_path) and os.path.exists(corpus_path):
        dictionary = corpora.Dictionary.load(dict_path)
        with open(corpus_path, 'rb') as f:
            corpus = pickle.load(f)
        logger.info(f"  Dictionary: {len(dictionary)} kata unik")
        logger.info(f"  Corpus: {len(corpus)} dokumen")
    else:
        logger.info("  File tidak ditemukan, rebuild...")
        dictionary = corpora.Dictionary(tokenized_docs)
        dictionary.filter_extremes(no_below=2, no_above=0.85)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
        dictionary.save(dict_path)
        with open(corpus_path, 'wb') as f:
            pickle.dump(corpus, f)
        logger.info(f"  Dictionary: {len(dictionary)} kata unik")

    logger.info(f"\n[3/6] Training model LDA...")

    if args.no_auto_tune or args.num_topics is not None:
        k = args.num_topics or 5
        logger.info(f"  Manual K={k} (auto-tune disabled)")
        lda_model = train_model(
            corpus, dictionary, k,
            passes=args.passes, iterations=args.iterations,
            alpha=args.alpha, eta=args.eta
        )
        OPTIMAL_K = k
        cm = CoherenceModel(
            model=lda_model, texts=tokenized_docs,
            dictionary=dictionary, coherence='c_v', processes=1
        )
        coherence_final = cm.get_coherence()
        perplexity_final = lda_model.log_perplexity(corpus)
        logger.info(f"  K={k}: coherence={coherence_final:.4f}, "
                    f"perplexity={perplexity_final:.4f}")
    else:
        logger.info(f"  Auto-tune K ({args.min_k}-{args.max_k})...")
        k_range = range(args.min_k, args.max_k + 1)
        OPTIMAL_K, coherence_final, lda_model = auto_tune(
            corpus, dictionary, tokenized_docs,
            k_range=k_range,
            passes=args.passes, iterations=args.iterations,
            alpha=args.alpha, eta=args.eta
        )
        perplexity_final = lda_model.log_perplexity(corpus)

    plot_topic_words(lda_model, OPTIMAL_K)

    logger.info(f"\n[4/6] Menentukan topik dominan & labeling...")

    topic_df = get_dominant_topic(lda_model, corpus)
    df_result = pd.concat([df[['Judul', 'Tahun']].reset_index(drop=True), topic_df], axis=1)
    df_result.insert(0, 'Nama', '')

    plot_topic_distribution(df_result, OPTIMAL_K)
    plot_topic_trend(df_result)

    all_stopwords = get_all_stopwords()

    if args.label_method == 'tfidf':
        from modeling.auto_labeling import auto_label_all_topics
        logger.info("  Labeling method: TF-IDF")
        topic_labels = auto_label_all_topics(
            lda_model,
            df['Judul'].tolist(),
            dictionary,
            all_stopwords,
            method='tfidf'
        )
    elif args.label_method == 'dynamic':
        from modeling.auto_labeling import auto_label_all_topics
        logger.info("  Labeling method: Dynamic")
        topic_labels = auto_label_all_topics(
            lda_model,
            df['Judul'].tolist(),
            dictionary,
            all_stopwords,
            method='dynamic'
        )
    else:
        from modeling.keybert_labeler import label_topics_with_keybert
        logger.info("  Labeling method: KeyBERT")
        topic_labels = label_topics_with_keybert(
            lda_model, all_stopwords,
            df_result=df_result,
            text_column='Judul'
        )

    labels_rows = []
    for tid_str, info in topic_labels.items():
        tid = int(tid_str) + 1
        doc_count = len(df_result[df_result['topik_dominan'] == tid])
        quality_str = info.get('quality', {}).get('coherence', '')
        labels_rows.append({
            'topic_id': tid,
            'label': info['label_final'] if 'label_final' in info else info['label_auto'],
            'description': f"Topic with {doc_count} documents",
            'keywords': ';'.join(info.get('top_words', info.get('top_ngrams', []))),
            'label_score': info.get('score', ''),
            'quality_coherence': quality_str
        })

    labels_df = pd.DataFrame(labels_rows)
    labels_df.to_csv(f'{MODEL_DIR}/topic_labels.csv', index=False)
    logger.info(f"  topic_labels.csv -> {MODEL_DIR}/topic_labels.csv")
    for _, row in labels_df.iterrows():
        logger.info(f"  Topik {row['topic_id']:2d}: {row['label']}")

    logger.info(f"\n[5/6] Menyimpan model & visualisasi...")

    vis_data = gensimvis.prepare(
        lda_model, corpus, dictionary,
        mds='mmds',
        sort_topics=False
    )
    pyLDAvis.save_html(vis_data, f'{MODEL_DIR}/lda_visualization.html')
    logger.info(f"  LDA viz -> {MODEL_DIR}/lda_visualization.html")

    lda_model.save(f'{MODEL_DIR}/lda_model.gensim')
    df_result.to_csv(f'{MODEL_DIR}/topic_distribution.csv', index=False)

    eval_df = pd.DataFrame({
        'Metrik': ['Jumlah Topik (K)', 'Coherence Score (CV)', 'Log Perplexity'],
        'Nilai': [OPTIMAL_K, round(coherence_final, 4), round(perplexity_final, 4)]
    })
    eval_df.to_csv(f'{MODEL_DIR}/evaluation_metrics.csv', index=False)

    logger.info(f"\n[6/6] Selesai!")
    logger.info(f"  Semua output disimpan ke folder {MODEL_DIR}/")
    logger.info(f"  K={OPTIMAL_K}, Coherence={coherence_final:.4f}, "
                f"Perplexity={perplexity_final:.4f}")
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("PIPELINE SELESAI")
    print(f"  K         = {OPTIMAL_K}")
    print(f"  Coherence = {coherence_final:.4f}")
    print(f"  Perplexity= {perplexity_final:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
