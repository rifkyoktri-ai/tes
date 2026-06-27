import argparse
import logging
import os
import sys
import pickle
import ast
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

from indonesian_stopwords import get_all_stopwords

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'logs/optimize_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

PROSES_DIR = 'data/intermediate'
MODEL_DIR = 'model'


def load_preprocessed_data() -> Tuple[pd.DataFrame, corpora.Dictionary, List]:
    logger.info("Memuat data preprocessed...")
    df = pd.read_csv(f'{PROSES_DIR}/dataset_preprocessed.csv')
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    tokenized_docs = df['tokens'].tolist()

    dict_path = f'{PROSES_DIR}/dictionary.gensim'
    corpus_path = f'{PROSES_DIR}/corpus.pkl'
    if os.path.exists(dict_path) and os.path.exists(corpus_path):
        dictionary = corpora.Dictionary.load(dict_path)
        with open(corpus_path, 'rb') as f:
            corpus = pickle.load(f)
        logger.info(f"Dictionary: {len(dictionary)} kata, Corpus: {len(corpus)} dokumen")
    else:
        logger.info("Dictionary/corpus tidak ditemukan, rebuild...")
        dictionary = corpora.Dictionary(tokenized_docs)
        dictionary.filter_extremes(no_below=2, no_above=0.85)
        corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
        dictionary.save(dict_path)
        with open(corpus_path, 'wb') as f:
            pickle.dump(corpus, f)
        logger.info(f"Dictionary: {len(dictionary)} kata unik")

    return df, dictionary, corpus, tokenized_docs


def train_and_evaluate(
    corpus,
    dictionary,
    tokenized_docs,
    k: int,
    passes: int = 20,
    iterations: int = 400,
    random_state: int = 42
) -> Tuple[LdaModel, float, float]:
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        alpha='auto',
        eta='auto',
        passes=passes,
        iterations=iterations,
        random_state=random_state,
        per_word_topics=True
    )
    cm = CoherenceModel(
        model=model,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence='c_v',
        processes=1
    )
    coherence = cm.get_coherence()
    perplexity = model.log_perplexity(corpus)
    return model, coherence, perplexity


def plot_optimization_results(results_df: pd.DataFrame):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color_cv = '#2E86AB'
    color_pp = '#A23B72'

    ax1.set_xlabel('Jumlah Topik (K)', fontsize=12)
    ax1.set_ylabel('Coherence Score (c_v)', fontsize=12, color=color_cv)
    line1 = ax1.plot(results_df['K'], results_df['coherence'],
                     marker='o', color=color_cv, linewidth=2, markersize=8,
                     label='Coherence (c_v)')
    ax1.tick_params(axis='y', labelcolor=color_cv)
    ax1.axhline(y=results_df['coherence'].max(), color=color_cv,
                linestyle='--', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Log Perplexity', fontsize=12, color=color_pp)
    line2 = ax2.plot(results_df['K'], results_df['perplexity'],
                     marker='s', color=color_pp, linewidth=2, markersize=8,
                     label='Log Perplexity')
    ax2.tick_params(axis='y', labelcolor=color_pp)

    best_k = results_df.loc[results_df['coherence'].idxmax(), 'K']
    ax1.axvline(x=best_k, color='green', linestyle=':', alpha=0.7,
                label=f'K optimal={best_k}')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')

    plt.title('Optimasi Jumlah Topik LDA: Coherence vs Perplexity',
              fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(f'{MODEL_DIR}/topic_optimization.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot optimization -> {MODEL_DIR}/topic_optimization.png")


def main():
    parser = argparse.ArgumentParser(description='Optimasi jumlah topik LDA')
    parser.add_argument('--min-k', type=int, default=5,
                        help='Jumlah topik minimum (default: 5)')
    parser.add_argument('--max-k', type=int, default=25,
                        help='Jumlah topik maksimum (default: 25)')
    parser.add_argument('--step', type=int, default=1,
                        help='Langkah K (default: 1)')
    parser.add_argument('--passes', type=int, default=20,
                        help='Jumlah passes LDA (default: 20)')
    parser.add_argument('--iterations', type=int, default=400,
                        help='Jumlah iterasi LDA (default: 400)')
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    logger.info("=" * 60)
    logger.info("OPTIMASI JUMLAH TOPIK LDA")
    logger.info(f"Rentang K: {args.min_k} - {args.max_k} (step={args.step})")
    logger.info("=" * 60)

    df, dictionary, corpus, tokenized_docs = load_preprocessed_data()

    results = []
    k_range = range(args.min_k, args.max_k + 1, args.step)

    for k in k_range:
        logger.info(f"Training K={k}...")
        try:
            model, coherence, perplexity = train_and_evaluate(
                corpus, dictionary, tokenized_docs, k,
                passes=args.passes, iterations=args.iterations
            )
            results.append({
                'K': k,
                'coherence': round(coherence, 4),
                'perplexity': round(perplexity, 4)
            })
            logger.info(f"  K={k}: coherence={coherence:.4f}, perplexity={perplexity:.4f}")
        except Exception as e:
            logger.error(f"  K={k} gagal: {e}")

    if not results:
        logger.error("Tidak ada hasil. Ada error pada training.")
        sys.exit(1)

    results_df = pd.DataFrame(results)
    results_path = f'{MODEL_DIR}/topic_optimization_results.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"Hasil disimpan -> {results_path}")

    best_idx = results_df['coherence'].idxmax()
    best_k = int(results_df.loc[best_idx, 'K'])
    best_coherence = results_df.loc[best_idx, 'coherence']
    best_perplexity = results_df.loc[best_idx, 'perplexity']

    logger.info("=" * 60)
    logger.info("REKOMENDASI K OPTIMAL")
    logger.info(f"  K={best_k}: coherence={best_coherence}, perplexity={best_perplexity}")
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("REKOMENDASI K OPTIMAL")
    print(f"  K = {best_k}")
    print(f"  Coherence (c_v) = {best_coherence}")
    print(f"  Log Perplexity  = {best_perplexity}")
    print("=" * 60)

    top5 = results_df.nlargest(5, 'coherence')
    print(f"\nTop 5 K terbaik:")
    for _, row in top5.iterrows():
        print(f"  K={int(row['K'])}: coherence={row['coherence']:.4f}, "
              f"perplexity={row['perplexity']:.4f}")

    plot_optimization_results(results_df)

    logger.info("Optimasi selesai!")


if __name__ == '__main__':
    main()
