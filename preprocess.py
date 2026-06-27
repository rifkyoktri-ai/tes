import logging
import sys
import pandas as pd
import pickle
import os
import re
from datetime import datetime

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from indonesian_stopwords import get_all_stopwords

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'logs/preprocess_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

PROSES_DIR = 'data/intermediate'
os.makedirs(PROSES_DIR, exist_ok=True)
os.makedirs('logs', exist_ok=True)

logger.info("=" * 60)
logger.info("PREPROCESSING DATASET 2021-2025")
logger.info("=" * 60)

logger.info("\n[1/4] Memuat data...")
df = pd.read_csv('data/processed/thesis_for_pipeline.csv')
logger.info(f"  Total: {len(df)}")
logger.info(f"  Dengan abstrak: {df['abstract'].notna().sum()}")

df['teks_gabung'] = df.apply(
    lambda r: (str(r['title']) + ' ' + str(r['abstract'])).strip()
    if pd.notna(r['abstract']) and str(r['abstract']).strip()
    else str(r['title']),
    axis=1
)

logger.info(f"\n[2/4] Membersihkan teks...")

factory = StemmerFactory()
stemmer = factory.create_stemmer()
all_stopwords = get_all_stopwords()

def bersihkan(teks):
    if not isinstance(teks, str):
        return ''
    teks = teks.lower()
    teks = re.sub(r'[^a-zA-Z\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks

def tokenize(teks):
    return [w for w in teks.split() if w not in all_stopwords and len(w) > 2]

def stem_tokens(tokens):
    return [stemmer.stem(t) for t in tokens]

df['teks_bersih'] = df['teks_gabung'].apply(bersihkan)
df['tokens_raw'] = df['teks_bersih'].apply(tokenize)
df['tokens'] = df['tokens_raw'].apply(stem_tokens)

logger.info(f"\n[3/4] Filter extremes...")
df = df[df['tokens'].apply(len) > 0].reset_index(drop=True)
logger.info(f"  Setelah filter dokumen kosong: {len(df)}")

from gensim import corpora
dictionary = corpora.Dictionary(df['tokens'].tolist())
dictionary.filter_extremes(no_below=2, no_above=0.80)
logger.info(f"  Vocabulary: {len(dictionary)} kata unik")

DICT_PATH = f'{PROSES_DIR}/dictionary.gensim'
CORPUS_PATH = f'{PROSES_DIR}/corpus.pkl'

dictionary.save(DICT_PATH)
corpus = [dictionary.doc2bow(doc) for doc in df['tokens'].tolist()]
with open(CORPUS_PATH, 'wb') as f:
    pickle.dump(corpus, f)
logger.info(f"  Dictionary, corpus saved")

df_final = pd.DataFrame({
    'Nama': '',
    'Judul': df['title'],
    'Abstrak': df['abstract'].fillna(''),
    'Tahun': df['year'],
    'teks_gabung': df['teks_gabung'],
    'teks_bersih': df['teks_bersih'],
    'tokens': df['tokens'].apply(lambda x: str(x))
})

df_final.to_csv(f'{PROSES_DIR}/dataset_preprocessed.csv', index=False)

logger.info(f"\n[4/4] Selesai!")
logger.info(f"  Output: {PROSES_DIR}/dataset_preprocessed.csv")
logger.info(f"  Total dokumen: {len(df_final)}")
logger.info(f"  Vocabulary: {len(dictionary)} kata unik")

word_freq = [(dictionary[id], dictionary.dfs.get(id, 0)) for id in dictionary]
word_freq.sort(key=lambda x: -x[1])
logger.info(f"\n  Top 20 kata:")
for w, f in word_freq[:20]:
    logger.info(f"    {w}: {f}")
