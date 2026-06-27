import nltk
try:
    from nltk.corpus import stopwords
    NLTK_STOPWORDS = set(stopwords.words('indonesian'))
except LookupError:
    NLTK_STOPWORDS = set()

CUSTOM_STOPWORDS_SI = {
    # ── KATA UMUM BAHASA INDONESIA (dasar) ──
    "yang", "dan", "di", "ke", "dari", "dengan", "untuk", "pada",
    "ini", "itu", "adalah", "sebagai", "dalam", "oleh", "serta",
    "atau", "tidak", "akan", "telah", "juga", "dapat", "sudah",
    "ada", "hal", "cara", "lebih", "sangat", "semakin", "setelah",
    "sebelum", "ketika", "saat", "karena", "sehingga", "maka",
    "bahwa", "dengan", "secara", "antara", "tersebut", "yaitu",
    "yakni", "seperti", "mengenai", "tentang", "sambil", "pula",
    "lalu", "pun", "para", "ia", "mereka", "kami", "kita", "dia",

    # ── KATA UMUM AKADEMIK (frekuensi tinggi di abstrak) ──
    "penelitian", "hasil", "data", "analisis", "berdasarkan",
    "tujuan", "dilakukan", "metode", "pengembangan", "rancang",
    "bangun", "menghasilkan", "digunakan", "dihasilkan", "output",
    "input", "studi", "kasus", "proses", "sistem", "informasi",
    "aplikasi", "website", "web", "pengguna", "user",

    # ── KATA METODOLOGI PENELITIAN ──
    "kuantitatif", "kualitatif", "deskriptif", "eksploratif",
    "eksperimen", "eksperimental", "regresi", "korelasi",
    "validasi", "verifikasi", "signifikan", "populasi", "sampel",
    "responden", "partisipan", "informan", "wawancara", "observasi",
    "kuesioner", "angket", "dokumentasi", "uji", "hipotesis",
    "normalitas", "reliabilitas", "validitas", "varian", "anova",

    # ── KATA PENGEMBANGAN SISTEM ──
    "tahap", "analisa", "perancangan", "implementasi", "pengujian",
    "blackbox", "whitebox", "deploy", "maintenance", "testing",
    "use", "case", "diagram", "entity", "relationship", "flowchart",
    "dfd", "uml", "basis", "database", "server", "client",
    "interface", "modul", "modul", "arsitektur", "fungsionalitas",

    # ── BOILERPLATE TEMPLATE ABSTRAK ──
    "urgensi", "dasar", "tingkat", "efisiensi", "akurasi", "performa",
    "tata", "kelola", "operasional", "kait", "laksana", "metodologi",
    "terap", "ancang", "struktur", "sistematis", "cakup", "tahap",
    "utama", "butuh", "fungsional", "non", "kumpul", "primer", "sekunder",
    "bebas", "kendala", "teknis", "harap", "kontribusi", "nyata",
    "dokumen", "rekomendasi", "analitis", "solutif", "civitas",
    "manfaat", "instrumen", "dukung", "putus", "optimal", "publik",
    "internal", "organisasi", "fokus", "topik", "kerja", "institut",
    "instansi", "tahapan", "kinerja", "guna", "pihak", "terkait",
    "luaran", "berkelanjutan", "diawali", "fase", "bersifat",
    "berjalan", "sasaran", "berupa", "akhirnya", "berkaitan",

    # ── KATA INSTITUSIONAL ──
    "universitas", "islam", "negeri", "raden", "fatah", "palembang",
    "fakultas", "sains", "teknologi", "program", "studi", "mahasiswa",
    "skripsi", "tugas", "akhir", "dosen", "pembimbing", "penguji",
    "uin", "smk", "sma", "smp", "sd", "ma", "mts", "mi",

    # ── KATA TEKNIS SI YANG TERLALU GENERIK ──
    "teknologi", "informasi", "komunikasi", "komputer", "digital",
    "elektronik", "online", "offline", "daring", "luring",
    "platform", "perangkat", "lunak", "keras", "jaringan",

    # ── ENTITAS LOKAL / STUDI KASUS (noise) ──
    "sakit", "rumah", "toko", "agama", "sungai", "pondok",
    "pesantren", "sekolah", "buku", "uang", "kota", "kabupaten",
    "kecamatan", "desa", "kelurahan", "provinsi", "wilayah",
    "daerah", "nasional", "indonesia", "sumatera", "selatan",
    "banyuasin", "ogan", "ilir", "komering", "musi", "rawas",
    "lahat", "prabumulih", "lubuklinggau", "pagaralam", "empat",
    "lawang", "muara", "kantor", "dinas", "badan", "lembaga",
    "perusahaan", "cabang", "pusat", "muhammadiyah", "nahdlatul",
    "ulama", "camat", "bupati", "walikota", "gubernur",

    # ── STOPWORD BAHASA INGGRIS UMUM ──
    "the", "and", "of", "in", "to", "a", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "can", "need", "dare", "ought", "used",
    "this", "that", "these", "those", "it", "its", "with",
    "for", "on", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "then", "than", "so", "but",
    "or", "if", "while", "although", "because", "since", "until",
    "about", "against", "within", "without", "some", "any",
    "every", "each", "both", "few", "more", "most", "other",

    # ── KATA INGGRIS GENERIK (tidak membedakan topik SI) ──
    "information", "system", "method", "using", "based",
    "application", "result", "approach", "study", "case", "data",
    "fit", "human", "organization", "process", "function",
    "feature", "module", "research", "development", "analysis",
    "model", "value", "test", "factor", "variable",

    # ── AKRONIM & KODE TIDAK BERMAKNA ──
    "eucs", "iso", "ipa", "hot", "mus", "iii", "ii", "iv",
    "vi", "vii", "viii", "ix", "xi", "xii", "pos", "lan",
    "wan", "vpn", "http", "https", "api", "sql", "php", "css",
    "html", "xml", "json", "ram", "cpu", "gpu", "os", "pc",
    "it", "ict",
}

def get_all_stopwords():
    try:
        nltk_sw = set(stopwords.words('indonesian'))
    except LookupError:
        nltk.download('stopwords')
        nltk_sw = set(stopwords.words('indonesian'))
    return nltk_sw.union(CUSTOM_STOPWORDS_SI)
