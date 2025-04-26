# Thesis Similarity Detector

Thesis Similarity Detector adalah aplikasi web berbasis Flask yang dirancang untuk mendeteksi kemiripan antar judul skripsi. Aplikasi ini memungkinkan pengguna untuk mengidentifikasi potensi plagiarisme dan kesamaan topik penelitian dengan mengimplementasikan teknik pemrosesan bahasa alami (NLP) dan perhitungan kemiripan berbasis cosine similarity.

## Fitur Utama

- Mengambil data judul skripsi dari database MySQL
- Pemrosesan teks dan pembersihan data
- Ekstraksi fitur menggunakan TF-IDF
- Deteksi kemiripan menggunakan cosine similarity
- Evaluasi model dengan cross-validation
- Visualisasi hasil dengan grafik metrik
- Antarmuka web untuk pengecekan judul baru

## Komponen Utama Kode

### 1. Pengambilan Data

Aplikasi ini mengambil data judul skripsi dari database MySQL dengan fungsi `collect_data()`:

```python
def collect_data(self):
    """Retrieve thesis titles from database"""
    try:
        conn = mysql.connector.connect(**self.db_config)
        cursor = conn.cursor()

        # Fetch data from the fkip table (limit to 1000 records)
        query = "SELECT * FROM fkip LIMIT 1000"
        cursor.execute(query)

        # Get column names
        columns = [desc[0] for desc in cursor.description]

        # Fetch all rows
        rows = cursor.fetchall()

        # Create DataFrame
        df = pd.DataFrame(rows, columns=columns)

        # Check if there's a column for thesis titles
        title_column = None
        for possible_column in ['judul', 'judul_skripsi', 'title']:
            if possible_column in df.columns:
                title_column = possible_column
                break

        if title_column is None:
            raise ValueError("Could not find a column for thesis titles in the database")

        self.titles = df[title_column].tolist()
        self.ids = df.iloc[:, 0].tolist()  # Assuming first column is ID

        cursor.close()
        conn.close()

        self.log(f"Successfully collected {len(self.titles)} thesis titles")
        return df

    except Exception as e:
        self.log(f"Error collecting data: {e}")
        return None
```

### 2. Pembersihan Data

Proses pembersihan data dilakukan pada fungsi `preprocess_data()` yang meliputi:

- Konversi ke lowercase
- Penghapusan karakter khusus dan angka
- Tokenisasi
- Penghapusan stopwords
- Stemming

```python
def preprocess_data(self):
    """Preprocess the thesis titles"""
    if not self.titles:
        self.log("No data to preprocess")
        return

    processed_titles = []

    for title in self.titles:
        # Convert to lowercase
        title = str(title).lower()

        # Remove special characters and numbers
        title = re.sub(r'[^\w\s]', '', title)
        title = re.sub(r'\d+', '', title)

        # Tokenize
        tokens = word_tokenize(title)

        # Remove stop words
        filtered_tokens = [word for word in tokens if word not in self.stop_words]

        # Stemming
        stemmed_tokens = [self.stemmer.stem(word) for word in filtered_tokens]

        # Join tokens back to string
        processed_title = ' '.join(stemmed_tokens)
        processed_titles.append(processed_title)

    self.processed_titles = processed_titles
    self.log("Data preprocessing completed")
```

### 3. Tokenisasi

Proses tokenisasi dilakukan dengan bantuan NLTK, khususnya dengan fungsi `word_tokenize()`. Teks dipecah menjadi token-token individual (kata) yang kemudian akan diproses lebih lanjut dengan menghilangkan stopwords dan melakukan stemming.

```python
# Tokenize
tokens = word_tokenize(title)

# Remove stop words
filtered_tokens = [word for word in tokens if word not in self.stop_words]

# Stemming
stemmed_tokens = [self.stemmer.stem(word) for word in filtered_tokens]
```

### 4. Ekstraksi Fitur dengan TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) digunakan untuk mengekstrak fitur dari judul yang telah diproses. Implementasi ini menggunakan `TfidfVectorizer` dari scikit-learn:

```python
def extract_features(self):
    """Extract features using TF-IDF"""
    if not hasattr(self, 'processed_titles'):
        self.log("No processed data available")
        return

    # Apply TF-IDF transformation
    self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_titles)
    self.log(f"Feature extraction completed. TF-IDF matrix shape: {self.tfidf_matrix.shape}")
```

### 5. Cosine Similarity

Perhitungan kemiripan antar judul menggunakan metode cosine similarity. Fungsi `detect_similarity()` menghitung cosine similarity antara setiap pasangan judul dan mengidentifikasi pasangan dengan nilai kemiripan di atas ambang batas tertentu:

```python
def detect_similarity(self, threshold=0.6):
    """Detect similarity between thesis titles"""
    if self.tfidf_matrix is None:
        self.log("No TF-IDF matrix available")
        return []

    # Compute cosine similarity matrix
    cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    # Create a list to store similar pairs
    similar_pairs = []

    # Check for similar pairs
    for i in range(len(cosine_sim)):
        for j in range(i+1, len(cosine_sim)):
            similarity = cosine_sim[i][j]
            # Extract author names from IDs if available
            author1 = str(self.ids[i]).split('_')[0] if '_' in str(self.ids[i]) else str(self.ids[i])
            author2 = str(self.ids[j]).split('_')[0] if '_' in str(self.ids[j]) else str(self.ids[j])

            # Skip if exact same title (similarity=1.0) AND same author
            if similarity >= 0.999 and author1 == author2:
                continue

            if similarity >= threshold:
                similar_pairs.append({
                    'id1': self.ids[i],
                    'title1': self.titles[i],
                    'id2': self.ids[j],
                    'title2': self.titles[j],
                    'similarity': float(similarity)
                })

    return similar_pairs
```

### 6. Evaluasi Model

Evaluasi model dilakukan dengan cross-validation untuk menentukan nilai ambang batas (threshold) optimal. Proses ini dilakukan dengan membuat dataset evaluasi yang seimbang dan menghitung berbagai metrik seperti akurasi, presisi, recall, dan F1-score:

```python
def evaluate_model(self, thresholds=[0.5, 0.6, 0.7, 0.8], k_folds=5):
    """Evaluate the model using cross-validation"""
    # ... (code for creating evaluation dataset)

    # Test multiple thresholds
    for threshold in thresholds:
        # ... (code for cross-validation)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # ... (code for storing metrics)

    # Find the best threshold based on F1 score
    best_threshold = max(threshold_results, key=lambda t: threshold_results[t]['f1_score'])

    # ... (return best threshold and metrics)
```

### 7. Pengecekan Judul Baru

Aplikasi menyediakan fungsi untuk memeriksa kemiripan judul baru dengan judul yang ada dalam database:

```python
def check_title_similarity(self, new_title):
    """Check similarity of a new title against existing titles"""
    if not self.processed or self.tfidf_matrix is None:
        return {"error": "Model not trained yet. Please run the training process first."}

    # Preprocess the new title
    # ... (code for preprocessing)

    # Transform using the existing vectorizer
    new_title_vector = self.tfidf_vectorizer.transform([processed_title])

    # Calculate similarity with all existing titles
    similarities = cosine_similarity(new_title_vector, self.tfidf_matrix)[0]

    # Get top 10 most similar titles
    top_indices = similarities.argsort()[-10:][::-1]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0:  # Only include if there's some similarity
            results.append({
                'title': self.titles[idx],
                'similarity': float(similarities[idx]),
                'id': self.ids[idx]
            })

    return {
        'input_title': new_title,
        'similar_titles': results
    }
```

## Alur Kerja Aplikasi

1. **Inisialisasi**: Aplikasi dimulai dengan inisialisasi detector dan konfigurasi Flask.
2. **Pengambilan Data**: Data judul skripsi diambil dari database MySQL.
3. **Preprocessing**: Judul-judul skripsi dibersihkan dan ditokenisasi.
4. **Ekstraksi Fitur**: Fitur TF-IDF dihasilkan dari judul yang telah diproses.
5. **Evaluasi Model**: Nilai ambang batas optimal ditentukan melalui cross-validation.
6. **Deteksi Kemiripan**: Kemiripan antar judul dideteksi menggunakan cosine similarity.
7. **Visualisasi**: Hasil diinterpretasikan dan divisualisasikan dalam bentuk grafik.
8. **Interaksi Web**: Pengguna dapat mengakses hasil dan melakukan pengecekan judul baru melalui antarmuka web.

## Cara Penggunaan

1. Pastikan Python dan semua dependensi telah terinstal.
2. Konfigurasikan koneksi database di file `.env`.
3. Jalankan aplikasi dengan perintah `python app.py`.
4. Akses aplikasi melalui browser di `http://localhost:5000`.

## Teknologi yang Digunakan

- Python
- Flask
- NLTK untuk pemrosesan bahasa alami
- scikit-learn untuk TF-IDF dan cosine similarity
- MySQL untuk penyimpanan data
- matplotlib dan seaborn untuk visualisasi

## Ringkasan

Thesis Similarity Detector merupakan solusi komprehensif untuk mendeteksi kemiripan antar judul skripsi menggunakan teknik NLP dan cosine similarity. Aplikasi ini tidak hanya mengidentifikasi judul yang mirip tetapi juga memberikan evaluasi kinerja model dan visualisasi hasil untuk membantu pengguna memahami pola kemiripan.
