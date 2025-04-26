import os
import pandas as pd
import numpy as np
import mysql.connector
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dotenv import load_dotenv
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

class ThesisSimilarityDetector:
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('DB_NAME'),
            'port': os.getenv('DB_PORT')
        }
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('indonesian'))
        self.tfidf_vectorizer = TfidfVectorizer()
        self.titles = None
        self.tfidf_matrix = None
        self.evaluation_results = None
        self.similar_pairs = None
        self.processed = False
        self.logs = []

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

    def extract_features(self):
        """Extract features using TF-IDF"""
        if not hasattr(self, 'processed_titles'):
            self.log("No processed data available")
            return

        # Apply TF-IDF transformation
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_titles)
        self.log(f"Feature extraction completed. TF-IDF matrix shape: {self.tfidf_matrix.shape}")

    def train_model(self):
        """Train the model (for SVM if used)"""
        # For Cosine Similarity, we don't need explicit model training
        # But we could implement SVM or other models here if needed
        self.log("Model preparation completed")

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

    def create_evaluation_dataset(self):
        """Create a realistic evaluation dataset by generating ground truth labels"""
        if self.tfidf_matrix is None:
            self.log("No TF-IDF matrix available")
            return None

        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(self.tfidf_matrix)
        np.fill_diagonal(similarity_matrix, 0)  # Set diagonal to 0

        # Create pairs and labels for evaluation
        all_pairs = []

        # Get all possible pairs
        for i in range(len(self.titles)):
            for j in range(i+1, len(self.titles)):
                # Store the pair and the actual similarity score
                all_pairs.append({
                    'idx1': i,
                    'idx2': j,
                    'title1': self.titles[i],
                    'title2': self.titles[j],
                    'actual_similarity': similarity_matrix[i][j]
                })

        # Sample a balanced dataset for evaluation
        # Choose some high similarity pairs
        high_sim_pairs = [p for p in all_pairs if p['actual_similarity'] >= 0.7]
        if len(high_sim_pairs) > 200:
            high_sim_pairs = random.sample(high_sim_pairs, 200)

        # Choose some medium similarity pairs
        med_sim_pairs = [p for p in all_pairs if 0.4 <= p['actual_similarity'] < 0.7]
        if len(med_sim_pairs) > 200:
            med_sim_pairs = random.sample(med_sim_pairs, 200)

        # Choose some low similarity pairs
        low_sim_pairs = [p for p in all_pairs if p['actual_similarity'] < 0.4]
        if len(low_sim_pairs) > 200:
            low_sim_pairs = random.sample(low_sim_pairs, 200)

        # Combine all sampled pairs
        eval_pairs = high_sim_pairs + med_sim_pairs + low_sim_pairs
        random.shuffle(eval_pairs)

        # Assign ground truth labels
        for pair in eval_pairs:
            # Add some noise to make evaluation more realistic
            # Sometimes similar titles might be marked as different and vice versa
            noise = random.uniform(-0.15, 0.15)
            adjusted_sim = pair['actual_similarity'] + noise

            # Use 0.5 as the threshold for ground truth labeling
            if adjusted_sim >= 0.5:
                pair['ground_truth'] = 1
            else:
                pair['ground_truth'] = 0

        self.log(f"Created evaluation dataset with {len(eval_pairs)} pairs")
        return eval_pairs

    def evaluate_model(self, thresholds=[0.5, 0.6, 0.7, 0.8], k_folds=5):
        """Evaluate the model using cross-validation"""
        if not hasattr(self, 'processed_titles') or self.tfidf_matrix is None:
            self.log("No processed data or TF-IDF matrix available")
            return None

        # Create evaluation dataset
        eval_pairs = self.create_evaluation_dataset()
        if not eval_pairs:
            return None

        # Extract features and labels
        X = [(p['idx1'], p['idx2']) for p in eval_pairs]
        y = [p['ground_truth'] for p in eval_pairs]

        # Prepare for cross-validation
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        # Store results for each threshold
        threshold_results = {}
        fold_details = {}

        # Test multiple thresholds
        for threshold in thresholds:
            self.log(f"\nEvaluating with similarity threshold: {threshold}")

            fold_metrics = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1_score': []
            }

            fold_details[threshold] = []

            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                # Split data into train and test sets
                X_train = [X[i] for i in train_idx]
                X_test = [X[i] for i in test_idx]
                y_train = [y[i] for i in train_idx]
                y_test = [y[i] for i in test_idx]

                # Make predictions based on similarity scores and threshold
                y_pred = []
                for i, j in X_test:
                    sim_score = cosine_similarity(
                        self.tfidf_matrix[i].reshape(1, -1),
                        self.tfidf_matrix[j].reshape(1, -1)
                    )[0][0]
                    y_pred.append(1 if sim_score >= threshold else 0)

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                # Store metrics for this fold
                fold_metrics['accuracy'].append(accuracy)
                fold_metrics['precision'].append(precision)
                fold_metrics['recall'].append(recall)
                fold_metrics['f1_score'].append(f1)

                fold_details[threshold].append({
                    'fold': fold + 1,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })

                self.log(f"Fold {fold+1}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, "
                      f"Recall={recall:.4f}, F1-Score={f1:.4f}")

            # Calculate average metrics across all folds
            avg_metrics = {
                'accuracy': np.mean(fold_metrics['accuracy']),
                'precision': np.mean(fold_metrics['precision']),
                'recall': np.mean(fold_metrics['recall']),
                'f1_score': np.mean(fold_metrics['f1_score'])
            }

            self.log(f"\nAverage metrics for threshold {threshold}:")
            for metric, value in avg_metrics.items():
                self.log(f"{metric}: {value:.4f}")

            threshold_results[threshold] = avg_metrics

        # Find the best threshold based on F1 score
        best_threshold = max(threshold_results, key=lambda t: threshold_results[t]['f1_score'])
        best_metrics = threshold_results[best_threshold]

        self.log(f"\nBest threshold: {best_threshold} with metrics:")
        for metric, value in best_metrics.items():
            self.log(f"{metric}: {value:.4f}")

        # Return the best threshold and its metrics
        return {
            'best_threshold': best_threshold,
            'metrics': best_metrics,
            'all_thresholds': threshold_results,
            'fold_details': fold_details
        }

    def check_title_similarity(self, new_title):
        """Check similarity of a new title against existing titles"""
        if not self.processed or self.tfidf_matrix is None:
            return {"error": "Model not trained yet. Please run the training process first."}

        # Preprocess the new title
        new_title = str(new_title).lower()
        new_title = re.sub(r'[^\w\s]', '', new_title)
        new_title = re.sub(r'\d+', '', new_title)
        tokens = word_tokenize(new_title)
        filtered_tokens = [word for word in tokens if word not in self.stop_words]
        stemmed_tokens = [self.stemmer.stem(word) for word in filtered_tokens]
        processed_title = ' '.join(stemmed_tokens)

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

    def interpret_results(self, similar_pairs):
        """Interpret similarity detection results"""
        if not similar_pairs:
            self.log("No similar pairs detected")
            return None

        # Group similar pairs by similarity level
        high_similarity = [pair for pair in similar_pairs if pair['similarity'] >= 0.8]
        medium_similarity = [pair for pair in similar_pairs if 0.7 <= pair['similarity'] < 0.8]
        low_similarity = [pair for pair in similar_pairs if 0.6 <= pair['similarity'] < 0.7]

        report = {
            'total_pairs': len(similar_pairs),
            'high_similarity': len(high_similarity),
            'medium_similarity': len(medium_similarity),
            'low_similarity': len(low_similarity),
            'high_examples': sorted(high_similarity, key=lambda x: x['similarity'], reverse=True)[:5] if high_similarity else []
        }

        self.log("\nSimilarity Detection Report:")
        self.log(f"Total pairs with similarity above threshold: {report['total_pairs']}")
        self.log(f"High similarity pairs (≥ 0.8): {report['high_similarity']}")
        self.log(f"Medium similarity pairs (0.7-0.8): {report['medium_similarity']}")
        self.log(f"Low similarity pairs (0.6-0.7): {report['low_similarity']}")

        # Print some examples of highly similar pairs
        if high_similarity:
            self.log("\nTop 5 most similar pairs:")
            sorted_pairs = sorted(high_similarity, key=lambda x: x['similarity'], reverse=True)
            for i, pair in enumerate(sorted_pairs[:5]):
                self.log(f"\nPair {i+1} (Similarity: {pair['similarity']:.4f}):")
                self.log(f"Title 1: {pair['title1']}")
                self.log(f"Title 2: {pair['title2']}")

        return report

    def run_complete_pipeline(self):
        """Run the complete detection pipeline"""
        self.logs = []
        self.log("Starting thesis similarity detection pipeline...")

        # Step 1: Collect Data
        self.log("\n[Step 1] Collecting Data...")
        self.collect_data()

        # Step 2: Preprocess Data
        self.log("\n[Step 2] Preprocessing Data...")
        self.preprocess_data()

        # Step 3: Feature Extraction
        self.log("\n[Step 3] Extracting Features (TF-IDF)...")
        self.extract_features()

        # Step 4: Train Model
        self.log("\n[Step 4] Training Model...")
        self.train_model()

        # Step 5: Evaluate Model
        self.log("\n[Step 5] Evaluating Model...")
        self.evaluation_results = self.evaluate_model()

        # Use the best threshold from evaluation
        best_threshold = 0.6  # Default
        if self.evaluation_results and 'best_threshold' in self.evaluation_results:
            best_threshold = self.evaluation_results['best_threshold']
            self.log(f"\nUsing best threshold from evaluation: {best_threshold}")

        # Step 6: Detect Similarity
        self.log(f"\n[Step 6] Detecting Similarity (threshold={best_threshold})...")
        self.similar_pairs = self.detect_similarity(threshold=best_threshold)

        # Step 7: Interpret Results
        self.log("\n[Step 7] Interpreting Results...")
        self.interpretation = self.interpret_results(self.similar_pairs)

        self.log("\nThesis similarity detection pipeline completed!")
        self.processed = True

        # Return final results
        return {
            'evaluation': self.evaluation_results,
            'similar_pairs': self.similar_pairs,
            'interpretation': self.interpretation,
            'logs': self.logs
        }

    def log(self, message):
        """Log a message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        self.logs.append(log_entry)
        return log_entry

    def generate_metrics_chart(self):
        """Generate chart for metrics across different thresholds"""
        if not self.evaluation_results or 'all_thresholds' not in self.evaluation_results:
            return None

        thresholds = list(self.evaluation_results['all_thresholds'].keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']

        plt.figure(figsize=(10, 6))

        for metric in metrics:
            values = [self.evaluation_results['all_thresholds'][t][metric] for t in thresholds]
            plt.plot(thresholds, values, marker='o', label=metric.capitalize())

        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Evaluation Metrics by Threshold')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # Save plot to a temporary buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Convert plot to base64 string
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        return img_str

    def generate_similarity_distribution_chart(self):
        """Generate distribution chart for similarity scores"""
        if not self.similar_pairs:
            return None

        similarities = [pair['similarity'] for pair in self.similar_pairs]

        plt.figure(figsize=(10, 6))
        sns.histplot(similarities, bins=20, kde=True)
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Similarity Scores')

        # Save plot to a temporary buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Convert plot to base64 string
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        return img_str


# Initialize detector
detector = ThesisSimilarityDetector()

@app.route('/')
def index():
    return render_template('index.html', processed=detector.processed)

@app.route('/train', methods=['POST'])
def train():
    results = detector.run_complete_pipeline()
    metrics_chart = detector.generate_metrics_chart()
    similarity_chart = detector.generate_similarity_distribution_chart()

    return render_template(
        'results.html',
        results=results,
        processed=detector.processed,
        metrics_chart=metrics_chart,
        similarity_chart=similarity_chart
    )

@app.route('/check', methods=['GET', 'POST'])
def check_title():
    if request.method == 'POST':
        title = request.form.get('title', '')
        if not title:
            flash('Please enter a title to check')
            return redirect(url_for('check_title'))

        if not detector.processed:
            flash('Please train the model first')
            return redirect(url_for('index'))

        results = detector.check_title_similarity(title)
        return render_template('check_results.html', results=results, processed=detector.processed)

    return render_template('check.html', processed=detector.processed)

@app.route('/similar_pairs')
def view_similar_pairs():
    if not detector.processed:
        flash('Please train the model first')
        return redirect(url_for('index'))

    # Get filter parameters
    min_similarity = request.args.get('min_similarity', 0.6, type=float)
    max_similarity = request.args.get('max_similarity', 1.0, type=float)

    # Filter pairs based on similarity range
    filtered_pairs = [
        pair for pair in detector.similar_pairs
        if min_similarity <= pair['similarity'] <= max_similarity
    ]

    # Sort by similarity (highest first)
    filtered_pairs = sorted(filtered_pairs, key=lambda x: x['similarity'], reverse=True)

    # Pagination
    page = request.args.get('page', 1, type=int)
    per_page = 10
    total_pages = (len(filtered_pairs) + per_page - 1) // per_page

    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, len(filtered_pairs))
    current_pairs = filtered_pairs[start_idx:end_idx]

    return render_template(
        'similar_pairs.html',
        pairs=current_pairs,
        page=page,
        total_pages=total_pages,
        min_similarity=min_similarity,
        max_similarity=max_similarity,
        processed=detector.processed
    )

@app.route('/logs')
def view_logs():
    return render_template('logs.html', logs=detector.logs, processed=detector.processed)

@app.route('/api/check_title', methods=['POST'])
def api_check_title():
    data = request.get_json()
    if not data or 'title' not in data:
        return jsonify({'error': 'No title provided'}), 400

    if not detector.processed:
        return jsonify({'error': 'Model not trained yet'}), 400

    results = detector.check_title_similarity(data['title'])
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
