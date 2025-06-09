"""
Advanced Analysis Functions
===========================
Machine learning and statistical analysis functions.
Part of Movie Reviews Analysis project by Chis Bogdan & Ciurescu Raul.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
from sklearn.feature_extraction.text import CountVectorizer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    """Advanced sentiment analysis and classification."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        
    def compare_classification_models(self, X, y, cv_folds=5):
        """
        Compare multiple classification algorithms for sentiment prediction.
        
        Args:
            X: Feature matrix
            y: Target labels
            cv_folds: Number of cross-validation folds
            
        Returns:
            dict: Results for each model
        """
        print("Comparing classification models for sentiment prediction...")
        
        # Define models to compare
        self.models = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='linear', random_state=42, probability=True)
        }
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print("-" * 50)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
            
            # Fit and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            test_accuracy = (y_pred == y_test).mean()
            
            # Store results
            self.results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_accuracy': test_accuracy,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'cv_scores': cv_scores
            }
            
            print(f"  CV Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
            print(f"  Test Accuracy: {test_accuracy:.3f}")
            print()
        
        # Identify best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['test_accuracy'])
        self.best_model = self.results[best_model_name]['model']
        
        print(f"Best performing model: {best_model_name}")
        print(f"Best accuracy: {self.results[best_model_name]['test_accuracy']:.3f}")
        
        return self.results
    
    def get_feature_importance(self, feature_names, model_name='Random Forest'):
        """
        Get feature importance from trained model.
        
        Args:
            feature_names: Names of features
            model_name: Name of model to analyze
            
        Returns:
            pandas.DataFrame: Feature importance scores
        """
        if model_name not in self.results:
            return None
        
        model = self.results[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(importance_scores)],
                'importance': importance_scores
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return None

class ClusterAnalyzer:
    """Advanced clustering analysis for audience segmentation."""
    
    def __init__(self):
        self.kmeans_model = None
        self.dbscan_model = None
        self.optimal_k = None
        
    def find_optimal_clusters(self, X, k_range=range(2, 8)):
        """
        Find optimal number of clusters using silhouette analysis.
        
        Args:
            X: Feature matrix
            k_range: Range of k values to test
            
        Returns:
            int: Optimal number of clusters
        """
        print("Finding optimal number of clusters...")
        
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            print(f"  K={k}: Silhouette Score = {silhouette_avg:.3f}")
        
        # Find optimal K
        optimal_idx = np.argmax(silhouette_scores)
        self.optimal_k = list(k_range)[optimal_idx]
        
        print(f"Optimal K: {self.optimal_k} (Silhouette Score: {max(silhouette_scores):.3f})")
        
        return self.optimal_k, silhouette_scores
    
    def perform_clustering(self, X):
        """
        Perform clustering analysis with optimal parameters.
        
        Args:
            X: Feature matrix
            
        Returns:
            dict: Clustering results
        """
        print("Performing clustering analysis...")
        
        # K-Means clustering
        self.kmeans_model = KMeans(n_clusters=self.optimal_k, random_state=42, n_init=10)
        kmeans_labels = self.kmeans_model.fit_predict(X)
        kmeans_silhouette = silhouette_score(X, kmeans_labels)
        
        # DBSCAN clustering
        self.dbscan_model = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = self.dbscan_model.fit_predict(X)
        n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        
        results = {
            'kmeans_labels': kmeans_labels,
            'dbscan_labels': dbscan_labels,
            'kmeans_silhouette': kmeans_silhouette,
            'n_clusters_kmeans': self.optimal_k,
            'n_clusters_dbscan': n_clusters_dbscan,
            'kmeans_centers': self.kmeans_model.cluster_centers_
        }
        
        if n_clusters_dbscan > 1:
            dbscan_silhouette = silhouette_score(X, dbscan_labels)
            results['dbscan_silhouette'] = dbscan_silhouette
        
        print(f"K-Means: {self.optimal_k} clusters, Silhouette: {kmeans_silhouette:.3f}")
        print(f"DBSCAN: {n_clusters_dbscan} clusters")
        
        return results

class TopicAnalyzer:
    """Topic modeling and analysis for review content."""
    
    def __init__(self):
        self.lda_model = None
        self.count_vectorizer = None
        self.optimal_topics = None
        
    def find_optimal_topics(self, texts, topic_range=range(2, 8)):
        """
        Find optimal number of topics using perplexity.
        
        Args:
            texts: List of text documents
            topic_range: Range of topic numbers to test
            
        Returns:
            int: Optimal number of topics
        """
        print("Finding optimal number of topics...")
        
        # Prepare text data
        self.count_vectorizer = CountVectorizer(
            max_features=100,
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        
        doc_term_matrix = self.count_vectorizer.fit_transform(texts)
        
        perplexities = []
        models = []
        
        for n_topics in topic_range:
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10,
                learning_method='online'
            )
            lda.fit(doc_term_matrix)
            perplexity = lda.perplexity(doc_term_matrix)
            perplexities.append(perplexity)
            models.append(lda)
            print(f"  Topics={n_topics}: Perplexity = {perplexity:.1f}")
        
        # Choose optimal number (lowest perplexity)
        optimal_idx = np.argmin(perplexities)
        self.optimal_topics = list(topic_range)[optimal_idx]
        self.lda_model = models[optimal_idx]
        
        print(f"Optimal Topics: {self.optimal_topics} (Perplexity: {min(perplexities):.1f})")
        
        return self.optimal_topics, perplexities
    
    def get_topic_words(self, n_words=10):
        """
        Get top words for each topic.
        
        Args:
            n_words: Number of top words per topic
            
        Returns:
            dict: Top words for each topic
        """
        if not self.lda_model or not self.count_vectorizer:
            return {}
        
        feature_names = self.count_vectorizer.get_feature_names_out()
        topic_words = {}
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_words[f'Topic {topic_idx + 1}'] = top_words
        
        return topic_words
    
    def assign_document_topics(self, texts):
        """
        Assign dominant topic to each document.
        
        Args:
            texts: List of text documents
            
        Returns:
            numpy.ndarray: Dominant topic for each document
        """
        if not self.lda_model or not self.count_vectorizer:
            return None
        
        doc_term_matrix = self.count_vectorizer.transform(texts)
        doc_topic_dist = self.lda_model.transform(doc_term_matrix)
        dominant_topics = np.argmax(doc_topic_dist, axis=1) + 1  # 1-indexed
        
        return dominant_topics

class StatisticalAnalyzer:
    """Statistical analysis and hypothesis testing."""
    
    @staticmethod
    def correlation_analysis(df, target_col, feature_cols):
        """
        Perform correlation analysis between features and target.
        
        Args:
            df: DataFrame with data
            target_col: Name of target column
            feature_cols: List of feature column names
            
        Returns:
            pandas.DataFrame: Correlation results
        """
        correlations = []
        
        for col in feature_cols:
            if col in df.columns:
                corr_coef = df[col].corr(df[target_col])
                
                # Statistical significance test
                if len(df) > 30:  # Large enough sample
                    _, p_value = stats.pearsonr(df[col].dropna(), 
                                              df[target_col][df[col].notna()])
                else:
                    p_value = np.nan
                
                correlations.append({
                    'feature': col,
                    'correlation': corr_coef,
                    'abs_correlation': abs(corr_coef),
                    'p_value': p_value,
                    'significant': p_value < 0.05 if not np.isnan(p_value) else False
                })
        
        correlation_df = pd.DataFrame(correlations)
        correlation_df = correlation_df.sort_values('abs_correlation', ascending=False)
        
        return correlation_df
    
    @staticmethod
    def group_comparison(df, group_col, target_col):
        """
        Compare target variable across different groups.
        
        Args:
            df: DataFrame with data
            group_col: Column defining groups
            target_col: Column with values to compare
            
        Returns:
            dict: Comparison results
        """
        groups = df[group_col].unique()
        group_stats = {}
        
        for group in groups:
            group_data = df[df[group_col] == group][target_col]
            group_stats[group] = {
                'count': len(group_data),
                'mean': group_data.mean(),
                'std': group_data.std(),
                'median': group_data.median()
            }
        
        # ANOVA test if more than 2 groups
        if len(groups) > 2:
            group_arrays = [df[df[group_col] == group][target_col].values for group in groups]
            f_stat, p_value = stats.f_oneway(*group_arrays)
            
            results = {
                'group_stats': group_stats,
                'anova_f_stat': f_stat,
                'anova_p_value': p_value,
                'significant_difference': p_value < 0.05
            }
        else:
            # T-test for 2 groups
            group1_data = df[df[group_col] == groups[0]][target_col]
            group2_data = df[df[group_col] == groups[1]][target_col]
            t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
            
            results = {
                'group_stats': group_stats,
                'ttest_t_stat': t_stat,
                'ttest_p_value': p_value,
                'significant_difference': p_value < 0.05
            }
        
        return results

# Example usage and testing
if __name__ == "__main__":
    print("Testing analysis functions...")
    print("All analysis modules loaded successfully!")
    print("Ready for comprehensive movie reviews analysis.")