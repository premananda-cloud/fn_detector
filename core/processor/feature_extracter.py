"""
Feature Extractor Module
Extracts linguistic and statistical features for AI text detection
"""

import re
import logging
import string
from typing import Dict, List, Optional
from collections import Counter
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract linguistic features from text for AI-generated text detection
    Features include statistical, lexical, syntactic, and stylistic measures
    """
    
    def __init__(self):
        """Initialize feature extractor"""
        # Common words and stopwords for analysis
        self.common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'
        }
        
        # Function words that AI often uses differently
        self.function_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'it',
            'for', 'not', 'on', 'with', 'as', 'you', 'do', 'at', 'this', 'but',
            'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an',
            'will', 'my', 'one', 'all', 'would', 'there', 'their'
        }
        
        # Transition words common in AI text
        self.transition_words = {
            'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'consequently', 'nevertheless', 'nonetheless', 'meanwhile',
            'subsequently', 'thus', 'hence', 'accordingly', 'similarly'
        }
        
        # Punctuation for analysis
        self.punctuation = set(string.punctuation)
        
    def extract(self, text: str) -> Dict:
        """
        Extract all features from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing all extracted features
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input must be a non-empty string")
        
        features = {}
        
        # Basic statistics
        features.update(self._extract_basic_stats(text))
        
        # Lexical features
        features.update(self._extract_lexical_features(text))
        
        # Syntactic features
        features.update(self._extract_syntactic_features(text))
        
        # Stylistic features
        features.update(self._extract_stylistic_features(text))
        
        # Readability metrics
        features.update(self._extract_readability_features(text))
        
        # N-gram features
        features.update(self._extract_ngram_features(text))
        
        # Perplexity-related features
        features.update(self._extract_perplexity_features(text))
        
        return features
    
    def _extract_basic_stats(self, text: str) -> Dict:
        """Extract basic statistical features"""
        words = text.split()
        sentences = self._split_sentences(text)
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'std_sentence_length': np.std([len(s.split()) for s in sentences]) if sentences else 0,
        }
    
    def _extract_lexical_features(self, text: str) -> Dict:
        """Extract lexical diversity and vocabulary features"""
        words = [w.lower() for w in re.findall(r'\b\w+\b', text)]
        
        if not words:
            return {
                'lexical_diversity': 0,
                'unique_word_ratio': 0,
                'rare_word_ratio': 0,
                'stopword_ratio': 0,
                'function_word_ratio': 0,
                'content_word_ratio': 0
            }
        
        unique_words = set(words)
        word_freq = Counter(words)
        
        # Calculate ratios
        stopword_count = sum(1 for w in words if w in self.common_words)
        function_word_count = sum(1 for w in words if w in self.function_words)
        rare_words = sum(1 for w, count in word_freq.items() if count == 1)
        
        return {
            'lexical_diversity': len(unique_words) / len(words),
            'unique_word_ratio': len(unique_words) / len(words),
            'rare_word_ratio': rare_words / len(unique_words),
            'stopword_ratio': stopword_count / len(words),
            'function_word_ratio': function_word_count / len(words),
            'content_word_ratio': 1 - (function_word_count / len(words))
        }
    
    def _extract_syntactic_features(self, text: str) -> Dict:
        """Extract syntactic and grammatical features"""
        sentences = self._split_sentences(text)
        words = text.split()
        
        # Count different sentence types
        declarative = sum(1 for s in sentences if s.strip().endswith('.'))
        interrogative = sum(1 for s in sentences if s.strip().endswith('?'))
        exclamatory = sum(1 for s in sentences if s.strip().endswith('!'))
        
        # Count punctuation
        punct_count = sum(1 for c in text if c in self.punctuation)
        comma_count = text.count(',')
        semicolon_count = text.count(';')
        colon_count = text.count(':')
        
        # Count transition words
        words_lower = [w.lower().strip(string.punctuation) for w in words]
        transition_count = sum(1 for w in words_lower if w in self.transition_words)
        
        return {
            'declarative_ratio': declarative / len(sentences) if sentences else 0,
            'interrogative_ratio': interrogative / len(sentences) if sentences else 0,
            'exclamatory_ratio': exclamatory / len(sentences) if sentences else 0,
            'punctuation_density': punct_count / len(words) if words else 0,
            'comma_density': comma_count / len(words) if words else 0,
            'semicolon_density': semicolon_count / len(words) if words else 0,
            'colon_density': colon_count / len(words) if words else 0,
            'transition_word_ratio': transition_count / len(words) if words else 0
        }
    
    def _extract_stylistic_features(self, text: str) -> Dict:
        """Extract stylistic and writing pattern features"""
        words = text.split()
        
        # Count different character types
        uppercase_count = sum(1 for c in text if c.isupper())
        lowercase_count = sum(1 for c in text if c.islower())
        digit_count = sum(1 for c in text if c.isdigit())
        
        # Count specific patterns
        contraction_count = len(re.findall(r"\w+'\w+", text))  # e.g., don't, it's
        hyphen_count = len(re.findall(r'\w+-\w+', text))  # hyphenated words
        
        # Sentence starters
        sentences = self._split_sentences(text)
        capitalized_starts = sum(1 for s in sentences if s and s[0].isupper())
        
        # Word length distribution
        word_lengths = [len(w) for w in words if w]
        
        return {
            'uppercase_ratio': uppercase_count / len(text) if text else 0,
            'lowercase_ratio': lowercase_count / len(text) if text else 0,
            'digit_ratio': digit_count / len(text) if text else 0,
            'contraction_ratio': contraction_count / len(words) if words else 0,
            'hyphen_ratio': hyphen_count / len(words) if words else 0,
            'proper_sentence_starts': capitalized_starts / len(sentences) if sentences else 0,
            'short_word_ratio': sum(1 for l in word_lengths if l <= 3) / len(word_lengths) if word_lengths else 0,
            'long_word_ratio': sum(1 for l in word_lengths if l >= 7) / len(word_lengths) if word_lengths else 0,
        }
    
    def _extract_readability_features(self, text: str) -> Dict:
        """Extract readability metrics"""
        words = re.findall(r'\b\w+\b', text)
        sentences = self._split_sentences(text)
        syllables = sum(self._count_syllables(w) for w in words)
        
        if not words or not sentences:
            return {
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'avg_syllables_per_word': 0,
                'complex_word_ratio': 0
            }
        
        # Flesch Reading Ease
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllables / len(words)
        
        flesch_reading_ease = (
            206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        )
        
        # Flesch-Kincaid Grade Level
        flesch_kincaid_grade = (
            0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
        )
        
        # Complex words (3+ syllables)
        complex_words = sum(1 for w in words if self._count_syllables(w) >= 3)
        
        return {
            'flesch_reading_ease': max(0, min(100, flesch_reading_ease)),
            'flesch_kincaid_grade': max(0, flesch_kincaid_grade),
            'avg_syllables_per_word': avg_syllables_per_word,
            'complex_word_ratio': complex_words / len(words)
        }
    
    def _extract_ngram_features(self, text: str) -> Dict:
        """Extract n-gram based features"""
        words = [w.lower() for w in re.findall(r'\b\w+\b', text)]
        
        if len(words) < 2:
            return {
                'bigram_repetition': 0,
                'trigram_repetition': 0,
                'unique_bigram_ratio': 0,
                'unique_trigram_ratio': 0
            }
        
        # Generate bigrams
        bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
        bigram_freq = Counter(bigrams)
        
        # Generate trigrams if possible
        trigrams = []
        trigram_freq = Counter()
        if len(words) >= 3:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            trigram_freq = Counter(trigrams)
        
        return {
            'bigram_repetition': sum(1 for count in bigram_freq.values() if count > 1) / len(bigrams) if bigrams else 0,
            'trigram_repetition': sum(1 for count in trigram_freq.values() if count > 1) / len(trigrams) if trigrams else 0,
            'unique_bigram_ratio': len(bigram_freq) / len(bigrams) if bigrams else 0,
            'unique_trigram_ratio': len(trigram_freq) / len(trigrams) if trigrams else 0
        }
    
    def _extract_perplexity_features(self, text: str) -> Dict:
        """Extract features related to text predictability"""
        words = [w.lower() for w in re.findall(r'\b\w+\b', text)]
        
        if not words:
            return {
                'word_entropy': 0,
                'character_entropy': 0,
                'burstiness': 0
            }
        
        # Word frequency entropy
        word_freq = Counter(words)
        word_probs = [count / len(words) for count in word_freq.values()]
        word_entropy = -sum(p * np.log2(p) for p in word_probs if p > 0)
        
        # Character frequency entropy
        chars = [c.lower() for c in text if c.isalpha()]
        char_freq = Counter(chars)
        char_probs = [count / len(chars) for count in char_freq.values()]
        char_entropy = -sum(p * np.log2(p) for p in char_probs if p > 0) if chars else 0
        
        # Burstiness (variance in word usage)
        word_gaps = []
        for word in word_freq:
            positions = [i for i, w in enumerate(words) if w == word]
            if len(positions) > 1:
                gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                word_gaps.extend(gaps)
        
        burstiness = np.std(word_gaps) if word_gaps else 0
        
        return {
            'word_entropy': word_entropy,
            'character_entropy': char_entropy,
            'burstiness': burstiness / len(words) if words else 0
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _count_syllables(self, word: str) -> int:
        """
        Estimate syllable count for a word
        Simple heuristic-based approach
        """
        word = word.lower()
        syllables = 0
        vowels = 'aeiouy'
        
        # Remove trailing 'e'
        if word.endswith('e'):
            word = word[:-1]
        
        # Count vowel groups
        previous_was_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllables += 1
            previous_was_vowel = is_vowel
        
        # Every word has at least one syllable
        return max(1, syllables)
    
    def extract_feature_vector(self, text: str) -> np.ndarray:
        """
        Extract features as a numpy array
        
        Args:
            text: Input text
            
        Returns:
            Numpy array of feature values
        """
        features = self.extract(text)
        # Return features in consistent order
        feature_names = sorted(features.keys())
        return np.array([features[name] for name in feature_names])
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names in order
        
        Returns:
            List of feature names
        """
        # Extract from a sample text to get all feature names
        sample_features = self.extract("Sample text for feature extraction.")
        return sorted(sample_features.keys())
    
    def extract_batch(self, texts: List[str]) -> List[Dict]:
        """
        Extract features from multiple texts
        
        Args:
            texts: List of texts
            
        Returns:
            List of feature dictionaries
        """
        features_list = []
        for text in texts:
            try:
                features = self.extract(text)
                features_list.append(features)
            except Exception as e:
                logger.error(f"Error extracting features: {e}")
                features_list.append({})
        
        return features_list
    
    def get_feature_importance_summary(self, features: Dict) -> Dict:
        """
        Get a summary of key features that might indicate AI generation
        
        Args:
            features: Feature dictionary
            
        Returns:
            Summary of important features
        """
        summary = {
            'consistency_score': 0,  # How consistent/uniform the text is
            'complexity_score': 0,   # How complex the language is
            'naturalness_score': 0   # How natural the writing appears
        }
        
        # Consistency indicators (high = more AI-like)
        consistency_indicators = [
            features.get('std_sentence_length', 0) < 5,  # Low variance
            features.get('unique_bigram_ratio', 1) > 0.8,  # High uniqueness
            features.get('burstiness', 0) < 0.1  # Low burstiness
        ]
        summary['consistency_score'] = sum(consistency_indicators) / len(consistency_indicators)
        
        # Complexity indicators
        complexity_indicators = [
            features.get('complex_word_ratio', 0) > 0.2,
            features.get('avg_word_length', 0) > 5,
            features.get('transition_word_ratio', 0) > 0.05
        ]
        summary['complexity_score'] = sum(complexity_indicators) / len(complexity_indicators)
        
        # Naturalness indicators (high = more human-like)
        naturalness_indicators = [
            features.get('contraction_ratio', 0) > 0.01,
            features.get('interrogative_ratio', 0) > 0.1,
            features.get('lexical_diversity', 0) < 0.8
        ]
        summary['naturalness_score'] = sum(naturalness_indicators) / len(naturalness_indicators)
        
        return summary


if __name__ == "__main__":
    # Example usage
    extractor = FeatureExtractor()
    
    sample_text = """
    Artificial intelligence has revolutionized the way we approach complex problems.
    Machine learning algorithms can now process vast amounts of data with remarkable efficiency.
    However, it's important to consider the ethical implications of these technologies.
    How will AI impact our daily lives in the coming years?
    """
    
    features = extractor.extract(sample_text)
    
    print("Extracted Features:")
    print("-" * 50)
    for feature_name, value in sorted(features.items()):
        print(f"{feature_name:30s}: {value:.4f}")
    
    print("\n" + "=" * 50)
    print("Feature Importance Summary:")
    print("-" * 50)
    summary = extractor.get_feature_importance_summary(features)
    for key, value in summary.items():
        print(f"{key:25s}: {value:.4f}")
