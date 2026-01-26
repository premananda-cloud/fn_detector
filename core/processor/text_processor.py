"""
Text Processor Module
Handles text cleaning, normalization, and preprocessing for AI text detection
"""

import re
import logging
from typing import Optional, List, Dict
import unicodedata
import html

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Text preprocessing pipeline for AI-generated text detection
    """
    
    def __init__(
        self,
        lowercase: bool = False,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_special_chars: bool = False,
        normalize_whitespace: bool = True,
        remove_html: bool = True,
        min_length: int = 10,
        max_length: Optional[int] = None
    ):
        """
        Initialize text processor
        
        Args:
            lowercase: Convert text to lowercase
            remove_urls: Remove URLs from text
            remove_emails: Remove email addresses
            remove_special_chars: Remove special characters
            normalize_whitespace: Normalize whitespace
            remove_html: Remove HTML tags
            min_length: Minimum text length (characters)
            max_length: Maximum text length (characters)
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_special_chars = remove_special_chars
        self.normalize_whitespace = normalize_whitespace
        self.remove_html = remove_html
        self.min_length = min_length
        self.max_length = max_length
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for text processing"""
        # URL pattern
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Email pattern
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # HTML tag pattern
        self.html_pattern = re.compile(r'<[^>]+>')
        
        # Multiple whitespace pattern
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Special characters pattern (keep basic punctuation)
        self.special_chars_pattern = re.compile(r'[^a-zA-Z0-9\s\.,!?\-\'\"]')
        
        # Repeated punctuation pattern
        self.repeated_punct_pattern = re.compile(r'([!?.]){2,}')
        
        # Mention pattern (@username)
        self.mention_pattern = re.compile(r'@\w+')
        
        # Hashtag pattern
        self.hashtag_pattern = re.compile(r'#\w+')
    
    def process(self, text: str) -> str:
        """
        Process text through the full pipeline
        
        Args:
            text: Raw input text
            
        Returns:
            Processed text
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input must be a non-empty string")
        
        # Store original for logging
        original_text = text
        
        # Apply processing steps
        text = self._decode_html_entities(text)
        
        if self.remove_html:
            text = self._remove_html_tags(text)
        
        if self.remove_urls:
            text = self._remove_urls(text)
        
        if self.remove_emails:
            text = self._remove_emails(text)
        
        text = self._normalize_unicode(text)
        text = self._normalize_quotes(text)
        text = self._fix_spacing(text)
        
        if self.remove_special_chars:
            text = self._remove_special_characters(text)
        
        text = self._normalize_punctuation(text)
        
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)
        
        if self.lowercase:
            text = text.lower()
        
        # Validate length
        text = self._validate_length(text)
        
        # Log if significant change
        if len(text) < len(original_text) * 0.5:
            logger.debug(f"Significant text reduction: {len(original_text)} -> {len(text)} chars")
        
        return text.strip()
    
    def _decode_html_entities(self, text: str) -> str:
        """Decode HTML entities like &amp; to &"""
        try:
            return html.unescape(text)
        except Exception as e:
            logger.warning(f"Error decoding HTML entities: {e}")
            return text
    
    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags"""
        return self.html_pattern.sub('', text)
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        # Remove URLs but keep the protocol for better sentence flow
        return self.url_pattern.sub('[URL]', text)
    
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses"""
        return self.email_pattern.sub('[EMAIL]', text)
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters"""
        try:
            # Normalize to NFKC form (compatibility composition)
            text = unicodedata.normalize('NFKC', text)
            return text
        except Exception as e:
            logger.warning(f"Error normalizing Unicode: {e}")
            return text
    
    def _normalize_quotes(self, text: str) -> str:
        """Normalize different types of quotes"""
        # Replace fancy quotes with standard ones
        quote_map = {
            '"': '"', '"': '"',  # Double quotes
            ''': "'", ''': "'",  # Single quotes
            '«': '"', '»': '"',  # Guillemets
            '‹': "'", '›': "'"
        }
        for fancy, standard in quote_map.items():
            text = text.replace(fancy, standard)
        return text
    
    def _fix_spacing(self, text: str) -> str:
        """Fix spacing issues around punctuation"""
        # Add space after punctuation if missing
        text = re.sub(r'([.!?,;:])([A-Za-z])', r'\1 \2', text)
        # Remove space before punctuation
        text = re.sub(r'\s+([.!?,;:])', r'\1', text)
        return text
    
    def _remove_special_characters(self, text: str) -> str:
        """Remove special characters (keep basic punctuation)"""
        return self.special_chars_pattern.sub(' ', text)
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize repeated punctuation"""
        # Replace multiple punctuation with single one
        text = self.repeated_punct_pattern.sub(r'\1', text)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace (multiple spaces to single space)"""
        # Replace multiple whitespace with single space
        text = self.whitespace_pattern.sub(' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    def _validate_length(self, text: str) -> str:
        """Validate and truncate text length"""
        if len(text) < self.min_length:
            logger.warning(f"Text too short ({len(text)} < {self.min_length}). Padding may be needed.")
        
        if self.max_length and len(text) > self.max_length:
            logger.debug(f"Truncating text from {len(text)} to {self.max_length} characters")
            text = text[:self.max_length]
        
        return text
    
    def process_batch(self, texts: List[str]) -> List[str]:
        """
        Process multiple texts
        
        Args:
            texts: List of raw texts
            
        Returns:
            List of processed texts
        """
        processed_texts = []
        for text in texts:
            try:
                processed = self.process(text)
                processed_texts.append(processed)
            except Exception as e:
                logger.error(f"Error processing text: {e}")
                processed_texts.append(text)  # Return original on error
        
        return processed_texts
    
    def clean_for_display(self, text: str, max_display_length: int = 200) -> str:
        """
        Clean text for display purposes (less aggressive)
        
        Args:
            text: Input text
            max_display_length: Maximum length for display
            
        Returns:
            Cleaned text suitable for display
        """
        # Light cleaning
        text = self._decode_html_entities(text)
        text = self._normalize_unicode(text)
        text = self._normalize_whitespace(text)
        
        # Truncate if needed
        if len(text) > max_display_length:
            text = text[:max_display_length] + "..."
        
        return text.strip()
    
    def get_text_stats(self, text: str) -> Dict:
        """
        Get statistics about the text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text statistics
        """
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len([s for s in sentences if s.strip()]) if sentences else 0,
            'whitespace_ratio': text.count(' ') / len(text) if text else 0,
            'punctuation_count': sum(1 for c in text if c in '.,!?;:'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }
    
    def remove_social_media_artifacts(self, text: str) -> str:
        """
        Remove social media specific artifacts
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Remove mentions
        text = self.mention_pattern.sub('', text)
        # Remove hashtags (optional: could keep the word part)
        text = self.hashtag_pattern.sub('', text)
        # Remove retweet markers
        text = re.sub(r'\bRT\b', '', text, flags=re.IGNORECASE)
        
        return self._normalize_whitespace(text)
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def split_into_chunks(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text
            chunk_size: Size of each chunk (in characters)
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_question = chunk.rfind('?')
                last_exclaim = chunk.rfind('!')
                
                break_point = max(last_period, last_question, last_exclaim)
                if break_point > chunk_size * 0.5:  # Only if in second half
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks


if __name__ == "__main__":
    # Example usage
    processor = TextProcessor(
        lowercase=False,
        remove_urls=True,
        remove_emails=True,
        normalize_whitespace=True
    )
    
    sample_text = """
    Check out this article: https://example.com/article
    Contact me at test@example.com for more info!!!
    
    This is a   test with    multiple spaces and &amp; HTML entities.
    <p>Some HTML tags</p> should be removed.
    """
    
    processed = processor.process(sample_text)
    print("Original:")
    print(sample_text)
    print("\nProcessed:")
    print(processed)
    print("\nStats:")
    stats = processor.get_text_stats(processed)
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
