# modules/ml/smart_account_linking.py

from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import Dict, List, Optional, Tuple
import numpy as np


class SmartAccountLinker:
    """
    ML-based account correlation system.
    Links accounts across platforms using multiple signals.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=100
        )
    
    def link_accounts(
        self,
        target_person: Dict,
        candidate_accounts: List[Dict],
        confidence_threshold: float = 0.6
    ) -> List[Dict]:
        """
        Find accounts that likely belong to the same person.
        
        Args:
            target_person: {
                'name': str,
                'usernames': list,
                'bio': str (optional),
                'known_accounts': list (optional)
            }
            candidate_accounts: List of {
                'platform': str,
                'username': str,
                'display_name': str,
                'bio': str,
                'profile_image_url': str
            }
            confidence_threshold: Minimum confidence to link
        
        Returns:
            List of accounts with confidence scores
        """
        print(f"\n[AccountLinker] Analyzing {len(candidate_accounts)} candidates...")
        
        linked_accounts = []
        
        for account in candidate_accounts:
            score = self._calculate_match_score(target_person, account)
            
            if score >= confidence_threshold:
                account['match_confidence'] = score * 100
                account['match_reasons'] = self._explain_match(target_person, account)
                linked_accounts.append(account)
                
                print(f"  ✓ {account['platform']}: @{account['username']} ({score*100:.1f}%)")
            else:
                print(f"  ✗ {account['platform']}: @{account['username']} ({score*100:.1f}%) - Below threshold")
        
        # Sort by confidence
        linked_accounts.sort(key=lambda x: x['match_confidence'], reverse=True)
        
        return linked_accounts
    
    def _calculate_match_score(self, target: Dict, account: Dict) -> float:
        """
        Calculate match score using multiple signals.
        
        Signals:
        1. Username similarity (30%)
        2. Name similarity (25%)
        3. Bio similarity (20%)
        4. Cross-platform patterns (15%)
        5. Writing style (10%)
        """
        scores = []
        weights = []
        
        # 1. Username similarity
        username_score = self._username_similarity(
            target.get('usernames', []),
            account.get('username', '')
        )
        scores.append(username_score)
        weights.append(0.30)
        
        # 2. Name similarity
        name_score = self._name_similarity(
            target.get('name', ''),
            account.get('display_name', '')
        )
        scores.append(name_score)
        weights.append(0.25)
        
        # 3. Bio similarity
        if target.get('bio') and account.get('bio'):
            bio_score = self._text_similarity(
                target.get('bio', ''),
                account.get('bio', '')
            )
            scores.append(bio_score)
            weights.append(0.20)
        
        # 4. Cross-platform patterns
        pattern_score = self._detect_patterns(target, account)
        scores.append(pattern_score)
        weights.append(0.15)
        
        # 5. Writing style (if bio available)
        if target.get('bio') and account.get('bio'):
            style_score = self._writing_style_similarity(
                target.get('bio', ''),
                account.get('bio', '')
            )
            scores.append(style_score)
            weights.append(0.10)
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Weighted average
        final_score = sum(s * w for s, w in zip(scores, normalized_weights))
        
        return final_score
    
    def _username_similarity(self, known_usernames: List[str], candidate: str) -> float:
        """
        Calculate username similarity using fuzzy matching.
        
        Techniques:
        - Exact match
        - Fuzzy ratio
        - Token matching
        - Numeric pattern matching
        """
        if not candidate:
            return 0.0
        
        candidate_clean = self._clean_username(candidate)
        
        max_score = 0.0
        
        for known in known_usernames:
            known_clean = self._clean_username(known)
            
            # Exact match
            if known_clean == candidate_clean:
                return 1.0
            
            # Fuzzy ratio
            fuzzy_score = fuzz.ratio(known_clean, candidate_clean) / 100.0
            
            # Token sort (ignores order)
            token_score = fuzz.token_sort_ratio(known_clean, candidate_clean) / 100.0
            
            # Partial match (substring)
            partial_score = fuzz.partial_ratio(known_clean, candidate_clean) / 100.0
            
            # Check if one contains the other
            contains_score = 0.0
            if known_clean in candidate_clean or candidate_clean in known_clean:
                contains_score = 0.8
            
            # Check numeric patterns (e.g., aariyan vs aariyan007)
            numeric_pattern_score = self._check_numeric_pattern(known_clean, candidate_clean)
            
            # Take best score
            score = max(fuzzy_score, token_score, partial_score, contains_score, numeric_pattern_score)
            max_score = max(max_score, score)
        
        return max_score
    
    def _name_similarity(self, target_name: str, candidate_name: str) -> float:
        """Calculate name similarity"""
        if not target_name or not candidate_name:
            return 0.0
        
        target_clean = self._clean_text(target_name)
        candidate_clean = self._clean_text(candidate_name)
        
        # Exact match
        if target_clean == candidate_clean:
            return 1.0
        
        # Fuzzy match
        fuzzy_score = fuzz.ratio(target_clean, candidate_clean) / 100.0
        
        # Token match (handles "John Doe" vs "Doe John")
        token_score = fuzz.token_sort_ratio(target_clean, candidate_clean) / 100.0
        
        # Check if first/last name matches
        target_parts = target_clean.split()
        candidate_parts = candidate_clean.split()
        
        name_overlap = len(set(target_parts) & set(candidate_parts)) / max(len(target_parts), len(candidate_parts))
        
        return max(fuzzy_score, token_score, name_overlap)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using TF-IDF"""
        if not text1 or not text2:
            return 0.0
        
        try:
            # Create TF-IDF vectors
            tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return similarity
        
        except:
            # Fallback to fuzzy matching
            return fuzz.ratio(text1, text2) / 100.0
    
    def _detect_patterns(self, target: Dict, account: Dict) -> float:
        """
        Detect cross-platform patterns.
        
        Patterns:
        - Same numbers (aariyan007 across platforms)
        - Same symbols/underscores
        - Consistent capitalization
        - Common suffixes (_official, _07, etc.)
        """
        known_usernames = target.get('usernames', [])
        candidate = account.get('username', '')
        
        if not candidate or not known_usernames:
            return 0.0
        
        score = 0.0
        
        # Extract patterns from known usernames
        known_numbers = set()
        known_symbols = set()
        
        for username in known_usernames:
            # Extract numbers
            numbers = re.findall(r'\d+', username)
            known_numbers.update(numbers)
            
            # Extract symbols
            symbols = re.findall(r'[_\-.]', username)
            known_symbols.update(symbols)
        
        # Check if candidate has same patterns
        candidate_numbers = set(re.findall(r'\d+', candidate))
        candidate_symbols = set(re.findall(r'[_\-.]', candidate))
        
        # Number pattern match
        if known_numbers and candidate_numbers:
            number_overlap = len(known_numbers & candidate_numbers) / len(known_numbers)
            score += number_overlap * 0.5
        
        # Symbol pattern match
        if known_symbols and candidate_symbols:
            symbol_overlap = len(known_symbols & candidate_symbols) / len(known_symbols)
            score += symbol_overlap * 0.3
        
        # Check for common suffixes
        common_suffixes = ['official', 'real', 'og', '07', '007', 'dev', 'tech']
        
        for suffix in common_suffixes:
            if any(suffix in u.lower() for u in known_usernames) and suffix in candidate.lower():
                score += 0.2
                break
        
        return min(1.0, score)
    
    def _writing_style_similarity(self, text1: str, text2: str) -> float:
        """
        Analyze writing style similarity.
        
        Features:
        - Capitalization patterns
        - Punctuation usage
        - Emoji usage
        - Sentence length
        """
        if not text1 or not text2:
            return 0.0
        
        score = 0.0
        
        # Capitalization ratio
        cap_ratio1 = sum(1 for c in text1 if c.isupper()) / len(text1) if text1 else 0
        cap_ratio2 = sum(1 for c in text2 if c.isupper()) / len(text2) if text2 else 0
        cap_similarity = 1 - abs(cap_ratio1 - cap_ratio2)
        score += cap_similarity * 0.3
        
        # Punctuation usage
        punct1 = len(re.findall(r'[!?.,;:]', text1)) / len(text1) if text1 else 0
        punct2 = len(re.findall(r'[!?.,;:]', text2)) / len(text2) if text2 else 0
        punct_similarity = 1 - abs(punct1 - punct2)
        score += punct_similarity * 0.3
        
        # Emoji usage
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            "]+", flags=re.UNICODE)
        
        emoji1 = len(emoji_pattern.findall(text1))
        emoji2 = len(emoji_pattern.findall(text2))
        
        if emoji1 > 0 or emoji2 > 0:
            emoji_similarity = 1 - abs(emoji1 - emoji2) / max(emoji1, emoji2, 1)
            score += emoji_similarity * 0.4
        
        return min(1.0, score)
    
    def _explain_match(self, target: Dict, account: Dict) -> List[str]:
        """Generate human-readable reasons for the match"""
        reasons = []
        
        # Username match
        username_score = self._username_similarity(
            target.get('usernames', []),
            account.get('username', '')
        )
        if username_score > 0.8:
            reasons.append(f"Strong username match")
        elif username_score > 0.6:
            reasons.append(f"Similar username pattern")
        
        # Name match
        name_score = self._name_similarity(
            target.get('name', ''),
            account.get('display_name', '')
        )
        if name_score > 0.8:
            reasons.append(f"Name matches")
        
        # Bio similarity
        if target.get('bio') and account.get('bio'):
            bio_score = self._text_similarity(
                target.get('bio', ''),
                account.get('bio', '')
            )
            if bio_score > 0.5:
                reasons.append(f"Similar bio content")
        
        # Patterns
        pattern_score = self._detect_patterns(target, account)
        if pattern_score > 0.5:
            reasons.append(f"Consistent cross-platform patterns")
        
        if not reasons:
            reasons.append("Weak match - manual verification recommended")
        
        return reasons
    
    @staticmethod
    def _clean_username(username: str) -> str:
        """Clean username for comparison"""
        return re.sub(r'[^a-z0-9]', '', username.lower())
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text for comparison"""
        return ' '.join(text.lower().split())
    
    @staticmethod
    def _check_numeric_pattern(str1: str, str2: str) -> float:
        """Check if strings differ only by numbers"""
        # Remove numbers
        base1 = re.sub(r'\d+', '', str1)
        base2 = re.sub(r'\d+', '', str2)
        
        if base1 == base2 and base1:
            return 0.85  # High confidence if base is same
        
        return 0.0


# Example usage
if __name__ == "__main__":
    linker = SmartAccountLinker()
    
    target = {
        'name': 'Aariyan S',
        'usernames': ['aariyan007', 'aariyan_07'],
        'bio': 'Computer science student. Love coding and AI.'
    }
    
    candidates = [
        {
            'platform': 'twitter',
            'username': 'aariyan_official',
            'display_name': 'Aariyan',
            'bio': 'CS student | AI enthusiast'
        },
        {
            'platform': 'reddit',
            'username': 'random_user123',
            'display_name': 'John Doe',
            'bio': 'Just browsing'
        }
    ]
    
    matches = linker.link_accounts(target, candidates, confidence_threshold=0.6)
    
    for match in matches:
        print(f"\nMatched: {match['platform']} - @{match['username']}")
        print(f"Confidence: {match['match_confidence']:.1f}%")
        print(f"Reasons: {', '.join(match['match_reasons'])}")