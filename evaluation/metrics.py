import re
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize

def calculate_em(pred, gold):
    """Calculate Exact Match (case-insensitive)"""
    return int(pred.strip().lower() == gold.strip().lower())

def calculate_f1(pred, gold):
    """Calculate token-level F1 score"""
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    
    common_tokens = set(pred_tokens) & set(gold_tokens)
    if len(common_tokens) == 0:
        return 0.0
    
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gold_tokens)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
    return f1

def calculate_rouge(pred, gold):
    """Calculate ROUGE-L score using LCS-based approach"""
    # Tokenize sentences and words
    pred_sents = sent_tokenize(pred)
    gold_sents = sent_tokenize(gold)
    
    pred_words = [word_tokenize(sent) for sent in pred_sents]
    gold_words = [word_tokenize(sent) for sent in gold_sents]
    
    # Flatten word lists
    pred_flat = [w for sent in pred_words for w in sent]
    gold_flat = [w for sent in gold_words for w in sent]
    
    # Calculate longest common subsequence
    lcs_length = _lcs_length(pred_flat, gold_flat)
    
    if len(pred_flat) == 0 or len(gold_flat) == 0:
        return 0.0
        
    precision = lcs_length / len(pred_flat) if pred_flat else 0
    recall = lcs_length / len(gold_flat) if gold_flat else 0
    
    if precision == 0 or recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def _lcs_length(a, b):
    """Calculate length of longest common subsequence"""
    if not a or not b:
        return 0
    
    m, n = len(a), len(b)
    # Initialize 2D array for dynamic programming
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

def calculate_medical_accuracy(pred, gold):
    """Medical-specific accuracy metric (placeholder)"""
    # In real implementation, this would use clinical NLP models
    # For now, use keyword matching for demo
    medical_keywords = ['heart', 'cardio', 'angina', 'stroke', 'attack', 
                        'blood pressure', 'cholesterol', 'aortic', 'valve',
                        'beta-blocker', 'anticoagulation', 'statin',
                        'arrhythmia', 'fibrillation']
    
    pred_score = sum(1 for kw in medical_keywords if kw.lower() in pred.lower())
    gold_score = sum(1 for kw in medical_keywords if kw.lower() in gold.lower())
    
    if gold_score == 0:
        return 0.0
    
    return min(pred_score / gold_score, 1.0)

def evaluate_answer(pred, gold):
    """Comprehensive evaluation of answer quality"""
    return {
        'em': calculate_em(pred, gold),
        'f1': calculate_f1(pred, gold),
        'rouge': calculate_rouge(pred, gold),
        'medical_acc': calculate_medical_accuracy(pred, gold)
    }