"""
Evaluation metrics for summarization.
"""
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate ROUGE scores.
    
    Args:
        predictions: List of predicted summaries
        references: List of reference summaries
        
    Returns:
        Dictionary of ROUGE scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge-1': sum(rouge1_scores) / len(rouge1_scores),
        'rouge-2': sum(rouge2_scores) / len(rouge2_scores),
        'rouge-l': sum(rougeL_scores) / len(rougeL_scores)
    }


def calculate_bleu_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate BLEU scores.
    
    Args:
        predictions: List of predicted summaries
        references: List of reference summaries
        
    Returns:
        Dictionary of BLEU scores
    """
    smoothing = SmoothingFunction().method1
    
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = [ref.split()]
        
        try:
            bleu1 = sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            bleu2 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu3 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
            bleu4 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
            
            bleu1_scores.append(bleu1)
            bleu2_scores.append(bleu2)
            bleu3_scores.append(bleu3)
            bleu4_scores.append(bleu4)
        except:
            continue
    
    if not bleu1_scores:
        return {'bleu-1': 0.0, 'bleu-2': 0.0, 'bleu-3': 0.0, 'bleu-4': 0.0}
    
    return {
        'bleu-1': sum(bleu1_scores) / len(bleu1_scores),
        'bleu-2': sum(bleu2_scores) / len(bleu2_scores),
        'bleu-3': sum(bleu3_scores) / len(bleu3_scores),
        'bleu-4': sum(bleu4_scores) / len(bleu4_scores)
    }