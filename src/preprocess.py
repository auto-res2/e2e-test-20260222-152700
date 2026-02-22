"""Dataset preprocessing for GSM8K and other datasets."""

import re
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset


def load_gsm8k(
    split: str = "test",
    num_samples: Optional[int] = None,
    cache_dir: str = ".cache",
) -> List[Dict[str, str]]:
    """
    Load GSM8K dataset.
    
    Args:
        split: Dataset split (train or test)
        num_samples: Number of samples to load (None for all)
        cache_dir: Cache directory for datasets
        
    Returns:
        List of dictionaries with 'question' and 'answer' keys
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split=split,
        cache_dir=str(cache_path),
    )
    
    # Convert to list of dicts
    samples = []
    for i, example in enumerate(dataset):
        if num_samples is not None and i >= num_samples:
            break
            
        # Extract numeric answer from the answer field
        # GSM8K answers are in format "#### <number>"
        answer_text = example["answer"]
        numeric_answer = extract_numeric_answer(answer_text)
        
        samples.append({
            "question": example["question"],
            "answer": answer_text,
            "numeric_answer": numeric_answer,
        })
    
    return samples


def extract_numeric_answer(text: str) -> float:
    """
    Extract numeric answer from GSM8K answer text.
    
    GSM8K answers are formatted as: "#### <number>"
    
    Args:
        text: Answer text
        
    Returns:
        Numeric answer as float
    """
    # Look for "#### <number>" pattern
    match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if match:
        # Remove commas and convert to float
        number_str = match.group(1).replace(",", "")
        return float(number_str)
    
    # Fallback: try to find any number at the end
    match = re.search(r"(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$", text)
    if match:
        number_str = match.group(1).replace(",", "")
        return float(number_str)
    
    raise ValueError(f"Could not extract numeric answer from: {text}")


def extract_final_answer_from_response(response: str) -> Optional[float]:
    """
    Extract final numeric answer from model response.
    
    Looks for patterns like:
    - FINAL: <number>
    - Final Answer: <number>
    - #### <number>
    - The answer is <number>
    
    Args:
        response: Model response text
        
    Returns:
        Extracted numeric answer or None if not found
    """
    # Try FINAL: pattern (TIL-RV format)
    match = re.search(r"FINAL:\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", response, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(",", ""))
    
    # Try #### pattern (GSM8K format)
    match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", response)
    if match:
        return float(match.group(1).replace(",", ""))
    
    # Try "Final Answer:" pattern
    match = re.search(r"Final Answer:\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", response, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(",", ""))
    
    # Try "The answer is" pattern
    match = re.search(r"(?:the answer is|answer:)\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", response, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(",", ""))
    
    # Try to find last number in response
    numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", response)
    if numbers:
        return float(numbers[-1].replace(",", ""))
    
    return None
