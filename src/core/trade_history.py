"""Temporary file-based persistence for Phase 1."""

import os
import json
from typing import Optional

def calculate_total_profit(file_path: str) -> Optional[float]:
    """
    Reads a JSON trade-history file and sums each trade's profit.usdt field.
    
    Args:
        file_path (str): The path to the JSON file containing the trade history.
        
    Returns:
        Optional[float]: The sum of all 'profit.usdt' fields, or None if the file
                         does not exist or is malformed.
    """
    if not os.path.exists(file_path):
        print("File does not exist.")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except json.JSONDecodeError:
        print("File is empty or has invalid JSON format.")
        return None

    total_profit = 0.0
    for item in data:
        if isinstance(item, dict):
            profit = item.get('profit', {}).get('usdt', None)
            if isinstance(profit, (int, float)):
                total_profit += profit

    return total_profit
