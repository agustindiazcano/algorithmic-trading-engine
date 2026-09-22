import json

# We haven't implemented calculate_total_profit in src/core/trade_history.py yet
from src.core.trade_history import calculate_total_profit


def test_calculate_total_profit_valid_file(tmp_path):
    # Asserts that calling the function with a valid path to a JSON file containing sample trades with profit.usdt values returns the correct sum.
    test_file = tmp_path / "trade_history.json"
    data = [
        {"profit": {"usdt": 10.5}},
        {"profit": {"usdt": 5.0}},
        {"profit": {"usdt": -2.0}},
        {"profit": {}} # missing usdt
    ]
    with open(test_file, 'w') as f:
        json.dump(data, f)
        
    result = calculate_total_profit(str(test_file))
    assert result == 13.5

def test_calculate_total_profit_nonexistent_file():
    # Asserts that calling it with a nonexistent path returns None
    result = calculate_total_profit("nonexistent_file.json")
    assert result is None

def test_calculate_total_profit_malformed_json(tmp_path):
    # Asserts that a malformed/empty JSON file is handled the same way the legacy code does (returns None, doesn't crash).
    test_file = tmp_path / "malformed.json"
    with open(test_file, 'w') as f:
        f.write("{ invalid json")
        
    result = calculate_total_profit(str(test_file))
    assert result is None

def test_calculate_total_profit_variable_regression(tmp_path):
    # Asserts that constructing the call with the literal variable name (regression)
    # won't just fail silently but actually reads the correct file if we pass the right variable
    
    # We test that passing the variable content works, instead of the string literal
    TRADE_HISTORY_FILE = tmp_path / "actual_history.json"
    with open(TRADE_HISTORY_FILE, 'w') as f:
        json.dump([{"profit": {"usdt": 42.0}}], f)
        
    # Correct call uses the variable
    result = calculate_total_profit(str(TRADE_HISTORY_FILE))
    assert result == 42.0
    
    # Original bug was calculate_total_profit("TRADE_HISTORY_FILE")
    # This should return None because "TRADE_HISTORY_FILE" file doesn't exist
    buggy_result = calculate_total_profit("TRADE_HISTORY_FILE")
    assert buggy_result is None
