import sys
from pathlib import Path
from typing import Dict, Any, Optional

RULEARENA_PATH = Path(__file__).parent.parent.parent.parent.parent / "external" / "RuleArena"
TAX_DIR = str(RULEARENA_PATH / "tax")

if TAX_DIR not in sys.path:
    sys.path.insert(0, TAX_DIR)

try:
    from structured_forms import TaxPayer
    from micro_evaluation import compute_answer
except Exception as e:
    print(f"Warning: Could not load tax reference implementation: {e}")
    TaxPayer = None
    compute_answer = None


def compute_tax_fee(info: Dict[str, Any]) -> Optional[float]:
    """
    Compute tax amount from a RuleArena tax problem dict.

    Args:
        info: dict with a "pydantic" key whose value is the TaxPayer fields

    Returns:
        Amount owed (positive) or overpaid (negative), or None on failure
    """
    if TaxPayer is None or compute_answer is None:
        return None

    try:
        tp = TaxPayer(**info["pydantic"])
        amount, _ = compute_answer(tp)
        return float(amount)
    except Exception as e:
        print(f"Error computing tax fee: {e}")
        return None
