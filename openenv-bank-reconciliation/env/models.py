from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel

from openenv.core.env_server import Action as BaseAction
from openenv.core.env_server import Observation as BaseObservation


class Transaction(BaseModel):
    id: str
    amount: float
    merchant_raw: str
    timestamp: datetime
    upi_ref: Optional[str] = None
    account_type: str


class Observation(BaseObservation):
    transactions: List[Transaction]
    resolved_count: int
    episode_step: int
    context_hints: Dict[str, str]


class Action(BaseAction):
    transaction_id: str
    assigned_category: str
    merchant_label: str
    flag_type: Optional[str] = None


class Reward(BaseModel):
    value: float
    breakdown: Dict[str, float]
    done: bool


class State(BaseModel):
    episode_id: Optional[str] = None
    step_count: int = 0
    all_transactions: List[Transaction] = []
    resolved_transactions: Dict[str, dict] = {}
    ground_truth_categories: Dict[str, str] = {}
    ground_truth_merchants: Dict[str, str] = {}
    unresolved_steps: Dict[str, int] = {}
    duplicates: List[tuple] = []
    anomalies: List[str] = []


CATEGORIES = ["Food", "Travel", "Utilities", "Shopping", "Unknown"]

MERCHANT_CATEGORIES = {
    "SWIGGY": "Food",
    "ZOMATO": "Food",
    "IRCTC": "Travel",
    "AMAZON": "Shopping",
    "FLIPKART": "Shopping",
    "NETFLIX": "Utilities",
    "BESCOM": "Utilities",
    "BBNL": "Utilities",
    "PHONEPE": "Unknown",
    "PAYTM": "Unknown",
}
