import uuid
from typing import List, Optional

from .models import (
    Action,
    Observation,
    Reward,
    State,
    Transaction,
    MERCHANT_CATEGORIES,
    CATEGORIES,
)
from .data_generator import generate_full_statement, get_context_hints


class BankReconciliationEnv:
    SUPPORTS_CONCURRENT_SESSIONS = True

    MAX_STEPS = 60
    BATCH_SIZE = 10

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._state = State()
        self._transactions: List[Transaction] = []
        self._resolved: dict = {}
        self._last_reward = 0.0
        self._last_done = False

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        num_transactions: int = 30,
        task_type: str = "full",
        **kwargs,
    ) -> Observation:
        if seed is not None:
            self._seed = seed

        self._seed += 1
        self._state = State(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
        )
        self._resolved = {}

        if task_type == "categorize":
            num_transactions = 10
            inject_duplicates = 0
            inject_anomalies = 0
            clear_only = True
        elif task_type == "decode_upi":
            num_transactions = 10
            inject_duplicates = 0
            inject_anomalies = 0
            clear_only = False
        elif task_type == "full":
            inject_duplicates = 3
            inject_anomalies = 2
            clear_only = False
        else:
            inject_duplicates = 0
            inject_anomalies = 0
            clear_only = False

        (
            self._transactions,
            gt_categories,
            gt_merchants,
            duplicates,
            anomalies,
        ) = generate_full_statement(
            num_transactions=num_transactions,
            seed=self._seed,
            inject_duplicates=inject_duplicates,
            inject_anomalies=inject_anomalies,
            clear_only=clear_only,
        )

        self._state.all_transactions = self._transactions
        self._state.duplicates = duplicates
        self._state.anomalies = anomalies
        self._state.ground_truth_categories = gt_categories
        self._state.ground_truth_merchants = gt_merchants

        batch = self._get_batch()
        context_hints = get_context_hints(self._transactions)

        return Observation(
            transactions=batch,
            resolved_count=len(self._resolved),
            episode_step=self._state.step_count,
            context_hints=context_hints,
        )

    def _get_batch(self) -> List[Transaction]:
        unresolved = [t for t in self._transactions if t.id not in self._resolved]
        return unresolved[: self.BATCH_SIZE]

    def step(self, action: Action, timeout_s=None, **kwargs) -> Observation:
        self._state.step_count += 1

        reward_breakdown = {}
        reward_value = 0.0

        transaction = None
        for t in self._transactions:
            if t.id == action.transaction_id:
                transaction = t
                break

        if transaction is None:
            reward_value = -0.5
            reward_breakdown["invalid_transaction"] = -0.5
        else:
            if transaction.id in self._resolved:
                reward_value = 0.0
                reward_breakdown["already_resolved"] = 0.0
            else:
                ground_truth_category = self._state.ground_truth_categories.get(
                    transaction.id, "Unknown"
                )
                ground_truth_merchant = self._state.ground_truth_merchants.get(
                    transaction.id, ""
                )

                if action.assigned_category == ground_truth_category:
                    reward_value += 1.0
                    reward_breakdown["exact_category"] = 1.0
                elif self._get_parent_category(
                    action.assigned_category
                ) == self._get_parent_category(ground_truth_category):
                    reward_value += 0.5
                    reward_breakdown["parent_category"] = 0.5
                else:
                    reward_breakdown["category_mismatch"] = 0.0

                if ground_truth_merchant:
                    fuzzy_score = self._fuzzy_match(
                        action.merchant_label.lower(),
                        ground_truth_merchant.lower(),
                    )
                    if fuzzy_score > 0.85:
                        reward_value += 0.8
                        reward_breakdown["merchant_match"] = 0.8
                    else:
                        reward_breakdown["merchant_score"] = fuzzy_score * 0.5

                is_duplicate = any(
                    action.transaction_id in pair for pair in self._state.duplicates
                )
                is_anomaly = action.transaction_id in self._state.anomalies

                if action.flag_type == "duplicate" and is_duplicate:
                    reward_breakdown["correct_duplicate_flag"] = 1.0
                elif action.flag_type == "duplicate" and not is_duplicate:
                    reward_value -= 0.3
                    reward_breakdown["false_duplicate_flag"] = -0.3

                if action.flag_type == "anomaly" and is_anomaly:
                    reward_breakdown["correct_anomaly_flag"] = 1.0
                elif action.flag_type == "anomaly" and not is_anomaly:
                    reward_value -= 0.3
                    reward_breakdown["false_anomaly_flag"] = -0.3

                self._resolved[transaction.id] = {
                    "category": action.assigned_category,
                    "merchant_label": action.merchant_label,
                    "flag_type": action.flag_type,
                }
                self._state.resolved_transactions = self._resolved

        for tid in self._resolved:
            if tid not in self._state.unresolved_steps:
                self._state.unresolved_steps[tid] = 0
            self._state.unresolved_steps[tid] += 1

            if self._state.unresolved_steps[tid] > 3:
                reward_value -= 0.1
                reward_breakdown["stale_resolution"] = -0.1

        all_resolved = len(self._resolved) >= len(self._transactions)
        done = all_resolved or self._state.step_count >= self.MAX_STEPS

        if all_resolved:
            reward_value += 2.0
            reward_breakdown["completion_bonus"] = 2.0

        self._last_reward = reward_value
        self._last_done = done

        batch = self._get_batch()
        context_hints = get_context_hints(self._transactions)

        return Observation(
            transactions=batch,
            resolved_count=len(self._resolved),
            episode_step=self._state.step_count,
            context_hints=context_hints,
            reward=reward_value,
            done=done,
        )

    def _get_parent_category(self, category: str) -> str:
        return category

    def _fuzzy_match(self, s1: str, s2: str) -> float:
        if s1 == s2:
            return 1.0
        if s1 in s2 or s2 in s1:
            return 0.9
        if s1[:4] == s2[:4]:
            return 0.85
        return 0.0

    @property
    def state(self) -> State:
        return self._state
