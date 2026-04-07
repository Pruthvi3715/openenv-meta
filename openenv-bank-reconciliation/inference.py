import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from env import BankReconciliationEnv, Action
from env.graders import grade_task

load_dotenv(Path(__file__).resolve().parent / ".env")

API_BASE_URL = os.getenv(
    "API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")
HF_TOKEN = os.getenv("HF_TOKEN", "")

client = OpenAI(
    api_key=HF_TOKEN or os.getenv("GEMINI_API_KEY"),
    base_url=API_BASE_URL,
)

SYSTEM_PROMPT = """You are an AI agent that reconciles Indian bank statements.

You need to process transactions and return actions with the following schema:
{
    "transaction_id": "string - the ID of the transaction to process",
    "assigned_category": "string - one of: Food, Travel, Utilities, Shopping, Unknown",
    "merchant_label": "string - human readable merchant name",
    "flag_type": "string or null - null, 'duplicate', or 'anomaly'"
}

Available merchants and their categories:
- SWIGGY, ZOMATO -> Food
- IRCTC -> Travel
- AMAZON, FLIPKART -> Shopping
- NETFLIX -> Utilities
- BESCOM, BBNL -> Utilities
- PHONEPE, PAYTM -> Unknown

UPI references follow patterns like:
- UPI-9182XXXX@paytm.okhdfc
- UPI-1234567890@icici.okicici
- NEFT/123456/raun

When you see a UPI reference, try to decode the merchant from the reference.
Look for merchant hints in the UPI handle (e.g., @paytm means PAYTM, swig means SWIGGY).

Analyze each transaction carefully and return ONLY a valid JSON object with your action.
Do NOT wrap it in markdown code fences. Return raw JSON only."""


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str] = None
):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def get_action_from_llm(
    transactions: list, context_hints: Dict[str, str], step: int
) -> Action:
    transaction_list = []
    for t in transactions:
        transaction_list.append(
            f"ID: {t['id']}, Amount: {t['amount']}, Merchant: {t['merchant_raw']}, "
            f"Type: {t['account_type']}, UPI: {t.get('upi_ref', 'N/A')}"
        )

    user_prompt = f"""
Current step: {step}
Context hints: {context_hints}

Transactions to process (choose one):
{chr(10).join(transaction_list)}

Return a JSON object with your action. Only JSON, no markdown:"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    content = response.choices[0].message.content
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    action_data = json.loads(content)

    if action_data.get("flag_type") in ("null", "None", "none", ""):
        action_data["flag_type"] = None

    return Action(**action_data)


def run_task(task_name: str, seed: int = 42) -> Dict[str, Any]:
    log_start(task=task_name, env="bank-reconciliation", model=MODEL_NAME)

    env = BankReconciliationEnv(seed=seed)

    task_type_map = {
        "categorize": "categorize",
        "decode_upi": "decode_upi",
        "full_reconciliation": "full",
    }

    task_type = task_type_map.get(task_name, "full")
    obs = env.reset(task_type=task_type, seed=seed)

    step_count = 0
    max_steps = 60
    rewards_list: List[float] = []
    last_action_str = "null"
    last_error = None

    import time

    while step_count < max_steps:
        if not obs.transactions:
            break

        time.sleep(4)

        try:
            action = get_action_from_llm(
                [t.model_dump() for t in obs.transactions],
                obs.context_hints,
                step_count,
            )
            last_action_str = f"{action.transaction_id}:{action.assigned_category}:{action.merchant_label}:{action.flag_type}"
            obs = env.step(action)
        except Exception as e:
            last_error = str(e)
            if "429" in str(e):
                time.sleep(15)
                continue

            if obs.transactions:
                fallback = Action(
                    transaction_id=obs.transactions[0].id,
                    assigned_category="Unknown",
                    merchant_label=obs.transactions[0].merchant_raw,
                    flag_type=None,
                )
                last_action_str = f"{fallback.transaction_id}:{fallback.assigned_category}:{fallback.merchant_label}:{fallback.flag_type}"
                obs = env.step(fallback)

        reward = obs.reward if obs.reward is not None else 0.0
        done = obs.done if obs.done is not None else False
        rewards_list.append(reward)

        log_step(
            step=step_count,
            action=last_action_str,
            reward=reward,
            done=done,
            error=last_error,
        )

        step_count += 1
        last_error = None

        if len(env.state.resolved_transactions) >= len(env.state.all_transactions):
            break

    score = grade_task(task_name, env.state.resolved_transactions, env.state)
    if isinstance(score, dict):
        score = score.get("score", 0.0)
    score_float = float(score)

    success = score_float >= 0.5

    log_end(success=success, steps=step_count, score=score_float, rewards=rewards_list)

    return {
        "task": task_name,
        "score": score_float,
        "steps": step_count,
        "resolved": len(env.state.resolved_transactions),
        "total": len(env.state.all_transactions),
        "model": MODEL_NAME,
    }


def run_all_tasks() -> Dict[str, Any]:
    results = {}

    results["categorize"] = run_task("categorize", seed=42)
    results["decode_upi"] = run_task("decode_upi", seed=42)
    results["full_reconciliation"] = run_task("full_reconciliation", seed=42)

    return results


if __name__ == "__main__":
    results = run_all_tasks()
