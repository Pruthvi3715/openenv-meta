from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from env import BankReconciliationEnv, Action, Observation
from env.graders import grade_task
from tasks import Task1Categorize, Task2DecodeUpi, Task3Reconcile


class ActionInput(BaseModel):
    transaction_id: str
    assigned_category: str
    merchant_label: str
    flag_type: Optional[str] = None


class ResetInput(BaseModel):
    seed: Optional[int] = 42
    task_type: Optional[str] = "categorize"


class StepInput(BaseModel):
    action: ActionInput


class GraderInput(BaseModel):
    task_name: str


class TaskInfo(BaseModel):
    name: str
    description: str
    difficulty: str
    action_schema: dict


current_task: Optional[BankReconciliationEnv] = None
current_task_type: str = "categorize"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global current_task
    current_task = BankReconciliationEnv(seed=42)
    yield


app = FastAPI(title="Bank Reconciliation Env", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/tasks")
async def get_tasks():
    tasks = [
        Task1Categorize(),
        Task2DecodeUpi(),
        Task3Reconcile(),
    ]
    return {
        "tasks": [task.get_info() for task in tasks],
        "action_schema": {
            "transaction_id": "str",
            "assigned_category": "Food | Travel | Utilities | Shopping | Unknown",
            "merchant_label": "str",
            "flag_type": "None | duplicate | anomaly",
        },
    }


@app.post("/reset")
async def reset(input_data: ResetInput):
    global current_task, current_task_type
    current_task_type = input_data.task_type
    current_task = BankReconciliationEnv(seed=input_data.seed)
    obs = current_task.reset(task_type=input_data.task_type)
    return {
        "observation": obs.model_dump(),
        "done": False,
    }


@app.post("/step")
async def step(input_data: StepInput):
    global current_task
    if current_task is None:
        raise HTTPException(status_code=400, detail="Environment not reset")

    action = Action(**input_data.action.model_dump())
    obs = current_task.step(action)

    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.get("/state")
async def get_state():
    global current_task
    if current_task is None:
        raise HTTPException(status_code=400, detail="Environment not reset")

    return current_task.state.model_dump()


@app.post("/grader")
async def grade(input_data: GraderInput):
    global current_task
    if current_task is None:
        raise HTTPException(status_code=400, detail="Environment not reset")

    score = grade_task(
        input_data.task_name,
        current_task.state.resolved_transactions,
        current_task.state,
    )

    return {"task": input_data.task_name, "score": score}


@app.post("/baseline")
async def run_baseline():
    from baseline.inference import run_all_tasks

    results = run_all_tasks()
    return results


@app.get("/health")
async def health():
    return {"status": "healthy"}
