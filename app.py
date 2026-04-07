"""FastAPI server — OpenEnv HTTP specification for DebtRecoveryEnv."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from env.environment import DebtRecoveryEnv
from env.models import CollectionAction, PortfolioObservation, CollectionReward
from tasks.graders import grade

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DebtRecovery·ENV",
    description="Indian NBFC Loan Collections RL Environment — OpenEnv Compliant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global environment state (single session for hackathon scope) ────────────

_env: Optional[DebtRecoveryEnv] = None


# ── Request / Response models ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "task1_single_cooperative"
    seed: int = 42


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Dict[str, Any]


class TaskInfo(BaseModel):
    id: str
    difficulty: str
    max_steps: int
    reward_range: list
    description: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect the base URL to the dashboard."""
    return RedirectResponse(url="/dashboard", status_code=307)


@app.post("/reset")
async def reset_env(request: Optional[ResetRequest] = None) -> Dict[str, Any]:
    """Reset the environment with a task and seed. Returns initial observation."""
    global _env
    try:
        payload = request or ResetRequest()
        _env = DebtRecoveryEnv(task_id=payload.task_id, seed=payload.seed)
        obs = await _env.reset()
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step")
async def step_env(action: CollectionAction) -> StepResponse:
    """Take one step. Returns new observation and reward."""
    global _env
    if _env is None:
        raise HTTPException(
            status_code=400, detail="Environment not initialized. Call /reset first."
        )
    try:
        obs, reward = await _env.step(action)
        return StepResponse(
            observation=obs.model_dump(),
            reward=reward.model_dump(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def get_state() -> Dict[str, Any]:
    """Return full internal state including episode log."""
    global _env
    if _env is None:
        raise HTTPException(
            status_code=400, detail="Environment not initialized. Call /reset first."
        )
    return await _env.state()


@app.get("/tasks")
async def list_tasks():
    """List all available tasks with metadata."""
    return [
        TaskInfo(
            id="task1_single_cooperative",
            difficulty="easy",
            max_steps=10,
            reward_range=[0.0, 1.0],
            description="Single cooperative borrower. Establish contact and secure PTP.",
        ),
        TaskInfo(
            id="task2_portfolio_mixed",
            difficulty="medium",
            max_steps=30,
            reward_range=[0.0, 1.0],
            description="10 mixed accounts. Balance recovery, compliance, sentiment.",
        ),
        TaskInfo(
            id="task3_portfolio_adversarial",
            difficulty="hard",
            max_steps=60,
            reward_range=[0.0, 1.0],
            description="25 adversarial accounts with mid-episode regulatory shock.",
        ),
    ]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "env": "debt-recovery-collections"}


@app.get("/grade")
async def grade_episode() -> Dict[str, Any]:
    """Grade the current episode using the task-specific grader."""
    global _env
    if _env is None:
        raise HTTPException(
            status_code=400, detail="Environment not initialized. Call /reset first."
        )
    state = await _env.state()
    score = grade(
        task_id=_env.task_id,
        episode_log=state["episode_log"],
        final_state=state,
    )
    return {
        "task_id": _env.task_id,
        "seed": _env.seed,
        "score": score,
        "steps_taken": state["step_count"],
        "done": state["done"],
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the monitoring dashboard."""
    dashboard_path = Path(__file__).parent / "dashboard" / "index.html"
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return HTMLResponse(content=dashboard_path.read_text(encoding="utf-8"))


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=True)
