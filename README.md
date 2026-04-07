---
title: Debt Recovery Env
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# DebtRecovery ENV

# DebtRecovery·ENV 🏦

**OpenEnv-Compliant RL Environment for Indian NBFC Loan Collections**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v1.0.0-blue)](https://openenv.org)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-orange)](https://fastapi.tiangolo.com)

---

## 1. Environment Overview

DebtRecovery·ENV simulates the real-world challenge of managing overdue loan accounts at an Indian Non-Banking Financial Company (NBFC). An AI agent manages 1–25 overdue loan accounts per episode, learning the delicate balance between **empathy and pressure** under strict **RBI regulatory constraints**.

The environment models:
- **Probabilistic borrower responses** based on employment, sentiment, DPD, and contact history
- **RBI Fair Practices Code** as hard action masks (DNC, max contact limits, legal escalation thresholds)
- **Multi-account portfolio management** with shared call/visit budgets
- **Composite shaped rewards** capturing recovery, compliance, sentiment, and time pressure

## 2. Domain Motivation

India's NBFC sector manages **₹14+ lakh crore** in outstanding credit. Collections operations face:
- **Scale**: Millions of overdue accounts requiring prioritization
- **Regulation**: RBI's Fair Practices Code mandates strict contact limits and borrower protection
- **Complexity**: Borrower behavior varies by employment type, sentiment, and hardship status
- **Trade-offs**: Aggressive collection risks complaints; passive approaches risk write-offs

An RL/agent approach can learn optimal contact strategies that maximize recovery while maintaining regulatory compliance — a problem too complex for static rule-based systems.

## 3. Observation Space

| Field | Type | Description | Range |
|-------|------|-------------|-------|
| `account_id` | str | Unique borrower identifier | BRW_001–BRW_500 |
| `outstanding_inr` | float | Outstanding loan amount in INR | 8,000–500,000 |
| `dpd` | int | Days Past Due | 5–180 |
| `credit_score` | int | Borrower credit score | 450–780 |
| `employment_type` | enum | Employment category | salaried/self_employed/daily_wage/unemployed |
| `income_band` | str | Income bracket | low/low_mid/mid/high |
| `city_tier` | int | City tier classification | 1/2/3 |
| `hardship_flag` | bool | Financial hardship indicator | true/false |
| `legal_stage` | enum | Current legal proceeding stage | pre_legal/notice_sent/sarfaesi/drt/written_off |
| `dnc_status` | bool | Do Not Call registry status | true/false |
| `complaint_count` | int | Number of filed complaints | 0–5 |
| `sentiment` | enum | Borrower disposition | cooperative/avoidant/hostile/ghost |
| `contact_attempts_today` | int | Calls made today | 0–5 |
| `ptp_active` | bool | Active promise-to-pay | true/false |
| `days_since_last_payment` | int | Days since last payment | 0–200 |
| `is_resolved` | bool | Account fully settled | true/false |
| `session_calls_remaining` | int | Shared call budget | 0–75 |
| `session_visits_remaining` | int | Shared field visit budget | 0–5 |
| `regulatory_audit_active` | bool | RBI audit mode (Task 3) | true/false |

## 4. Action Space

| Action | Category | Description | Constraints |
|--------|----------|-------------|-------------|
| `CALL_MORNING` | Contact | Morning phone call | Max 3/day, no DNC |
| `CALL_AFTERNOON` | Contact | Afternoon phone call | Max 3/day, no DNC |
| `CALL_EVENING` | Contact | Evening phone call | Max 3/day, no DNC, blocked if complaints≥2 |
| `SMS_REMINDER` | Contact | Payment reminder SMS | No DNC |
| `SMS_WARNING` | Contact | Warning SMS | No DNC, blocked if complaints≥2 |
| `SMS_SETTLEMENT` | Contact | Settlement offer SMS | No DNC |
| `EMAIL_FORMAL` | Contact | Formal email notice | — |
| `EMAIL_EMPATHETIC` | Contact | Empathetic email | — |
| `WHATSAPP_NUDGE` | Contact | WhatsApp message | — |
| `FIELD_VISIT` | Contact | In-person visit | Requires visit budget |
| `NO_CONTACT` | Passive | Skip this step | Always valid |
| `NEGOTIATE_PTP` | Negotiate | Promise-to-pay negotiation | — |
| `OFFER_RESTRUCTURE` | Negotiate | Loan restructuring offer | — |
| `OFFER_OTS` | Negotiate | One-time settlement | DPD ≥ 60 |
| `GRANT_DEFERMENT` | Negotiate | Payment deferment | — |
| `ESCALATE_LEGAL` | Escalate | Legal proceedings | DPD ≥ 90 |
| `FLAG_WRITEOFF` | Escalate | Mark for write-off | Written-off stage only |

## 5. Reward Function

| Component | Formula | Range | Purpose |
|-----------|---------|-------|---------|
| Recovery Signal | Full: +1.0 × (outstanding/mean), Partial: +0.4 × (paid/outstanding), PTP: +0.15 | [0, 2+] | Incentivize payments |
| Contact Quality | Answered: +0.05, Productive: +0.08, Voicemail: +0.01, Failed 3rd+: −0.03 | [−0.03, 0.08] | Reward effective contact |
| Sentiment Delta | Improve: +0.06 to +0.12, Deteriorate: −0.08 | [−0.08, 0.12] | Encourage trust-building |
| Compliance | +0.02/step if no violation | [0, 0.02] | Reward rule-following |
| Complaint Penalty | −0.50 per complaint | [−0.50, 0] | Punish poor treatment |
| Violation Penalty | DNC: −1.50, Premature legal: −0.30 | [−1.50, 0] | Enforce regulations |
| Time Decay | −0.02/step if unresolved DPD>90 | [−0.02, 0] | Create urgency |

## 6. Tasks

| Task ID | Difficulty | Accounts | Max Steps | Grader Logic |
|---------|-----------|----------|-----------|-------------|
| `task1_single_cooperative` | Easy | 1 | 10 | 50% recovery + 30% contact + 20% compliance |
| `task2_portfolio_mixed` | Medium | 10 | 30 | 50% recovery + 30% compliance + 20% resolution |
| `task3_portfolio_adversarial` | Hard | 25 | 60 | (40% recovery + 25% compliance + 20% legal + 15% shock) × multiplier |

## 7. Regulatory Constraints

- ✅ Maximum 3 calls per day per borrower (1 during audit)
- ✅ No contact with DNC-registered borrowers via call or SMS
- ✅ Legal escalation only for DPD ≥ 90
- ✅ OTS offers only for DPD ≥ 60
- ✅ Complaint cooling: block evening calls and warning SMS if complaints ≥ 2
- ✅ Written-off accounts limited to FLAG_WRITEOFF or NO_CONTACT
- ✅ Field visits constrained by session budget
- ✅ Regulatory audit shock at step 30 in Task 3 (cuts call quota by 40%)

## 8. Setup & Installation

### Docker (Recommended)
```bash
docker build -t debt-recovery-env .
docker run -p 7860:7860 debt-recovery-env
```

### Local
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run Tests
```bash
pytest tests/ -v
```

### Submission Inference
```bash
python inference.py
```

Required environment variables for `inference.py`:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Optional:

- `ENV_BASE_URL` if you want to point to an already running server
- `LOCAL_IMAGE_NAME` only if you later switch to a `from_docker_image()` workflow

## 9. Usage Examples

### Python — Direct
```python
import asyncio
from env.environment import DebtRecoveryEnv
from env.models import CollectionAction, ActionType

async def main():
    env = DebtRecoveryEnv(task_id="task1_single_cooperative", seed=42)
    obs = await env.reset()
    
    action = CollectionAction(
        account_id=obs.accounts[0].account_id,
        action_type=ActionType.CALL_MORNING,
    )
    new_obs, reward = await env.step(action)
    print(f"Reward: {reward.total}, Done: {reward.episode_done}")

asyncio.run(main())
```

### HTTP — OpenEnv Client
```python
import httpx

client = httpx.Client(base_url="http://localhost:7860")

# Reset
obs = client.post("/reset", json={"task_id": "task1_single_cooperative", "seed": 42}).json()

# Step
result = client.post("/step", json={
    "account_id": "BRW_001",
    "action_type": "CALL_MORNING"
}).json()

# Grade
score = client.get("/grade").json()
print(f"Score: {score['score']}")
```

## 10. Baseline Scores

| Task | Seed | Model | Score |
|------|------|-------|-------|
| task1 (Easy) | 42 | gpt-4o-mini | 0.81 |
| task1 (Easy) | 137 | gpt-4o-mini | 0.78 |
| task1 (Easy) | 256 | gpt-4o-mini | 0.83 |
| task2 (Medium) | 1001 | gpt-4o-mini | 0.52 |
| task2 (Medium) | 2048 | gpt-4o-mini | 0.47 |
| task2 (Medium) | 3333 | gpt-4o-mini | 0.49 |
| task3 (Hard) | 9999 | gpt-4o-mini | 0.28 |
| task3 (Hard) | 8421 | gpt-4o-mini | 0.31 |
| task3 (Hard) | 7070 | gpt-4o-mini | 0.26 |

**Means**: Task 1: 0.807 | Task 2: 0.493 | Task 3: 0.283

## 11. Architecture

```
debt-recovery-env/
├── openenv.yaml              ← OpenEnv specification
├── Dockerfile                ← Container config
├── app.py                    ← FastAPI server (OpenEnv HTTP spec)
├── env/
│   ├── models.py             ← Pydantic v2 typed models
│   ├── environment.py        ← DebtRecoveryEnv main class
│   ├── borrower_simulator.py ← Probabilistic borrower responses
│   ├── regulatory_engine.py  ← RBI action masking
│   ├── reward_engine.py      ← Composite shaped reward
│   └── portfolio_manager.py  ← Multi-account budget tracking
├── tasks/
│   ├── task1_easy.py         ← Single cooperative (10 steps)
│   ├── task2_medium.py       ← Mixed portfolio (30 steps)
│   ├── task3_hard.py         ← Adversarial + shock (60 steps)
│   └── graders.py            ← Deterministic 0.0–1.0 scoring
├── data/
│   ├── borrower_profiles.json← 500 synthetic profiles
│   └── scenario_seeds.json   ← Fixed seeds per task
├── baseline/
│   ├── baseline_agent.py     ← OpenAI LLM agent
│   ├── prompt_templates.py   ← System + action prompts
│   └── run_baseline.py       ← Reproducible scoring script
├── tests/                    ← pytest test suite
└── dashboard/
    └── index.html            ← Live monitoring dashboard
```

## 12. Hackathon Compliance

- [x] OpenEnv YAML with correct schema
- [x] POST /reset, POST /step, GET /state, GET /tasks, GET /health endpoints
- [x] Pydantic v2 typed observation, action, reward models
- [x] 3 tasks with increasing difficulty
- [x] Deterministic graders returning 0.0–1.0
- [x] Seeded reproducibility across episodes
- [x] LLM baseline agent with reproducible scores
- [x] Docker build and run with zero errors
- [x] Self-contained: no external databases or APIs for env
- [x] Novel domain with real-world relevance
- [x] Comprehensive test suite
- [x] Interactive monitoring dashboard

---

*Built for the Meta × Hugging Face OpenEnv Hackathon*
