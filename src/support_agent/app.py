import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel

from .config import get_settings
from .evaluation import load_scenarios, run_eval, summarize
from .graph.build import build_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"
EVAL_SCENARIOS_PATH = Path(__file__).resolve().parent.parent.parent / "eval" / "scenarios.jsonl"


class ChatRequest(BaseModel):
    thread_id: str
    message: str


class ResumeRequest(BaseModel):
    thread_id: str
    approved: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    async with AsyncSqliteSaver.from_conn_string(settings.checkpoint_db_path) as checkpointer:
        graph = build_graph(settings, checkpointer=checkpointer)
        app.state.settings = settings
        app.state.graph = graph
        yield


app = FastAPI(title="Reliable LangGraph Support Agent", lifespan=lifespan)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


async def _app_state() -> dict:
    # `app.state` is Starlette's `State` object, which proxies attribute access
    # through an internal `_state` dict — `app.state.__dict__` returns that
    # internal wrapper (`{"_state": {...}}`), not the attributes themselves, so
    # this builds the dict explicitly rather than relying on `__dict__`/`vars()`.
    return {"settings": app.state.settings, "graph": app.state.graph}


def _shape_result(result: dict) -> dict:
    interrupts = result.get("__interrupt__")
    if interrupts:
        return {
            "status": "awaiting_approval",
            "approval": interrupts[0].value,
            "trace": result.get("trace", []),
        }
    return {
        "status": "complete",
        "reply": result.get("reply"),
        "escalated": bool(result.get("escalated", False)),
        "trace": result.get("trace", []),
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "support-agent"}


@app.get("/")
async def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="No static UI bundled.")
    return FileResponse(index_path)


@app.get("/graph/diagram")
async def graph_diagram(state: dict = Depends(_app_state)) -> dict:
    graph = state["graph"]
    return {"mermaid": graph.get_graph().draw_mermaid()}


@app.post("/chat")
async def chat(request: ChatRequest, state: dict = Depends(_app_state)) -> dict:
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="'message' must not be empty.")
    graph = state["graph"]
    config = {"configurable": {"thread_id": request.thread_id}}
    result = await graph.ainvoke({"messages": [{"role": "user", "content": request.message}]}, config)
    return _shape_result(result)


@app.get("/eval/run")
async def eval_run(state: dict = Depends(_app_state)) -> dict:
    """Runs the labeled scenario set (eval/scenarios.jsonl) against a fresh,
    isolated graph built with the same settings — and therefore the same
    classifier choice, rule-based or LLM — as the live app, but with its own
    MemorySaver and mock order data so it never touches real conversation
    state. Not the live `state["graph"]`, deliberately: an eval run should be
    reproducible regardless of what a human tester has already done in the
    demo UI."""

    if not EVAL_SCENARIOS_PATH.exists():
        raise HTTPException(status_code=404, detail="No eval scenarios bundled with this deployment.")
    scenarios = load_scenarios(EVAL_SCENARIOS_PATH)
    results = await run_eval(scenarios, settings=state["settings"])
    return summarize(results)


@app.post("/chat/resume")
async def chat_resume(request: ResumeRequest, state: dict = Depends(_app_state)) -> dict:
    graph = state["graph"]
    config = {"configurable": {"thread_id": request.thread_id}}
    snapshot = await graph.aget_state(config)
    if not snapshot.next:
        raise HTTPException(status_code=409, detail="This conversation has no pending approval to resume.")
    result = await graph.ainvoke(Command(resume={"approved": request.approved}), config)
    return _shape_result(result)
