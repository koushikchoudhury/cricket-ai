"""The LangGraph agent.

::

    START -> agent --(tool calls)--> tools --> agent --> ... --> END
                \\--(no tool calls)--> END
                 \\--(step limit reached)--> limit --> END

* ``agent`` - the LLM, with the tools bound, decides the next step.
* ``tools`` - runs every requested call and answers each one with a ToolMessage.
  Human in the loop: calls to expensive tools first go to an approver (a yes/no prompt
  on the command line); a declined call is answered "declined" and the model carries on.
  Errors (bad arguments, failing SQL) are returned to the model so it can recover.
* ``limit`` - stops runaway loops after ``max_steps`` tool rounds.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from cricket_ai.agent.tools import EXPENSIVE_TOOLS, CricketContext, build_tools
from cricket_ai.llm import call_stats, describe_timing, message_text

SYSTEM_PROMPT = """You are a cricket analyst with tools over a ball-by-ball database.
- Use a tool for any fact, number or result; never answer those from memory.
- Prefer player_stats for one player's figures, leaderboard for "who scored/took the most",
  search_cricket for what happened in a match, win_probability for a chase situation,
  run_sql only when no other tool fits.
- Player names are stored with initials (e.g. 'V Kohli'); partial names work in player_stats.
- Copy every number exactly as the tool or the question gives it. Do not recompute,
  round differently or restate the match situation in other numbers.
- Batting and bowling are separate: "runs conceded" by a bowler are never runs scored.
- Earlier turns are context: a follow-up such as "how about Kohli?" keeps the same season
  and the same aspect (batting or bowling) unless the user changes them.
- When you have the answer, reply in one to three sentences and say which tool it came from.
"""
HISTORY_MESSAGES = 6  # the last three question/answer pairs


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    steps: int


@dataclass
class AgentResult:
    question: str
    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    steps: int = 0
    seconds: float = 0.0
    stopped_by_limit: bool = False
    llm_calls: list[dict] = field(default_factory=list)

    @property
    def tools_used(self) -> list[str]:
        return [c["name"] for c in self.tool_calls]

    @property
    def timing(self) -> str:
        """Where the time went: model loading, reading, writing, and tools."""
        return describe_timing(self.llm_calls, sum(c["seconds"] for c in self.tool_calls))


class CricketAgent:
    def __init__(
        self,
        llm,
        ctx: CricketContext,
        max_steps: int = 6,
        approver: Callable[[str, dict], bool] | None = None,
    ):
        if llm is None:
            raise ValueError("The agent needs an LLM: use --llm ollama|anthropic|openai")
        self.ctx = ctx
        self.tools = build_tools(ctx)
        if not self.tools:
            raise ValueError("No tools available: build the data tables and the RAG index first")
        self.by_name = {t.name: t for t in self.tools}
        self.llm = llm.bind_tools(self.tools)
        self.max_steps = max_steps
        self.approver = approver or (lambda name, args: True)
        self._llm_calls: list[dict] = []
        self.graph = self._build()

    # ----- nodes -----
    def _agent(self, state: AgentState) -> dict:
        start = time.perf_counter()
        reply = self.llm.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        self._llm_calls.append(call_stats(reply, time.perf_counter() - start))
        return {"messages": [reply]}

    def _tools(self, state: AgentState) -> dict:
        last = state["messages"][-1]
        replies = []
        for call in last.tool_calls:
            name, args = call["name"], call.get("args", {})
            tool = self.by_name.get(name)
            if tool is None:
                content = f"Error: unknown tool {name!r}. Available: {sorted(self.by_name)}"
            elif name in EXPENSIVE_TOOLS and not self.approver(name, args):
                content = "The user declined this tool call. Answer without it."
            else:
                try:
                    content = tool.invoke(args)
                except Exception as exc:  # noqa: BLE001 - e.g. invalid arguments from the model
                    content = f"Error calling {name}: {exc}"
            replies.append(ToolMessage(str(content), tool_call_id=call["id"], name=name))
        return {"messages": replies, "steps": state.get("steps", 0) + 1}

    def _limit(self, state: AgentState) -> dict:
        return {
            "messages": [
                AIMessage("I stopped after reaching the step limit without a final answer.")
            ]
        }

    # ----- routing -----
    def _after_agent(self, state: AgentState) -> str:
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return END
        return "limit" if state.get("steps", 0) >= self.max_steps else "tools"

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("agent", self._agent)
        g.add_node("tools", self._tools)
        g.add_node("limit", self._limit)
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", self._after_agent, ["tools", "limit", END])
        g.add_edge("tools", "agent")
        g.add_edge("limit", END)
        return g.compile()

    # ----- running -----
    def ask(
        self,
        question: str,
        history: list[dict] | None = None,
        on_event: Callable[[dict], None] | None = None,
    ) -> AgentResult:
        """Answer one question.

        ``history`` is earlier turns as ``{"role": "user"|"assistant", "content": str}`` so
        follow-ups keep their context. ``on_event`` receives progress as it happens:
        ``{"type": "tool_call", "name", "args"}``, ``{"type": "tool_result", "name"}`` and
        ``{"type": "writing"}``.
        """
        start = time.perf_counter()
        first_call = len(self.ctx.calls)
        self._llm_calls = []
        notify = on_event or (lambda event: None)
        messages: list[BaseMessage] = [
            (HumanMessage if turn["role"] == "user" else AIMessage)(turn["content"])
            for turn in (history or [])[-HISTORY_MESSAGES:]
        ]
        messages.append(HumanMessage(question))

        final, steps = None, 0
        stream = self.graph.stream(
            {"messages": messages, "steps": 0},
            {"recursion_limit": 4 * self.max_steps + 10},
            stream_mode="updates",
        )
        for update in stream:
            for node, change in update.items():
                final = change["messages"][-1]
                if node == "agent":
                    for call in getattr(final, "tool_calls", None) or []:
                        notify({"type": "tool_call", "name": call["name"], "args": call["args"]})
                elif node == "tools":
                    steps = change.get("steps", steps)
                    for reply in change["messages"]:
                        notify({"type": "tool_result", "name": reply.name})
                    notify({"type": "writing"})
        return AgentResult(
            question=question,
            answer=message_text(final).strip(),
            tool_calls=[asdict(c) for c in self.ctx.calls[first_call:]],
            steps=steps,
            seconds=round(time.perf_counter() - start, 2),
            stopped_by_limit=message_text(final).startswith(
                "I stopped after reaching the step limit"
            ),
            llm_calls=list(self._llm_calls),
        )

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()


def write_trace(result: AgentResult, directory: Path) -> Path:
    """Append one JSON line per question to traces.jsonl."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "traces.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(
            json.dumps({**asdict(result), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}, default=str)
            + "\n"
        )
    return path
