"""A small chat UI for the agent (Gradio). Run with `cricket-ai agent ui`."""

from __future__ import annotations


def launch(agent, port: int = 7860, share: bool = False) -> None:
    import gradio as gr

    def respond(message, history):
        result = agent.ask(message)
        tools = ", ".join(result.tools_used) or "no tools"
        return f"{result.answer}\n\n*Tools: {tools} · {result.seconds}s*"

    gr.ChatInterface(
        respond,
        title="Cricket AI agent",
        description="Ask about IPL matches, players, chases or synthetic overs.",
        examples=[
            "How many runs did V Kohli score in the 2016 season?",
            "Who scored the most runs in the 2023 season?",
            "A team chasing 180 is 100/2 after 12 overs. What is their chance of winning?",
        ],
    ).launch(server_port=port, share=share)
