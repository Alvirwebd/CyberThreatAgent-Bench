"""
ctab.agent
==========
Multi-Regime Autonomous Cybersecurity Agents: Tool Dispatcher, Plan-and-Execute, and ReAct.
"""

import json
import re

class AgentToolDispatcher:
    """Dispatches agent tool calls safely against the local environment."""
    def __init__(self, mappers=None, attr=None, playbook=None):
        self.mappers = mappers or {}
        self.attr = attr
        self.playbook = playbook or {}

    def execute(self, tool_name, tool_input, incident_alerts):
        if tool_name == "extract_iocs":
            text = tool_input.get("text", "")
            ips = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
            domains = re.findall(r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b", text)
            return {"ips": ips, "domains": domains}

        elif tool_name == "map_attack_technique":
            text = tool_input.get("text", "")
            mapper = self.mappers.get("hybrid")
            if mapper:
                ranked = mapper.ranked(text, top_k=1, allow_abstain=True)
                if ranked and ranked[0].get("attack_id"):
                    return {"attack_id": ranked[0]["attack_id"], "score": round(float(ranked[0].get("score", 0.0)), 4)}
            return {"attack_id": None, "score": 0.0}

        elif tool_name == "score_risk":
            from .risk import compute_risk_score
            return compute_risk_score(tool_input)

        return {"status": "ok"}

class PlanAndExecuteAgent:
    """Two-phase agent paradigm: structured planning followed by deterministic tool execution."""
    def __init__(self, planner_llm, executor, synthesizer_llm):
        self.planner_llm = planner_llm
        self.executor = executor
        self.synthesizer_llm = synthesizer_llm

    def run(self, incident_alerts):
        plan = self.planner_llm.generate_plan(incident_alerts)
        observations = []
        for step in plan:
            obs = self.executor.execute(step["tool"], step.get("tool_input", {}), incident_alerts)
            observations.append(obs)
        verdict = self.synthesizer_llm.synthesize(incident_alerts, observations)
        return verdict

class AutonomousReActAgent:
    """Multi-turn iterative reasoning and tool execution loop."""
    def __init__(self, llm, executor, max_steps=6):
        self.llm = llm
        self.executor = executor
        self.max_steps = max_steps

    def run(self, incident_alerts):
        history = []
        for step in range(self.max_steps):
            action, action_input = self.llm.decide_action(incident_alerts, history)
            if action == "finish_triage":
                return action_input
            obs = self.executor.execute(action, action_input, incident_alerts)
            history.append({"step": step, "action": action, "obs": obs})
        return {"is_malicious": False, "identified_techniques": []}
