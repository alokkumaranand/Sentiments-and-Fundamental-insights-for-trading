"""Agents: the LLM's two thin layers around the deterministic core.

Below the core sits the **induction** agent, whose only output is a reviewable vocabulary
proposal that a deterministic gate must accept term by term. Above it sit the **explorer**
(which picks follow-up tools from the registry) and the **presenter** (which writes prose that
may only cite numbers a tool already computed). `null_agent.py` is the no-key path: the same
stage, templated, so a run without an endpoint is still complete and honest.
"""
