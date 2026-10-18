import json


class FakeLLM:
    """Stand-in for finpilot.llm.LLM. handler(system, user) -> dict."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def json(self, system, user):
        self.calls.append((system, user))
        return self.handler(system, user)

    def complete(self, system, user, json_mode=False):
        return json.dumps(self.json(system, user))
