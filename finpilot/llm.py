import json
import os
import time

from openai import OpenAI

from . import config


class LLMUnavailable(Exception):
    pass


class LLM:
    """All LLM traffic in FinPilot goes through this class, pointed at Mesh API."""

    def __init__(self, client=None, model=None):
        self.model = model or config.MODEL
        self._client = client

    @property
    def client(self):
        if self._client is None:
            key = os.getenv("MESH_API_KEY")
            if not key:
                raise LLMUnavailable("MESH_API_KEY not set")
            self._client = OpenAI(base_url=config.MESH_BASE_URL, api_key=key)
        return self._client

    def complete(self, system, user, json_mode=False):
        last = None
        for _ in range(2):
            try:
                kw = {"response_format": {"type": "json_object"}} if json_mode else {}
                r = self.client.chat.completions.create(
                    model=self.model, temperature=0,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}], **kw)
                return r.choices[0].message.content or ""
            except LLMUnavailable:
                raise
            except Exception as e:  # network, rate limit, bad model
                last = e
                time.sleep(1)
        raise LLMUnavailable(str(last))

    def json(self, system, user):
        txt = self.complete(system, user, json_mode=True)
        try:
            return json.loads(txt)
        except json.JSONDecodeError as e:
            raise LLMUnavailable(f"bad json from model: {e}")


_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm


def set_llm(obj):
    global _llm
    _llm = obj
