import json
import os
import time

from openai import OpenAI

from . import config


TIMEOUT = 20  # seconds per request
ATTEMPTS = 2  # one try plus one retry


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
            # our own loop below does the one retry; the SDK must not add retries or a 600 s timeout
            self._client = OpenAI(base_url=config.MESH_BASE_URL, api_key=key, timeout=TIMEOUT, max_retries=0)
        return self._client

    def complete(self, system, user, json_mode=False):
        last = None
        for attempt in range(ATTEMPTS):
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
                if attempt < ATTEMPTS - 1:  # no sleep after the final failed attempt
                    time.sleep(1)
        raise LLMUnavailable(str(last))

    def json(self, system, user):
        txt = self.complete(system, user, json_mode=True)
        try:
            val = json.loads(txt)
        except json.JSONDecodeError as e:
            raise LLMUnavailable(f"bad json from model: {e}")
        if not isinstance(val, dict):
            raise LLMUnavailable("expected a JSON object")
        return val


_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm


def set_llm(obj):
    global _llm
    _llm = obj
