"""SmolVLA 用の PARC2026 Track 1 ポリシーサーバー。"""

import argparse
from abc import ABC, abstractmethod

import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response

from smolvla_policy import SmolVLAPolicyAdapter


class BasePolicy(ABC):
    @abstractmethod
    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        ...

    @abstractmethod
    def reset(self, instruction: str = "") -> None:
        ...


class MyPolicy(BasePolicy):
    """学習済み SmolVLA を提出 API へ接続する。"""

    def __init__(self) -> None:
        self.adapter = SmolVLAPolicyAdapter()

    def get_action(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return self.adapter.get_action(obs)

    def reset(self, instruction: str = "") -> None:
        self.adapter.reset(instruction)


def deserialize_obs(data: bytes) -> dict[str, np.ndarray]:
    unpacked = msgpack.unpackb(data, raw=False)
    obs = {}
    for key, val in unpacked.items():
        arr = np.frombuffer(val["data"], dtype=np.dtype(val["dtype"]))
        obs[key] = arr.reshape(val["shape"]).copy()
    return obs


def serialize_action(action: np.ndarray) -> bytes:
    return msgpack.packb(
        {"data": action.astype(np.float32).tobytes()},
        use_bin_type=True,
    )


app = FastAPI(title="VLA Policy Server")
_policy: BasePolicy | None = None


def set_policy(policy: BasePolicy) -> None:
    global _policy
    _policy = policy


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset_policy(request: Request):
    body = await request.body()
    instruction = ""
    if body:
        import json

        data = json.loads(body)
        instruction = data.get("instruction", "")
    _policy.reset(instruction=instruction)
    return {"status": "ok"}


@app.post("/act")
async def act(request: Request):
    body = await request.body()
    obs = deserialize_obs(body)
    action = _policy.get_action(obs)
    return Response(
        content=serialize_action(action),
        media_type="application/x-msgpack",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    set_policy(MyPolicy())
    print(f"Policy server starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
