
from __future__ import annotations

import time
import traceback

import torch

_t0 = time.time()
STAGE = "startup"


def gpu_mem() -> str:
    if not torch.cuda.is_available():
        return "device=cpu"
    try:
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f"cuda_alloc={alloc:.2f}G reserved={reserved:.2f}G"
    except Exception as exc:
        return f"cuda_mem_err={exc}"


def progress(msg: str, stage: str | None = None) -> None:
    global STAGE
    if stage:
        STAGE = stage
    elapsed = time.time() - _t0
    line = f"[progress t+{elapsed:8.1f}s stage={STAGE}] {msg} | {gpu_mem()}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)


def fallen(where: str, err: BaseException) -> None:
    progress(f"FALLEN at {where}: {type(err).__name__}: {err}", stage=f"FAILED:{where}")
    traceback.print_exc()
