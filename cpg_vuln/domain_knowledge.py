
from __future__ import annotations

from typing import Dict, Iterable, List, Set

MEMORY_SINKS: Set[str] = {
    "strcpy",
    "strncpy",
    "strcat",
    "strncat",
    "sprintf",
    "vsprintf",
    "snprintf",
    "gets",
    "scanf",
    "sscanf",
    "fscanf",
    "memcpy",
    "memmove",
    "memset",
    "bcopy",
    "wcscpy",
    "wcsncpy",
}

ALLOC_APIS: Set[str] = {
    "malloc",
    "calloc",
    "realloc",
    "free",
    "strdup",
    "strndup",
    "new",
    "delete",
}

INJECTION_SINKS: Set[str] = {
    "system",
    "popen",
    "execl",
    "execlp",
    "execle",
    "execv",
    "execvp",
    "execve",
    "CreateProcess",
    "ShellExecute",
    "printf",
    "fprintf",
    "vprintf",
    "vfprintf",
}

SOURCES: Set[str] = {
    "recv",
    "recvfrom",
    "read",
    "fread",
    "fgets",
    "getline",
    "getenv",
    "gets",
    "scanf",
    "argv",
}

ALL_SECURITY_APIS: Set[str] = MEMORY_SINKS | ALLOC_APIS | INJECTION_SINKS | SOURCES

API_WHITELIST: Set[str] = set(ALL_SECURITY_APIS)


def classify_api(name: str) -> Dict[str, float]:
    n = (name or "").strip()
    return {
        "is_memory_sink": 1.0 if n in MEMORY_SINKS else 0.0,
        "is_alloc": 1.0 if n in ALLOC_APIS else 0.0,
        "is_injection": 1.0 if n in INJECTION_SINKS else 0.0,
        "is_source": 1.0 if n in SOURCES else 0.0,
        "is_any_sec_api": 1.0 if n in ALL_SECURITY_APIS else 0.0,
    }


def graph_domain_vector(
    node_types: Iterable[str],
    norm_tokens: Iterable[str],
    raw_texts: Iterable[str],
) -> List[float]:
    types = list(node_types)
    tokens = list(norm_tokens)
    texts = list(raw_texts)
    n = max(len(types), 1)

    mem = sum(1 for t in tokens if t in MEMORY_SINKS)
    alloc = sum(1 for t in tokens if t in ALLOC_APIS)
    inj = sum(1 for t in tokens if t in INJECTION_SINKS)
    src = sum(1 for t in tokens if t in SOURCES)
    ptr = sum(1 for t in texts if ("->" in t or "*" in t))
    ctrl = sum(
        1
        for t in types
        if t
        in {
            "if_statement",
            "for_statement",
            "while_statement",
            "switch_statement",
            "do_statement",
        }
    )
    calls = sum(1 for t in types if "call" in t)
    rets = sum(1 for t in types if t == "return_statement")
    arrays = sum(1 for t in types if "subscript" in t or "array" in t)
    assigns = sum(1 for t in types if t in {"assignment_expression", "init_declarator"})
    has_goto = 1.0 if any(t == "goto_statement" for t in types) else 0.0
    has_unsafe_mem = 1.0 if mem > 0 else 0.0

    return [
        mem / n,
        alloc / n,
        inj / n,
        src / n,
        ptr / n,
        ctrl / n,
        calls / n,
        rets / n,
        arrays / n,
        assigns / n,
        has_goto,
        has_unsafe_mem,
    ]


DOMAIN_GRAPH_DIM = 12
