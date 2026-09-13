from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from tree_sitter import Parser
from tree_sitter_languages import get_language


class CPGExtractor:

    def __init__(self) -> None:
        self._parser = Parser()
        self._parser.set_language(get_language("cpp"))

    def build_cpg_from_code(self, code: str, lang: str = "c") -> Dict[str, Any]:
        tree = self._parser.parse(code.encode("utf-8"))
        root = tree.root_node

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        parent_of: Dict[int, int | None] = {}
        children_of: Dict[int, List[int]] = {}
        preorder_ids: List[int] = []

        stack: List[tuple[Any, int | None, int, int]] = [(root, None, 0, 0)]
        next_id = 0

        while stack:
            node, parent_id, depth, child_pos = stack.pop()

            current_id = next_id
            next_id += 1
            parent_of[current_id] = parent_id
            children_of[current_id] = []
            preorder_ids.append(current_id)

            try:
                text = code[node.start_byte : node.end_byte]
            except Exception:
                text = ""

            nodes.append(
                {
                    "id": current_id,
                    "type": node.type,
                    "code": text,
                    "depth": depth,
                    "child_pos": child_pos,
                }
            )

            if parent_id is not None:
                children_of[parent_id].append(current_id)

            for pos, child in reversed(list(enumerate(node.children))):
                stack.append((child, current_id, depth + 1, pos))

        for child_id, parent_id in parent_of.items():
            if parent_id is None:
                continue
            edges.append({"src": parent_id, "dst": child_id, "kind": "AST"})
            edges.append({"src": child_id, "dst": parent_id, "kind": "AST_REV"})

        block_like = {
            "compound_statement",
            "translation_unit",
            "function_definition",
            "declaration_list",
        }
        for nid, n in enumerate(nodes):
            if n.get("type") not in block_like:
                continue
            ch = children_of.get(nid, [])
            for a, b in zip(ch, ch[1:]):
                edges.append({"src": a, "dst": b, "kind": "SEQ"})
                edges.append({"src": b, "dst": a, "kind": "SEQ_REV"})

        for nid, n in enumerate(nodes):
            nt = str(n.get("type", ""))
            if nt != "call_expression":
                continue
            ch = children_of.get(nid, [])
            for cid in ch:
                ct = str(nodes[cid].get("type", ""))
                if ct == "identifier":
                    edges.append({"src": nid, "dst": cid, "kind": "CALL_CALLEE"})
                    edges.append({"src": cid, "dst": nid, "kind": "CALL_CALLEE_REV"})
                else:
                    edges.append({"src": nid, "dst": cid, "kind": "CALL_ARG"})
                    edges.append({"src": cid, "dst": nid, "kind": "CALL_ARG_REV"})

        control_nodes = {
            "if_statement",
            "for_statement",
            "while_statement",
            "switch_statement",
            "do_statement",
        }

        def _descendants(start_id: int) -> List[int]:
            out: List[int] = []
            st = [start_id]
            while st:
                cur = st.pop()
                out.append(cur)
                st.extend(children_of.get(cur, []))
            return out

        for nid, n in enumerate(nodes):
            if str(n.get("type", "")) not in control_nodes:
                continue
            ch = children_of.get(nid, [])
            if len(ch) <= 1:
                continue
            body_roots = ch[1:]
            for root_id in body_roots:
                for did in _descendants(root_id):
                    if did == nid:
                        continue
                    edges.append({"src": nid, "dst": did, "kind": "CTRL_DEP"})
                    edges.append({"src": did, "dst": nid, "kind": "CTRL_DEP_REV"})


        id_type = {n["id"]: str(n.get("type", "")) for n in nodes}
        id_code = {n["id"]: str(n.get("code", "")) for n in nodes}

        def _is_identifier_id(i: int) -> bool:
            return id_type.get(i) == "identifier"

        def _find_first_identifier(start_id: int) -> int | None:
            stack2 = [start_id]
            seen = set()
            while stack2:
                cur = stack2.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                if _is_identifier_id(cur):
                    return cur
                stack2.extend(reversed(children_of.get(cur, [])))
            return None

        last_def: Dict[str, int] = {}

        for nid in preorder_ids:
            t = id_type.get(nid, "")
            if t in {"assignment_expression", "init_declarator"}:
                def_id = _find_first_identifier(nid)
                if def_id is not None:
                    name = id_code.get(def_id, "").strip()
                    if name:
                        last_def[name] = def_id

            if _is_identifier_id(nid):
                name = id_code.get(nid, "").strip()
                if not name:
                    continue
                if name in last_def and last_def[name] != nid:
                    edges.append({"src": last_def[name], "dst": nid, "kind": "DFG"})
                    edges.append({"src": nid, "dst": last_def[name], "kind": "DFG_REV"})

        for nid, n in enumerate(nodes):
            if str(n.get("type", "")) != "return_statement":
                continue
            p = parent_of.get(nid)
            while p is not None:
                if str(nodes[p].get("type", "")) == "function_definition":
                    edges.append({"src": nid, "dst": p, "kind": "RET_FLOW"})
                    edges.append({"src": p, "dst": nid, "kind": "RET_FLOW_REV"})
                    break
                p = parent_of.get(p)

        return {"nodes": nodes, "edges": edges}

