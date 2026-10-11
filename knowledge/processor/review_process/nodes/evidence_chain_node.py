"""节点 5：证据链构建。

把「图谱多跳路径 + 命中文档片段 + 图谱三元组」统一组织为结构化证据链，
并归一化出 relations（供合规规则计算金额/比例等）。证据链是审查报告的核心依据。
"""

import re
from typing import Any, Dict, List

from knowledge.processor.review_process.base import BaseNode
from knowledge.processor.review_process.state import ReviewGraphState

_TRIPLE_RE = re.compile(r"\[(?P<head>.+?)\]\s*-\((?P<rel>.+?)\)->\s*(?P<tail>.+)")


class EvidenceChainNode(BaseNode):
    name = "evidence_chain_node"

    def process(self, state: ReviewGraphState) -> ReviewGraphState:
        chain: List[Dict[str, Any]] = []

        # 1. 图谱多跳路径（最强证据：如 品牌→授权店铺→商家）
        for path in (state.get("kg_paths") or []):
            text = (path.get("path_text") or "").strip()
            if not text:
                continue
            chain.append({
                "type": "graph_path",
                "content": text,
                "nodes": path.get("nodes") or [],
                "hops": path.get("hops"),
                "source": "Neo4j",
            })

        # 2. 命中文档片段（带来源/编号/分数）
        for doc in (state.get("reranked_docs") or []):
            content = (doc.get("content") or "").strip()
            if not content:
                continue
            chain.append({
                "type": "document",
                "content": content,
                "chunk_id": doc.get("chunk_id"),
                "source": doc.get("source") or doc.get("title") or doc.get("file_title") or "",
                "doc_type": doc.get("doc_type", ""),
                "publish_date": doc.get("publish_date", ""),
                "score": doc.get("score"),
            })

        # 3. 图谱三元组（一跳关系）
        for triple in (state.get("kg_triples") or [])[:20]:
            chain.append({"type": "relation", "content": str(triple), "source": "Neo4j"})

        # 4. 归一化关系（供规则引擎使用）
        relations = self._build_relations(state)

        # 5. 兜底：若图谱/文档均无证据（如离线或上游直接注入关系），
        #    则把归一化关系本身作为「关系型证据」，避免证据链为空。
        if not chain:
            for rel in relations[:20]:
                props = rel.get("props") or {}
                prop_text = "，".join(f"{k}={v}" for k, v in props.items()) if props else ""
                text = f"{rel['head']} -[{rel['rel']}]-> {rel['tail']}"
                if prop_text:
                    text += f"（{prop_text}）"
                chain.append({"type": "relation", "content": text, "source": "input/Neo4j"})

        self.logger.info(f"证据链构建完成：{len(chain)} 条证据 / {len(relations)} 条关系")
        return {"evidence_chain": chain, "relations": relations}

    def _build_relations(self, state: ReviewGraphState) -> List[Dict[str, Any]]:
        """合并「外部注入关系 + 路径关系 + 三元组关系」，按 (head,rel,tail) 去重。"""
        result: List[Dict[str, Any]] = []
        seen = set()

        def _add(head: str, rel: str, tail: str, props: Dict[str, Any] = None):
            head, rel, tail = (head or "").strip(), (rel or "").strip(), (tail or "").strip()
            if not (head and rel and tail):
                return
            key = (head, rel, tail)
            if key in seen:
                return
            seen.add(key)
            result.append({"head": head, "rel": rel, "tail": tail, "props": props or {}})

        # a) 外部注入（便于离线测试/上游装配）
        for rel in (state.get("relations") or []):
            if isinstance(rel, dict):
                _add(rel.get("head"), rel.get("rel") or rel.get("type"), rel.get("tail"), rel.get("props"))

        # b) 图谱路径 → 逐段关系（带路径属性）
        for path in (state.get("kg_paths") or []):
            nodes = path.get("nodes") or []
            rels = path.get("rels") or []
            props_list = path.get("props") or []
            for i, rel in enumerate(rels):
                if i + 1 >= len(nodes):
                    break
                props = props_list[i] if i < len(props_list) and isinstance(props_list[i], dict) else {}
                _add(nodes[i], rel, nodes[i + 1], props)

        # c) 三元组文本 → 关系（无属性）
        for triple in (state.get("kg_triples") or []):
            match = _TRIPLE_RE.match(str(triple))
            if match:
                _add(match.group("head"), match.group("rel"), match.group("tail"))

        return result
