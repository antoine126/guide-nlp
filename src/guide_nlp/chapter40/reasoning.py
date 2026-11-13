"""Navigating tables of contents by reasoning, without vectors (chapter 40).

Usage: uv run --group llm python -m guide_nlp.chapter40.reasoning [N]
(needs data/processed/rapports.jsonl and qa_nav.jsonl; the first run summarises
the 159 sections with the LLM, a few minutes, then reads them from the cache)
"""

import sys
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from guide_nlp.chapter40.tree import Node, build_tree
from guide_nlp.corpus.load import load_report_questions, load_reports
from guide_nlp.corpus.schema import Report
from guide_nlp.llm import chat
from guide_nlp.rag import GROUNDED, abstains

LLM = Callable[..., str]


class Pick(BaseModel):
    raisonnement: str
    numeros: list[int]


def choose(question: str, listing: str, llm: LLM) -> Pick:
    prompt = (
        f"Voici un sommaire.\n\n{listing}\n\nQuestion : {question}\n\n"
        "Quelles sections faut-il lire pour répondre ? Explique en une phrase, "
        "puis donne au plus trois numéros."
    )
    return Pick.model_validate_json(llm(prompt, schema=Pick.model_json_schema()))


def llm(prompt: str, schema: dict[str, Any] | None = None, max_tokens: int = 120) -> str:
    return chat([{"role": "user", "content": prompt}], fmt=schema, max_tokens=max_tokens).content


class Navigator:
    """One tree per report; descend from the report to a leaf, backtrack if it does not answer."""

    def __init__(self, reports: list[Report]) -> None:
        self.reports = {r.source: r for r in reports}
        self.order = [r.source for r in reports]
        self.nodes: list[Node] = []
        self.roots: dict[str, list[int]] = {}
        for report in reports:
            offset = len(self.nodes)
            sections = [s.model_dump() for s in report.sections]
            tree = build_tree(sections, lambda s, r=report: self.summarize(r, s))
            for node in tree:
                node["id"] += offset
                node["children"] = [c + offset for c in node["children"]]
                node["parent"] = None if node["parent"] is None else node["parent"] + offset
                node["source"] = report.source
            self.nodes += tree
            self.roots[report.source] = [n["id"] for n in tree if n["parent"] is None]

    def body(self, node: Node, limit: int = 2500) -> str:
        text = self.reports[node["source"]].text[node["start"] : node["end"]]
        return " ".join(text.split()[:limit])

    @staticmethod
    def summarize(report: Report, section: dict) -> str:
        text = " ".join(report.text[section["start"] : section["end"]].split()[:1500])
        if len(text.split()) < 30:
            return text
        return llm(
            "Résume en une ou deux phrases de quoi traite cette section, en citant ses "
            f"sujets précis. Section « {section['title']} » :\n\n{text}",
            max_tokens=70,
        )

    def choose(self, question: str, ids: list[int], where: str) -> list[int]:
        listing = "\n".join(
            f"{i}. {self.nodes[i]['title']} (p. {self.nodes[i]['page']}) : "
            f"{' '.join(self.nodes[i]['summary'].split()[:25])}"
            for i in ids
        )
        reply = llm(
            f"Tu cherches dans {where}.\n\n{listing}\n\nQuestion : {question}\n\n"
            "Quelle entrée faut-il ouvrir ? Explique en une phrase, puis donne un ou "
            "deux numéros.",
            schema=Pick.model_json_schema(),
        )
        try:
            return [i for i in Pick.model_validate_json(reply).numeros if i in ids][:2]
        except ValidationError:
            return []

    def descend(self, question: str, budget: int = 3) -> tuple[str, list[str]]:
        titles = "\n".join(f"{i}. {self.reports[s].title}" for i, s in enumerate(self.order))
        reply = llm(
            f"Rapports disponibles :\n{titles}\n\nQuestion : {question}\n\nQuel rapport "
            "faut-il ouvrir ? Explique en une phrase, puis donne un ou deux numéros.",
            schema=Pick.model_json_schema(),
        )
        try:
            chosen = [
                self.order[i]
                for i in Pick.model_validate_json(reply).numeros
                if 0 <= i < len(self.order)
            ][:2]
        except ValidationError:
            chosen = []
        frontier: list[int] = []
        for source in chosen or self.order:
            where = f"le sommaire de « {self.reports[source].title} »"
            frontier += self.choose(question, self.roots[source], where)
        path, reads, answer = [], 0, "Je ne sais pas."
        while frontier and reads < budget:
            node = self.nodes[frontier.pop(0)]
            path.append(node["title"])
            if node["children"]:
                where = f"la partie « {node['title']} »"
                frontier = self.choose(question, node["children"], where) + frontier
                continue
            reads += 1
            context = f"[1] {node['title']} : {self.body(node)}"
            answer = chat(
                [
                    {"role": "system", "content": GROUNDED},
                    {"role": "user", "content": f"Extraits :\n{context}\n\nQuestion : {question}"},
                ],
                max_tokens=200,
            ).content
            if not abstains(answer):
                break
        return answer, path


def main(n: int = 3) -> None:
    navigator = Navigator(load_reports())
    for q in load_report_questions("qa_nav")[:n]:
        answer, path = navigator.descend(q.question)
        right = any(a.lower() in answer.lower() for a in q.answers)
        verdict = "right" if right else "wrong"
        print(f"{q.question}\n  path: {' > '.join(path)}\n  {answer[:120]} ({verdict})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
