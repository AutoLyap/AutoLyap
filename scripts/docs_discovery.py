# SPDX-FileCopyrightText: 2026 AutoLyap contributors
# SPDX-License-Identifier: GPL-3.0-only

"""Generate agent-readable documentation from the rendered Sphinx articles."""

from __future__ import annotations

import html
import re
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter


_TEX_COMMAND = re.compile(r"\\([A-Za-z]+)")
_API_PAGES = {
    "api_reference",
    "algorithms",
    "base_algorithms",
    "concrete_algorithms",
    "function_classes",
    "operator_classes",
    "problem_class",
    "lyapunov_analyses",
    "iteration_dependent_analysis",
    "iteration_independent_analysis",
    "solver_backends",
}
_CATEGORIES = (
    "Quick start and guides",
    "Theory",
    "Worked examples",
    "API reference",
    "Project and development",
)


def _plain_text(value: str) -> str:
    return " ".join(value.split())


def _markdown_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("[", r"\[").replace("]", r"\]")


def _anchor(identifier: str) -> str:
    return f'<a id="{html.escape(identifier, quote=True)}"></a>'


def _fenced(text: str, language: str = "") -> str:
    """Keep literal code, including embedded Markdown fences and indentation."""
    longest_run = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest_run + 1)
    newline = "" if text.endswith("\n") else "\n"
    return f"\n\n{fence}{language}\n{text}{newline}{fence}\n\n"


class _ArticleConverter(MarkdownConverter):
    """Preserve Sphinx's semantic content while removing theme-only markup."""

    def __init__(self, **options):
        super().__init__(**options)
        self.literal_content = {}

    def _protect_literal(self, content: str, *, block: bool) -> str:
        token = f"\x00AUTOLYAP{len(self.literal_content)}\x00"
        self.literal_content[token] = (content, block)
        return f"\n\n{token}\n\n" if block else token

    def convert_soup(self, soup):
        result = super().convert_soup(soup)
        for token, (content, block) in self.literal_content.items():
            if block:
                # Lists and quotes may prefix placeholder lines; those prefixes
                # must never become part of the original code or TeX source.
                result = re.sub(
                    rf"(?m)^[ \t>]*{re.escape(token)}[ \t]*$",
                    lambda _match, value=content: value,
                    result,
                )
            result = result.replace(token, content)
        return result

    def process_tag(self, node, parent_tags=None):
        parent_tags = parent_tags or set()
        classes = node.get("class", [])
        if "math" in classes:
            # Sphinx keeps the original delimited TeX in the pre-MathJax HTML.
            result = self._protect_literal(
                node.get_text(), block=node.name not in {"span", "a"}
            )
            for descendant in node.find_all(id=True):
                result = _anchor(str(descendant["id"])) + "\n" + result
        elif node.name == "dt" and "sig" in classes:
            # Sphinx's accessible signature layout may contain nested dl/dd tags.
            # Flatten that layout without escaping or dropping the signature.
            signature = _plain_text(node.get_text())
            result = self._protect_literal(_fenced(signature, "text"), block=True)
            references = []
            seen = set()
            for reference in node.find_all("a", href=True):
                label = _plain_text(reference.get_text())
                target = reference["href"]
                if label and (label, target) not in seen:
                    seen.add((label, target))
                    references.append(f"[{_markdown_label(label)}]({target})")
            if references:
                result += ", ".join(references) + "\n\n"
            for descendant in node.find_all(id=True):
                result = _anchor(str(descendant["id"])) + "\n" + result
        else:
            result = super().process_tag(node, parent_tags=parent_tags)

        identifier = node.get("id")
        if identifier:
            anchor = _anchor(str(identifier))
            if node.name in {"a", "span", "code", "em", "strong"}:
                result = anchor + result
            else:
                result = f"\n\n{anchor}\n\n{result}"
        return result

    def convert_pre(self, el, text, parent_tags):
        language = ""
        for ancestor in (el, *el.parents):
            for class_name in ancestor.get("class", []):
                match = re.fullmatch(r"(?:highlight|language)-([\w+-]+)", class_name)
                if match:
                    language = match.group(1)
                    break
            if language:
                break
        return self._protect_literal(_fenced(el.get_text(), language), block=True)

    def convert_dd(self, el, text, parent_tags):
        # Autodoc nests the entire API description inside definition lists.
        # Markdownify's definition-list indentation rewrites every code/math line.
        return f"\n\n{text.strip()}\n\n" if text.strip() else ""


def _used_macro_definitions(math_text: str, macros: dict) -> list[str]:
    required = set(_TEX_COMMAND.findall(math_text)) & macros.keys()
    pending = list(required)
    while pending:
        name = pending.pop()
        value = macros[name]
        definition = value[0] if isinstance(value, (list, tuple)) else value
        dependencies = set(_TEX_COMMAND.findall(str(definition))) & macros.keys()
        for dependency in dependencies - required:
            required.add(dependency)
            pending.append(dependency)

    definitions = []
    for name in sorted(required):
        value = macros[name]
        if isinstance(value, (list, tuple)):
            definition = str(value[0])
            arguments = int(value[1]) if len(value) > 1 else 0
            argument_spec = f"[{arguments}]" if arguments else ""
            if len(value) > 2:
                argument_spec += f"[{value[2]}]"
        else:
            definition = str(value)
            argument_spec = ""
        definitions.append(f"\\newcommand{{\\{name}}}{argument_spec}{{{definition}}}")
    return definitions


def _convert_article(article, *, canonical: str, macros: dict) -> str:
    for element in list(article.select("script, nav, .headerlink, .viewcode-link")):
        if element.name is not None:
            if (
                "viewcode-link" in element.get("class", [])
                and element.parent.name == "a"
            ):
                element.parent.decompose()
            else:
                element.decompose()
    for element in article.select("dt.sig .autolyap-sr-only"):
        # An accessibility label added to the expanded signature is navigation UI.
        element.decompose()

    math_text = "\n".join(element.get_text() for element in article.select(".math"))
    for element in article.find_all(href=True):
        element["href"] = urljoin(canonical, str(element["href"]))
    for image in article.find_all("img", src=True):
        image["src"] = urljoin(canonical, str(image["src"]))

    converter = _ArticleConverter(heading_style="ATX", bullets="-", wrap=False)
    result = converter.convert_soup(article).strip()
    definitions = _used_macro_definitions(math_text, macros)
    if definitions:
        result += "\n\nTeX macro definitions used in this page:" + _fenced(
            "\n".join(definitions), "latex"
        )
    return result.rstrip() + "\n"


def _category(route: str) -> str:
    if not route or route == "quick_start":
        return "Quick start and guides"
    if route == "theory" or route.startswith("theory/"):
        return "Theory"
    if route == "examples" or route.startswith("examples/"):
        return "Worked examples"
    if route in _API_PAGES:
        return "API reference"
    return "Project and development"


def write_discovery_exports(
    outdir: str | Path,
    *,
    base_url: str,
    project: str,
    release: str,
    description: str,
    macros: dict,
) -> None:
    """Write sibling Markdown pages and a categorized ``llms.txt`` directory.

    Only indexable generated HTML is exported. Canonical URLs remain the HTML
    URLs; all article links resolve against them, including local fragments.
    Content comes from the rendered article, so includes, autodoc and citations
    stay synchronized with the documentation visitors read.
    """
    output_directory = Path(outdir)
    base_url = base_url.rstrip("/")
    base = urlparse(base_url)
    pages = []
    noindex_exports = []
    for html_path in sorted(output_directory.rglob("*.html")):
        soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
        robots = soup.find("meta", attrs={"name": "robots"})
        directives = {
            value.strip().lower()
            for value in str(robots.get("content", "") if robots else "").split(",")
        }
        if "noindex" in directives or "none" in directives:
            noindex_exports.append(html_path.with_suffix(".md"))
            continue

        canonical_tag = soup.find("link", rel="canonical")
        canonical = str(canonical_tag.get("href", "") if canonical_tag else "")
        parsed = urlparse(canonical)
        if (
            not canonical
            or (parsed.scheme, parsed.netloc) != (base.scheme, base.netloc)
            or not parsed.path.startswith(base.path.rstrip("/") + "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                f"{html_path}: missing or invalid documentation canonical URL"
            )
        article = soup.find(attrs={"role": "main"})
        if article is None:
            raise ValueError(f"{html_path}: missing main documentation article")

        heading = article.find("h1")
        title = _plain_text(
            heading.get_text()
            if heading
            else soup.title.get_text()
            if soup.title
            else project
        )
        title = title.replace("", "").strip()
        description_tag = soup.find("meta", attrs={"name": "description"})
        page_description = _plain_text(
            str(description_tag.get("content", "") if description_tag else "")
        )
        markdown_path = html_path.with_suffix(".md")
        relative_path = markdown_path.relative_to(output_directory).as_posix()
        markdown_url = f"{base_url}/{quote(relative_path, safe='/')}"
        route = parsed.path[len(base.path.rstrip("/")) :].strip("/")
        if route.endswith(".html"):
            route = route[:-5]
        if route.endswith("/index"):
            route = route[:-6]
        if route == "index":
            route = ""
        content = (
            f"Canonical HTML: {canonical}\nDocumentation version: {release}\n\n"
            + _convert_article(article, canonical=canonical, macros=macros)
        )
        pages.append(
            (
                markdown_path,
                markdown_url,
                title,
                page_description,
                _category(route),
                content,
            )
        )

    if not pages:
        raise ValueError(f"{output_directory}: no indexable documentation pages found")

    lines = [
        f"# {project}",
        "",
        f"> {_plain_text(description)}",
        "",
        f"Documentation version: {release}",
        "",
        f"- [Canonical HTML documentation]({base_url}/)",
        "- [Source repository](https://github.com/AutoLyap/AutoLyap)",
        "- [Python package](https://pypi.org/project/autolyap/)",
        "",
        "The linked Markdown pages are generated from the canonical HTML articles in the same documentation build. Each page identifies its canonical HTML URL and documentation version.",
    ]
    for category in _CATEGORIES:
        members = [page for page in pages if page[4] == category]
        if not members:
            continue
        lines.extend(["", f"## {category}", ""])
        for _, url, title, page_description, _, _ in members:
            entry = f"- [{_markdown_label(title)}]({url})"
            if page_description:
                entry += f": {page_description}"
            lines.append(entry)
    for markdown_path, _, _, _, _, content in pages:
        markdown_path.write_text(content, encoding="utf-8")
    for markdown_path in noindex_exports:
        if markdown_path.is_file() and markdown_path.read_text(
            encoding="utf-8"
        ).startswith("Canonical HTML: "):
            markdown_path.unlink()
    (output_directory / "llms.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
