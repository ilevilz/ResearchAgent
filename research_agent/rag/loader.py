"""文档加载与解析：PDF（pypdf 优先，pypdfium2 兜底）与纯文本。

解析器选择理由：
- ``pypdf`` 纯 Python、无系统依赖，作为默认；
- 部分 PDF（扫描件之外的特殊编码）pypdf 会抛错或抽出空文本，此时自动切换
  ``pypdfium2`` 重试，显著提升解析成功率。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class Page:
    """PDF 单页。"""

    page_number: int
    text: str


@dataclass
class ParsedDocument:
    """解析后的文档：统一 PDF / 文本的表示。"""

    source: str
    title: str = ""
    pages: list[Page] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    parser: str = ""

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def num_pages(self) -> int:
        return len(self.pages)

    @property
    def num_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "title": self.title,
            "pages": self.num_pages,
            "chars": self.num_chars,
            "parser": self.parser,
        }


class PDFParseError(RuntimeError):
    """所有解析器均失败。"""


def _parse_with_pypdf(path: Path) -> list[Page]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[Page] = []
    for i, page in enumerate(reader.pages, start=1):
        pages.append(Page(page_number=i, text=(page.extract_text() or "").strip()))
    return pages


def _parse_with_pypdfium2(path: Path) -> list[Page]:
    import pypdfium2 as pdfium  # type: ignore

    pdf = pdfium.PdfDocument(str(path))
    try:
        pages: list[Page] = []
        for i in range(len(pdf)):
            text = pdf[i].get_textpage().get_text_range() or ""
            pages.append(Page(page_number=i + 1, text=text.strip()))
        return pages
    finally:
        pdf.close()


def _meaningful(pages: list[Page]) -> bool:
    """解析结果是否有效（防止抽出全空白页被判为成功）。"""
    return sum(len(p.text) for p in pages) >= 50


def parse_pdf(path: str | Path, title: str | None = None) -> ParsedDocument:
    """解析单个 PDF，自动选择可用解析器。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 不存在：{path}")

    errors: list[str] = []
    for name, parser in (("pypdf", _parse_with_pypdf), ("pypdfium2", _parse_with_pypdfium2)):
        try:
            pages = parser(path)
        except Exception as exc:  # noqa: BLE001 - 尝试下一个解析器
            errors.append(f"{name}: {exc}")
            continue
        if _meaningful(pages):
            logger.info("PDF 解析成功 [%s]：%s（%d 页）", name, path.name, len(pages))
            return ParsedDocument(
                source=str(path),
                title=title or path.stem,
                pages=pages,
                metadata={"filename": path.name},
                parser=name,
            )
        errors.append(f"{name}: 抽取文本过少")

    raise PDFParseError(f"解析失败 {path.name} -> {'; '.join(errors)}")


def parse_text(path: str | Path, title: str | None = None) -> ParsedDocument:
    """解析纯文本 / Markdown。"""
    path = Path(path)
    content = path.read_text(encoding="utf-8", errors="ignore")
    return ParsedDocument(
        source=str(path),
        title=title or path.stem,
        pages=[Page(page_number=1, text=content)],
        metadata={"filename": path.name},
        parser="text",
    )


def parse_document(path: str | Path, title: str | None = None) -> ParsedDocument:
    """按扩展名分派解析器。"""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return parse_pdf(path, title)
    return parse_text(path, title)


def load_documents(directory: str | Path, pattern: str = "*") -> list[ParsedDocument]:
    """批量加载目录下的文档，单个失败不影响其余。"""
    directory = Path(directory)
    docs: list[ParsedDocument] = []
    for path in sorted(directory.glob(pattern)):
        if path.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        try:
            docs.append(parse_document(path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("跳过 %s：%s", path.name, exc)
    return docs
