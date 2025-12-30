"""
PDF 工具：將 PDF 多頁渲染成圖片（PNG）

設計目標：
- 品質優先：預設 300 DPI
- 多頁：每頁輸出一張圖片
- 輸出到 UPLOAD_DIR，並回傳 (page_index, file_path, filename)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import uuid


@dataclass(frozen=True)
class RenderedPdfPage:
    page_index: int  # 1-based
    file_path: Path
    filename: str


def render_pdf_to_png_pages(
    pdf_path: Path,
    output_dir: Path,
    *,
    dpi: int = 300,
    filename_stem: Optional[str] = None,
) -> List[RenderedPdfPage]:
    """
    將 pdf_path 內所有頁渲染成 PNG，存到 output_dir。
    """
    # 延後 import，避免未安裝時影響非 PDF 流程
    import fitz  # PyMuPDF

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    stem = filename_stem or pdf_path.stem
    # 72 DPI 是 PDF 的基準解析度
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    doc = fitz.open(str(pdf_path))
    pages: List[RenderedPdfPage] = []
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            page_index = i + 1
            # 避免同名衝突：加 uuid
            unique = uuid.uuid4().hex[:8]
            filename = f"{stem}_page_{page_index:03d}_{unique}.png"
            out_path = output_dir / filename
            pix.save(str(out_path))
            pages.append(RenderedPdfPage(page_index=page_index, file_path=out_path, filename=filename))
    finally:
        doc.close()

    return pages


