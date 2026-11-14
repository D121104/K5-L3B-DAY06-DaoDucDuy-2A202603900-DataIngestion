import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src import chat


class AnnotatedPdfTests(unittest.TestCase):
    def test_chunk_phrases_include_final_trailing_window(self):
        words = [f"token{i}" for i in range(23)]
        phrases = chat._chunk_phrases(" ".join(words))
        self.assertIn(" ".join(words[-10:]), phrases)

    def test_synthetic_pdf_highlights_multiple_regions_including_trailing_text(self):
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF is unavailable")

        text = (
            "Alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima. "
            "Mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray. "
            "Yankee zulu amber bronze cobalt denim emerald fuchsia garnet hazel indigo jade."
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "synthetic.pdf"
            original = fitz.open()
            page = original.new_page()
            page.insert_textbox(fitz.Rect(40, 40, 550, 750), text, fontsize=11)
            original.save(source)
            original.close()

            with patch.object(chat, "PDF_PATH", source):
                output, matches, images = chat._annotated_pdf([
                    {"page": 1, "text": text + " completely unmatched words"}
                ])

            self.assertIsNotNone(output)
            self.assertGreater(matches[1]["matched"], 1)
            self.assertGreater(matches[1]["searched"], matches[1]["matched"])
            self.assertIn(1, images)
            annotated = fitz.open(stream=output, filetype="pdf")
            annotations = list(annotated[0].annots() or [])
            self.assertGreater(len(annotations), 1)
            self.assertTrue(source.read_bytes())
            annotated.close()

    def test_mock_pdf_search_highlights_distinct_and_trailing_windows(self):
        text = " ".join(f"word{i}" for i in range(23))
        highlights = []

        class Page:
            def search_for(self, phrase, quads=False):
                if phrase.startswith("word0 ") or phrase.startswith("word13 "):
                    index = int(phrase.split()[0][4:])
                    return [(index, 0, index + 1, 1)]
                return []

            def add_highlight_annot(self, rectangle):
                highlights.append(rectangle)

            def get_pixmap(self):
                return SimpleNamespace(tobytes=lambda _format: b"png")

        class Document:
            def __len__(self):
                return 1

            def __getitem__(self, _index):
                return page

            def tobytes(self, **_kwargs):
                return b"pdf"

        page = Page()
        fake_fitz = SimpleNamespace(open=lambda _path: Document())
        with patch.object(chat, "PDF_PATH", Path(__file__)), patch.dict("sys.modules", {"fitz": fake_fitz}):
            output, matches, images = chat._annotated_pdf([{"page": 1, "text": text}])

        self.assertEqual(output, b"pdf")
        self.assertGreaterEqual(len(highlights), 2)
        self.assertIn((13, 0, 14, 1), highlights)
        self.assertEqual(images[1], b"png")
        self.assertGreater(matches[1]["matched"], 1)

    def test_missing_pdf_and_missing_dependency_fallbacks(self):
        with patch.object(chat, "PDF_PATH", Path("/no/such/source.pdf")):
            self.assertEqual(chat._annotated_pdf([]), (None, {}, {}))

        with patch.object(chat, "PDF_PATH", Path(__file__)):
            with patch.dict("sys.modules", {"fitz": None}):
                self.assertEqual(chat._annotated_pdf([]), (None, {}, {}))


if __name__ == "__main__":
    unittest.main()
