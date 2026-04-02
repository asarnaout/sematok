"""Tests for fine-tuning data preparation."""

import json
import tempfile
from pathlib import Path

from sematok.dictionary import CompressionDictionary
from sematok.compressor import Compressor
from sematok.languages import get_language
from sematok.lexer import set_language

from data.prepare import prepare_data, _compress_file

set_language(get_language("csharp"))


# -- Helpers --

def _make_corpus(tmp_path: Path, files: dict[str, str], repos: dict[str, str] | None = None):
    """Create a mini corpus with .cs files and optional metadata.jsonl."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, content in files.items():
        (corpus / name).write_text(content, encoding="utf-8")
    if repos:
        meta_path = corpus / "metadata.jsonl"
        with open(meta_path, "w", encoding="utf-8") as f:
            for filename, repo in repos.items():
                entry = {"index": 0, "filename": filename, "original_path": "", "original_size": 0, "source": repo, "license": "MIT"}
                f.write(json.dumps(entry) + "\n")
    return corpus


SAMPLE_CS = """\
using System;
using System.Collections.Generic;

namespace MyApp
{
    public sealed class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("Hello");
        }

        public string Name { get; set; }
    }
}
"""

SIMPLE_CS = """\
class Foo
{
    int x = 42;
}
"""


# -- Tests --

def test_prepare_creates_output_files():
    """prepare_data should create train.jsonl, eval.jsonl, and meta.json."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        corpus = _make_corpus(tmp_path, {"a.cs": SAMPLE_CS, "b.cs": SIMPLE_CS})
        output = tmp_path / "out"

        prepare_data(corpus, output, language="csharp", eval_repos=[])

        assert (output / "train.jsonl").exists()
        assert (output / "eval.jsonl").exists()
        assert (output / "meta.json").exists()


def test_jsonl_format():
    """Each line should be valid JSON with a 'text' field."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        corpus = _make_corpus(tmp_path, {"a.cs": SAMPLE_CS, "b.cs": SAMPLE_CS})
        output = tmp_path / "out"

        prepare_data(corpus, output, language="csharp", eval_repos=[])

        with open(output / "train.jsonl", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                assert "text" in obj
                assert isinstance(obj["text"], str)
                assert len(obj["text"]) > 0


def test_compression_mix():
    """With compress_ratio=0.75 and enough files, some lines should have macros and some shouldn't."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # Create 20 files with boilerplate to get a statistical mix
        files = {f"{i:06d}.cs": SAMPLE_CS for i in range(20)}
        corpus = _make_corpus(tmp_path, files)
        output = tmp_path / "out"

        prepare_data(corpus, output, language="csharp", eval_repos=[], compress_ratio=0.75, seed=42)

        has_macros = 0
        no_macros = 0
        with open(output / "train.jsonl", encoding="utf-8") as f:
            for line in f:
                text = json.loads(line)["text"]
                if "<|M" in text:
                    has_macros += 1
                else:
                    no_macros += 1

        assert has_macros > 0, "Expected some compressed files"
        assert no_macros > 0, "Expected some original files"


def test_eval_all_compressed():
    """All eval lines should contain macro tokens."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        files = {"a.cs": SAMPLE_CS, "b.cs": SAMPLE_CS, "c.cs": SAMPLE_CS}
        repos = {"a.cs": "train-repo", "b.cs": "eval-repo", "c.cs": "eval-repo"}
        corpus = _make_corpus(tmp_path, files, repos)
        output = tmp_path / "out"

        prepare_data(corpus, output, language="csharp", eval_repos=["eval-repo"])

        with open(output / "eval.jsonl", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 2
        for line in lines:
            text = json.loads(line)["text"]
            assert "<|M" in text, "Eval lines should be compressed"


def test_repo_split():
    """Files from eval repos should only appear in eval.jsonl, not train.jsonl."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        files = {"a.cs": SAMPLE_CS, "b.cs": SAMPLE_CS, "c.cs": SIMPLE_CS}
        repos = {"a.cs": "train-repo", "b.cs": "eval-repo", "c.cs": "train-repo"}
        corpus = _make_corpus(tmp_path, files, repos)
        output = tmp_path / "out"

        prepare_data(corpus, output, language="csharp", eval_repos=["eval-repo"])

        meta = json.loads((output / "meta.json").read_text())
        assert meta["train_files"] == 2
        assert meta["eval_files"] == 1


def test_meta_json_contents():
    """meta.json should contain expected fields."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        corpus = _make_corpus(tmp_path, {"a.cs": SAMPLE_CS})
        output = tmp_path / "out"

        prepare_data(corpus, output, language="csharp", eval_repos=[])

        meta = json.loads((output / "meta.json").read_text())
        assert "dictionary_size" in meta
        assert "compress_ratio" in meta
        assert "train_files" in meta
        assert "eval_files" in meta
        assert "char_reduction_pct" in meta


def test_compress_file_roundtrip():
    """Compressed file should decompress back to original."""
    from sematok.decompressor import Decompressor

    d = CompressionDictionary.from_seed("csharp")
    compressor = Compressor(d, language="csharp")
    decompressor = Decompressor(d)

    compressed = _compress_file(SAMPLE_CS, compressor)
    assert compressed != SAMPLE_CS  # Something got compressed
    assert decompressor.decompress(compressed) == SAMPLE_CS
