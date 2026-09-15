import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOWCASE = ROOT / "showcase" / "intent-alignment-showcase.html"
RESULT = ROOT / "results" / "qwen3.5-4b-zero-shot-v1.1.json"
STATUS = ROOT / "results" / "qwen3.5-4b-zero-shot-v1.1.status.json"


def embedded_evidence() -> dict:
    html = SHOWCASE.read_text(encoding="utf-8")
    match = re.search(
        r'<script id="evidenceData" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert match, "standalone showcase must embed its evidence payload"
    return json.loads(match.group(1))


def test_showcase_embeds_exact_frozen_results():
    showcase = embedded_evidence()
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    status = json.loads(STATUS.read_text(encoding="utf-8"))

    assert showcase["protocol"]["name"] == result["protocol"]["name"]
    assert (
        showcase["protocol"]["dataset_sha256"] == result["protocol"]["dataset_sha256"]
    )
    assert showcase["protocol"]["decoding"] == result["protocol"]["decoding"]
    assert showcase["protocol"]["metric"] == result["protocol"]["metric"]
    assert showcase["protocol"]["threshold"] == result["protocol"]["threshold"]
    assert (
        showcase["protocol"]["result_sha256"]
        == hashlib.sha256(RESULT.read_bytes()).hexdigest()
    )
    assert showcase["protocol"]["result_sha256"] == status["result_sha256"]
    assert showcase["protocol"]["source_commit"] == status["source_commit"]
    assert showcase["protocol"]["model"] == Path(result["model"]["path"]).name
    assert showcase["protocol"]["device"] == f"1 × {result['runtime']['device']}"
    assert showcase["summary"] == result["summary"]
    assert showcase["tasks"] == result["tasks"]


def test_showcase_is_a_single_offline_html_file():
    html = SHOWCASE.read_text(encoding="utf-8")

    assert "<script src=" not in html
    assert re.search(r'<link[^>]+rel="stylesheet"', html) is None
    assert "http://" not in html and "https://" not in html
    assert "/home/" not in html and "/Users/" not in html and "D:\\" not in html
    assert "enable_thinking = false" in html
    assert "HTML 不在浏览器中运行模型" in html
