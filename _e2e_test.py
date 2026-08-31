"""WordSnap 端到端管线测试 v2（含大模型增强）：
生成含英文的测试图 -> OpenCV 预处理 -> Tesseract OCR(tessdata_best)
-> 本地正则提取 -> LLM OCR 纠错 -> LLM 中文释义 -> SQLite 入库 -> 校验。

需要 config.json 中配置了 DeepSeek API Key；未配置时相关步骤自动降级。
"""
import os
import re
import sys
import tempfile

# 项目根目录（脚本所在目录），保证从任意位置运行都能导入
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

import cv2
import numpy as np

import llm_service
from database import WordDatabase
from dictionary_service import DictionaryService
from ocr_engine import extract_words, ocr_windows, recognize_multi_psm
from screenshot_engine import preprocess_image

TMP = tempfile.gettempdir()
TEST_IMG = os.path.join(TMP, "wordsnap_e2e2.png")
DB_PATH = os.path.join(TMP, "wordsnap_e2e2.db")

report = []

# 1) 生成测试图：小字号 + 多行（模拟低清截图，制造 OCR 误差空间）
lines = [
    "The diplomacy of negotiation requires resolution",
    "sovereignty and unambiguous commitment",
    "to a lasting peace between the nations",
]
img = np.full((300, 1200, 3), 255, np.uint8)
for i, line in enumerate(lines):
    cv2.putText(img, line, (20, 60 + i * 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
cv2.imwrite(TEST_IMG, img)
report.append(("test image created", os.path.exists(TEST_IMG)))

# 2) 识别链路（与 main.py 一致）：视觉LLM(未配置跳过) -> Windows OCR -> Tesseract 多PSM
bgr = cv2.imread(TEST_IMG)
engine = "tesseract"
text = None
if llm_service.is_vision_available():
    text = llm_service.recognize_image(bgr)
    if text:
        engine = "vision"
if not text:
    text = ocr_windows(bgr)
    if text:
        engine = "windows"
if not text:
    binary = preprocess_image(bgr)
    text = recognize_multi_psm(binary)
raw_words = extract_words(text or "")
print("engine used:", engine)
print("OCR text:", repr((text or "").strip()))
print("raw words:", raw_words)

# 3) LLM 可用性
available = llm_service.is_available()
report.append(("llm configured", available))
print("llm available:", available)

corrected = None
if available:
    corrected = llm_service.correct_words(text)
    print("corrected words:", corrected)
    report.append(("llm correction returned list", isinstance(corrected, list) and len(corrected) > 0))
    if corrected:
        report.append(("correction keeps real words", all(re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", w) for w in corrected)))

# 4) LLM 中文释义
llm_defs = {}
if available and corrected:
    llm_defs = llm_service.fetch_definitions(corrected)
    print("llm defs:", {k: v["definition"][:30] for k, v in list(llm_defs.items())[:6]})
    report.append(("llm defs non-empty", len(llm_defs) > 0))
    has_cjk = any(re.search(r"[\u4e00-\u9fff]", v["definition"]) for v in llm_defs.values())
    report.append(("llm defs are Chinese", has_cjk))
    has_phonetic = any(v.get("phonetic") for v in llm_defs.values())
    report.append(("llm defs include phonetic", has_phonetic))

# 5) 入库（新签名：phonetic 列）
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
db = WordDatabase(DB_PATH)
svc = DictionaryService()
words = corrected if corrected else raw_words
seen = set()
new_words = []
for w in words:
    key = w.lower()
    if key not in seen:
        seen.add(key)
        new_words.append(w)

added = 0
for w in new_words:
    info = svc.lookup(w)
    ld = llm_defs.get(w.lower())
    if ld:
        info = {
            "phonetic": ld.get("phonetic", ""),
            "part_of_speech": ld.get("part_of_speech", "") or info["part_of_speech"],
            "definition": ld.get("definition", "") or info["definition"],
        }
    if db.add_word(w, info["phonetic"], info["part_of_speech"], info["definition"]):
        added += 1
report.append(("added to db", added == len(new_words)))

print("---- db contents ----")
for rec in db.get_words():
    print("{:<18} | {} | {:<22} | {:<5} | {}".format(
        rec["word"], rec["first_letter"], rec["phonetic"], rec["part_of_speech"], rec["definition"][:45]))

has_letter = all(rec["first_letter"] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ#" for rec in db.get_words())
report.append(("first_letter valid", has_letter))

for name, ok in report:
    print(("PASS" if ok else "FAIL") + " | " + name)
print("---- %d/%d passed ----" % (sum(1 for _, ok in report if ok), len(report)))

for p in (TEST_IMG, DB_PATH):
    try:
        os.remove(p)
    except OSError:
        pass
sys.exit(0 if all(ok for _, ok in report) else 1)
