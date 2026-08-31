"""WordSnap —— 大模型增强服务（可选，联网）

提供三个能力：
1. `recognize_image(image_bgr)`：**视觉大模型直接读图识别英文**（最高精度）。
   支持任意 OpenAI 兼容的视觉模型（豆包/火山方舟、硅基流动、千问百炼等），
   在 config.json 的 llm.vision 中配置。
2. `correct_words(raw_text)`：把 Tesseract 原始输出交给大模型纠错
   （修正拼写、合并断词、剔除乱码碎片）。
3. `fetch_definitions(words)`：批量返回中文释义（音标 + 词性 + 中文解释）。

DeepSeek 对话 API 为纯文本模型，无法看图，因此"看图识别"必须走 vision 配置；
DeepSeek 仅承担纠错与释义（文本任务）。全部使用标准库 urllib 调用，
未配置密钥 / 网络失败时自动降级（is_available / is_vision_available 返回 False）。
"""
import base64
import io
import json
import os
import re
import urllib.error
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
REQUEST_TIMEOUT = 45  # 秒
VISION_MAX_SIDE = 2048  # 发送给视觉模型的图片最大边长

_SYSTEM_VISION = (
    "You are a highly accurate OCR engine. Extract ALL English text from the "
    "given image. Preserve word order and line breaks as much as possible. "
    "Output ONLY the extracted text itself — no explanations, no JSON."
)

_SYSTEM_CORRECT = (
    "You are an OCR post-processing expert. You will receive raw text extracted "
    "from a screenshot by Tesseract OCR. It contains English text with typical "
    "OCR errors: misrecognized characters (0/O, 1/l/I, 5/S, rn/m ...), typos, "
    "words split or merged across line breaks, and noise fragments."
)

_SYSTEM_DICT = (
    "You are an English-Chinese dictionary. For each English word you must "
    "return: phonetic (IPA), part_of_speech (abbreviation: n. v. adj. adv. "
    "prep. conj. pron. art. num. interj.), and a concise Chinese definition "
    "explaining the meaning of the word in Chinese (use the most common senses, "
    "2-4 senses max, separated by '；')."
)


def _read_config() -> dict:
    cfg = {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            cfg = json.load(fh) or {}
    except (OSError, ValueError):
        cfg = {}
    return cfg


def load_config():
    """读取文本模型（DeepSeek）API 配置：config.json -> 环境变量覆盖。"""
    cfg = _read_config()
    llm = cfg.get("llm") or {}
    api_key = os.environ.get("DEEPSEEK_API_KEY") or llm.get("api_key") or ""
    base_url = os.environ.get("DEEPSEEK_BASE_URL") or llm.get("base_url") or DEFAULT_BASE_URL
    model = os.environ.get("DEEPSEEK_MODEL") or llm.get("model") or DEFAULT_MODEL
    return api_key.strip(), base_url.strip(), model.strip()


def load_vision_config():
    """读取视觉模型（豆包/硅基流动等）API 配置。"""
    cfg = _read_config()
    vision = (cfg.get("llm") or {}).get("vision") or {}
    api_key = (
        os.environ.get("DEEPSEEK_VISION_API_KEY")
        or vision.get("api_key")
        or ""
    )
    base_url = (
        os.environ.get("DEEPSEEK_VISION_BASE_URL")
        or vision.get("base_url")
        or ""
    )
    model = os.environ.get("DEEPSEEK_VISION_MODEL") or vision.get("model") or ""
    return api_key.strip(), base_url.strip(), model.strip()


def is_available() -> bool:
    """是否已配置文本大模型（DeepSeek）密钥。"""
    return bool(load_config()[0])


def is_vision_available() -> bool:
    """是否已配置视觉大模型（看图识别）。"""
    key, base_url, model = load_vision_config()
    return bool(key and base_url and model)
    return bool(load_config()[0])


def _chat(messages, temperature=0.2, json_mode=True, max_tokens=4096) -> str | None:
    """调用 chat completions，返回 content 字符串；失败返回 None。"""
    api_key, base_url, model = load_config()
    if not api_key:
        return None
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        return content
    except (urllib.error.URLError, KeyError, IndexError, ValueError, OSError):
        return None


def _image_to_data_url(image_bgr) -> str | None:
    """把 numpy BGR 图像缩放到 VISION_MAX_SIDE 内，编码为 base64 PNG 数据 URL。"""
    try:
        from PIL import Image

        if image_bgr is None or image_bgr.size == 0:
            return None
        if image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
            rgb = image_bgr[:, :, :3][:, :, ::-1]  # BGRA -> RGB
        elif image_bgr.ndim == 3:
            rgb = image_bgr[:, :, ::-1]  # BGR -> RGB
        else:
            rgb = image_bgr  # 灰度
        img = Image.fromarray(rgb)
        img.thumbnail((VISION_MAX_SIDE, VISION_MAX_SIDE), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return "data:image/png;base64," + b64
    except Exception:
        return None


def recognize_image(image_bgr):
    """视觉大模型直接读图识别英文，返回识别文本；失败返回 None。

    需要 config.json 的 llm.vision 已配置（豆包/硅基流动/千问等
    OpenAI 兼容视觉模型）。注意：DeepSeek 对话模型无法看图，
    因此该能力必须使用视觉模型。
    """
    api_key, base_url, model = load_vision_config()
    if not (api_key and base_url and model):
        return None
    data_url = _image_to_data_url(image_bgr)
    if not data_url:
        return None
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_VISION},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all English text from this image."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
        "stream": False,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        text = (content or "").strip()
        return text or None
    except (urllib.error.URLError, KeyError, IndexError, ValueError, OSError):
        return None


def _parse_json(content):
    """容错解析模型返回的 JSON（去代码围栏、截取首段 JSON）。"""
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except ValueError:
        pass
    for pattern in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pattern, text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except ValueError:
                continue
    return None


def correct_words(raw_text: str):
    """OCR 纠错：返回清洗后的单词列表；失败返回 None（调用方用本地结果）。

    raw_text 为 Tesseract 原始输出（带上下文，便于模型推断真实单词）。
    """
    user = (
        "Fix the OCR errors and extract the clean English words.\n\n"
        "Rules:\n"
        '- Output ONLY a JSON object: {{"words": ["word1", "word2", ...]}}\n'
        "- Correct obvious OCR typos using context (0->O, 1->l/I, rn->m, etc.)\n"
        "- Merge words wrongly split across lines when context clearly shows it\n"
        "- Drop non-words, pure numbers, and fragments (single stray letters "
        "except a/I, strings with no vowel, garbled tokens)\n"
        "- Deduplicate case-insensitively; keep the first-seen casing\n"
        "- Keep contractions as-is (don't, it's)\n\n"
        "Raw OCR text:\n\"\"\"\n{}\n\"\"\""
    ).format(raw_text or "")
    content = _chat(
        [
            {"role": "system", "content": _SYSTEM_CORRECT},
            {"role": "user", "content": user},
        ]
    )
    data = _parse_json(content)
    if isinstance(data, dict) and isinstance(data.get("words"), list):
        words = [str(w).strip() for w in data["words"] if str(w).strip()]
        return words or None
    return None


def fetch_definitions(words):
    """批量返回 {word: {phonetic, part_of_speech, definition(中文)}}；失败返回 {}。"""
    result = {}
    clean = [str(w).strip() for w in (words or []) if str(w).strip()]
    if not clean:
        return result
    # 分批，避免单次请求过大
    for start in range(0, len(clean), 25):
        batch = clean[start : start + 25]
        user = (
            "Provide dictionary entries for these English words.\n\n"
            "Output ONLY a JSON array, each element like:\n"
            '{{"word": "...", "phonetic": "/.../", '
            '"part_of_speech": "n.", "definition": "中文释义"}}\n\n'
            "If a word is unknown, still return an entry with "
            '"definition": "（生僻词，未收录）".\n\n'
            "Words:\n" + ", ".join(batch)
        )
        content = _chat(
            [
                {"role": "system", "content": _SYSTEM_DICT},
                {"role": "user", "content": user},
            ]
        )
        data = _parse_json(content)
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict) or not item.get("word"):
                    continue
                key = str(item["word"]).lower()
                result[key] = {
                    "word": item["word"],
                    "phonetic": str(item.get("phonetic") or "").strip(),
                    "part_of_speech": str(item.get("part_of_speech") or "").strip(),
                    "definition": str(item.get("definition") or "").strip(),
                }
    return result
