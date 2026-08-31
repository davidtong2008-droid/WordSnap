"""WordSnap —— OCR 引擎封装

封装 pytesseract 调用与英文单词提取。
依赖：Tesseract 5.x（LSTM 英文引擎），默认安装路径：
    C:\\Program Files\\Tesseract-OCR\\tesseract.exe
"""
import os
import re
import sys

import pytesseract
from pytesseract import TesseractNotFoundError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TESSERACT_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSDATA_DEFAULT = r"C:\Program Files\Tesseract-OCR\tessdata"

# 提取纯英文单词（允许撇号缩写，如 don't、it's）
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


class TesseractUnavailable(RuntimeError):
    """Tesseract 缺失或配置错误的专用异常（供 UI 展示安装引导）。"""


def resource_path(relative: str) -> str:
    """兼容 PyInstaller 打包：返回运行时资源目录（_MEIPASS 或项目目录）。"""
    base = getattr(sys, "_MEIPASS", BASE_DIR)
    return os.path.join(base, relative)


def configure_tesseract() -> None:
    """配置 Tesseract 可执行文件路径与 tessdata 目录。

    说明：pytesseract 在 Windows 上以 posix=False 的 shlex 拆分 config 字符串，
    带空格的路径加引号会被拆碎（shlex 不处理引号），因此这里改用
    TESSDATA_PREFIX 环境变量（tesseract 5.x 会直接把它当作 tessdata 目录），
    彻底绕开该问题。
    """
    bundled = resource_path(os.path.join("tessdata", "tesseract.exe"))
    pytesseract.pytesseract.tesseract_cmd = (
        bundled if os.path.exists(bundled) else TESSERACT_DEFAULT
    )
    tdir = tessdata_dir()
    if os.path.isdir(tdir):
        os.environ["TESSDATA_PREFIX"] = tdir


def tessdata_dir() -> str:
    """返回可用的 tessdata 目录（打包内优先）。"""
    bundled = resource_path("tessdata")
    if os.path.isdir(bundled):
        return bundled
    return TESSDATA_DEFAULT


def extract_words(text: str) -> list:
    """从 OCR 文本中提取英文单词，过滤长度 < 2 的项。

    返回按原文顺序出现的单词列表（大小写保留，供后续入库）。
    """
    out = []
    for match in WORD_RE.finditer(text or ""):
        word = match.group(0)
        if len(word) < 2:
            continue
        out.append(word)
    return out


def recognize(preprocessed_image, lang: str = "eng") -> str:
    """对预处理后的图像（numpy 数组）执行 OCR，返回原始文本。

    若未安装 Tesseract，抛出 TesseractUnavailable（含安装引导信息）。
    """
    configure_tesseract()
    try:
        # tessdata 目录已通过 TESSDATA_PREFIX 环境变量指定（见 configure_tesseract）
        config = "--psm 6"
        return pytesseract.image_to_string(preprocessed_image, lang=lang, config=config)
    except TesseractNotFoundError as exc:
        raise TesseractUnavailable(
            "未找到 Tesseract OCR 引擎！\n\n"
            "请安装 Tesseract 5.x（LSTM 英文语言包）后重试：\n"
            "  下载页：https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  默认安装路径：C:\\Program Files\\Tesseract-OCR\\tesseract.exe\n\n"
            "安装完成后请重启 WordSnap。"
        ) from exc


def recognize_multi_psm(preprocessed_image, lang: str = "eng") -> str:
    """多 PSM 策略：合并 PSM 6（块）与 PSM 11（稀疏文本）两次识别结果，
    提升漏检词的召回率。返回合并后的文本。"""
    configure_tesseract()
    texts = []
    for psm in ("6", "11"):
        try:
            texts.append(
                pytesseract.image_to_string(
                    preprocessed_image, lang=lang, config="--psm " + psm
                )
            )
        except TesseractNotFoundError as exc:
            raise TesseractUnavailable(
                "未找到 Tesseract OCR 引擎！\n\n"
                "请安装 Tesseract 5.x（LSTM 英文语言包）后重试：\n"
                "  下载页：https://github.com/UB-Mannheim/tesseract/wiki\n"
                "  默认安装路径：C:\\Program Files\\Tesseract-OCR\\tesseract.exe\n\n"
                "安装完成后请重启 WordSnap。"
            ) from exc
        except Exception:
            continue
    return "\n".join(t for t in texts if t)


def ocr_windows(image_bgr) -> str | None:
    """Windows 原生 OCR（Windows.Media.Ocr，本地、免费、无需 Key）。

    仅当系统装有英文 OCR 语言包（en-US 等）时启用——中文 OCR 语言包
    识别英文效果很差，会主动跳过，避免拖低精度。
    失败或不可用时返回 None。
    """
    try:
        import asyncio

        import winocr
        from PIL import Image
        from winrt.windows.media.ocr import OcrEngine

        langs = [lang.language_tag for lang in OcrEngine.available_recognizer_languages]
        english = [l for l in langs if l.lower().startswith("en")]
        if not english:
            return None  # 无英文语言包，跳过（避免中文 OCR 破坏英文识别）
        lang = "en-US" if "en-US" in english else english[0]
        if image_bgr is None or image_bgr.size == 0:
            return None
        if image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
            rgb = image_bgr[:, :, :3][:, :, ::-1]
        elif image_bgr.ndim == 3:
            rgb = image_bgr[:, :, ::-1]
        else:
            rgb = image_bgr
        img = Image.fromarray(rgb)
        result = asyncio.run(winocr.to_coroutine(winocr.recognize_pil(img, lang=lang)))
        text = (result.text or "").strip()
        return text or None
    except Exception:
        return None
