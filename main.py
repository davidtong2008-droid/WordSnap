"""WordSnap —— 程序入口

职责：启动 QApplication、初始化系统托盘、注册全局热键 Alt+E、
编排"截图 -> 预处理 -> OCR -> 词典查询 -> 入库"流程。
"""
import ctypes
import sys
from ctypes import wintypes

from PyQt6.QtCore import QAbstractNativeEventFilter, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QKeySequence, QPainter, QPixmap, QShortcut
from PyQt6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon

from database import WordDatabase
from dictionary_service import DictionaryService
from main_window import MainWindow
from ocr_engine import (
    TesseractUnavailable,
    extract_words,
    ocr_windows,
    recognize_multi_psm,
)
from screenshot_engine import ScreenshotOverlay, preprocess_image
import llm_service

# ---------------- Windows 全局热键常量 ----------------
WM_HOTKEY = 0x0312
MOD_ALT = 0x0001
HOTKEY_ID = 1
VK_E = 0x45  # 'E' 键虚拟键码


class OcrWorker(QThread):
    """后台线程：识别 -> 大模型纠错 -> 中文释义 -> 入库（不阻塞 UI）。

    识别链路（按精度优先）：
      1. 视觉大模型直接读图（豆包/硅基流动/千问等，需 llm.vision 配置）
      2. Windows 原生 OCR（本地免费，需系统 en-US OCR 语言包）
      3. Tesseract 多 PSM（本地兜底，tessdata_best 高精度模型）
    仅 Tesseract 路径额外做 DeepSeek 文本纠错。
    释义优先级：LLM 中文释义 -> 本地英汉词典 -> WordNet 英文释义。
    """

    succeeded = pyqtSignal(list, int)  # (新增单词结果列表, 跳过的重复数)
    failed = pyqtSignal(str)

    def __init__(self, crop, db: WordDatabase, dict_svc: DictionaryService, parent=None):
        super().__init__(parent)
        self._crop = crop
        self._db = db
        self._dict = dict_svc

    def run(self):
        try:
            engine = "tesseract"
            text = None

            # 1) 视觉大模型直接读图（最高精度）
            if llm_service.is_vision_available():
                text = llm_service.recognize_image(self._crop)
                if text:
                    engine = "vision"

            # 2) Windows 原生 OCR
            if not text:
                text = ocr_windows(self._crop)
                if text:
                    engine = "windows"

            # 3) Tesseract 多 PSM（本地兜底）
            if not text:
                image = preprocess_image(self._crop)
                text = recognize_multi_psm(image)
                if text:
                    engine = "tesseract"

            raw_words = extract_words(text or "")

            # 大模型 OCR 纠错（仅 Tesseract 引擎需要；视觉/Windows OCR 输出已较干净）
            words = raw_words
            if engine == "tesseract" and llm_service.is_available():
                corrected = llm_service.correct_words(text or "")
                if corrected:
                    words = corrected

            # 去重 + 过滤已入库单词
            new_words = []
            seen = set()
            skipped = 0
            for word in words:
                key = word.lower()
                if key in seen:
                    continue
                seen.add(key)
                if self._db.word_exists(word):
                    skipped += 1
                    continue
                new_words.append(word)

            # 大模型批量中文释义（含音标、词性）
            llm_defs = {}
            if new_words and llm_service.is_available():
                llm_defs = llm_service.fetch_definitions(new_words)

            results = []
            for word in new_words:
                info = self._dict.lookup(word)  # 本地词典 -> WordNet 兜底
                ld = llm_defs.get(word.lower())
                if ld:
                    info = {
                        "word": word,
                        "phonetic": ld.get("phonetic", ""),
                        "part_of_speech": ld.get("part_of_speech", "")
                        or info["part_of_speech"],
                        "definition": ld.get("definition", "")
                        or info["definition"],
                    }
                row_id = self._db.add_word(
                    word,
                    info["phonetic"],
                    info["part_of_speech"],
                    info["definition"],
                )
                if row_id is not None:
                    results.append(
                        {
                            "word": word,
                            "first_letter": self._db.first_letter_of(word),
                            "part_of_speech": info["part_of_speech"],
                            "definition": info["definition"],
                        }
                    )
            self.succeeded.emit(results, skipped)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class WinHotkeyFilter(QAbstractNativeEventFilter):
    """在 Qt 事件循环中捕获 WM_HOTKEY 消息（Windows 全局热键）。"""

    def __init__(self, callback):
        super().__init__()
        self._callback = callback

    def nativeEventFilter(self, event_type, message):
        if event_type == b"windows_generic_MSG":
            try:
                msg = wintypes.MSG.from_address(int(message))
            except (TypeError, ValueError):
                return False, 0
            if msg.message == WM_HOTKEY and msg.wParam == HOTKEY_ID:
                self._callback()
                return True, 0
        return False, 0


def make_tray_icon() -> QIcon:
    """程序化生成托盘图标（蓝色圆角方块 + W 字母），无需外部资源文件。"""
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#2d6cdf"))
    painter.drawRoundedRect(4, 4, 56, 56, 12, 12)
    painter.setPen(QColor("#ffffff"))
    font = QFont("Arial", 30, QFont.Weight.Bold)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "W")
    painter.end()
    return QIcon(pixmap)


class WordSnapApp:
    """应用主控制器：托盘、全局热键、截图取词流程。"""

    def __init__(self, app: QApplication):
        self.app = app
        self.db = WordDatabase()
        self.dict_svc = DictionaryService()
        self.window = MainWindow(self.db)
        self._worker = None
        self._capturing = False
        self._hotkey_filter = None
        self._fallback_shortcut = None

        self._setup_tray()
        self._setup_hotkey()

    # ---------------- 系统托盘 ----------------
    def _setup_tray(self):
        self.tray = QSystemTrayIcon(make_tray_icon(), self.app)
        self.tray.setToolTip("WordSnap - 按 Alt+E 截图取词")
        menu = QMenu()
        act_show = menu.addAction("显示主窗口")
        act_show.triggered.connect(self.show_window)
        act_capture = menu.addAction("截图取词 (Alt+E)")
        act_capture.triggered.connect(self.start_capture)
        menu.addSeparator()
        act_quit = menu.addAction("退出程序")
        act_quit.triggered.connect(self.quit)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._on_tray_activated)
        self.tray.show()

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.show_window()

    def show_window(self):
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()

    def quit(self):
        self._unregister_hotkey()
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(3000)
        self.tray.hide()
        self.app.quit()

    # ---------------- 全局热键 ----------------
    def _setup_hotkey(self):
        if sys.platform == "win32":
            try:
                hwnd = int(self.window.winId())
                ok = bool(ctypes.windll.user32.RegisterHotKey(hwnd, HOTKEY_ID, MOD_ALT, VK_E))
                if ok:
                    self._hotkey_filter = WinHotkeyFilter(self.start_capture)
                    self.app.installNativeEventFilter(self._hotkey_filter)
                    return
            except Exception:
                pass
            # 注册失败（热键被占用等）：退化为窗口内快捷键
        self._fallback_shortcut = QShortcut(QKeySequence("Alt+E"), self.window)
        self._fallback_shortcut.activated.connect(self.start_capture)

    def _unregister_hotkey(self):
        if self._hotkey_filter is not None:
            try:
                self.app.removeNativeEventFilter(self._hotkey_filter)
            except Exception:
                pass
            self._hotkey_filter = None
        if sys.platform == "win32":
            try:
                ctypes.windll.user32.UnregisterHotKey(int(self.window.winId()), HOTKEY_ID)
            except Exception:
                pass

    # ---------------- 截图取词流程 ----------------
    def start_capture(self):
        if self._capturing or (self._worker is not None and self._worker.isRunning()):
            return
        self._capturing = True
        self.window.hide()
        self.overlay = ScreenshotOverlay()
        self.overlay.region_selected.connect(self._on_region_selected)
        self.overlay.cancelled.connect(self._on_capture_cancelled)
        self.overlay.destroyed.connect(self._on_overlay_closed)
        self.overlay.start()

    def _on_overlay_closed(self):
        self._capturing = False

    def _on_capture_cancelled(self):
        self.show_window()

    def _on_region_selected(self, crop, rect):
        self._worker = OcrWorker(crop, self.db, self.dict_svc)
        self._worker.succeeded.connect(self._on_ocr_succeeded)
        self._worker.failed.connect(self._on_ocr_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_worker_finished(self):
        self._worker = None

    def _on_ocr_succeeded(self, results, skipped):
        self.show_window()
        if results:
            self.window.add_new_words(results)
            self.tray.showMessage(
                "WordSnap",
                "新增 {} 个单词（跳过 {} 个已有）".format(len(results), skipped),
                QSystemTrayIcon.MessageIcon.Information,
                3000,
            )
        else:
            self.tray.showMessage(
                "WordSnap",
                "未发现新单词（{} 个已在词库中）".format(skipped),
                QSystemTrayIcon.MessageIcon.Information,
                3000,
            )

    def _on_ocr_failed(self, message):
        self.show_window()
        QMessageBox.critical(self.window, "OCR 失败", message)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("WordSnap")
    app.setQuitOnLastWindowClosed(False)
    app.setWindowIcon(make_tray_icon())

    controller = WordSnapApp(app)
    controller.show_window()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
