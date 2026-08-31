"""WordSnap —— 截图覆盖层与图像预处理

参考 QQ 截图交互：全屏冻结背景 + 拖拽矩形选框，松开完成截图，
按 ESC 取消。支持 HiDPI 缩放与多显示器（跟随鼠标所在屏幕）。
"""
import ctypes
import sys
from ctypes import wintypes

import cv2
import mss
import numpy as np
from PyQt6.QtCore import QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QGuiApplication, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QWidget


class ScreenshotOverlay(QWidget):
    """全屏覆盖层：冻结屏幕 + 拖拽矩形选区。"""

    region_selected = pyqtSignal(object, QRect)  # (BGRA 裁切图, 选区矩形)
    cancelled = pyqtSignal()

    def __init__(self):
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._start = None     # 拖拽起点（本地坐标）
        self._current = None   # 当前鼠标位置
        self._frame = None     # 冻结帧（numpy BGRA）
        self._pixmap = None    # 冻结帧（QPixmap）
        self._dpr = 1.0        # 屏幕缩放比（DIP -> 物理像素）

    # ---------------- 启动 ----------------
    def start(self):
        """冻结鼠标所在屏幕，并全屏显示覆盖层。"""
        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        self._dpr = screen.devicePixelRatio()

        frame = self._freeze_screen(screen)
        self._frame = frame
        height, width = frame.shape[:2]
        qimg = QImage(frame.data, width, height, 4 * width, QImage.Format.Format_RGB32)
        self._pixmap = QPixmap.fromImage(qimg.copy())

        self.setGeometry(screen.geometry())
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus()

    def _freeze_screen(self, screen) -> np.ndarray:
        """使用 mss 捕获鼠标所在显示器的物理像素帧（BGRA）。"""
        with mss.mss() as sct:
            monitors = sct.monitors  # monitors[0] 为虚拟桌面，[1:] 为各显示器

            # 物理像素下的鼠标坐标
            if sys.platform == "win32":
                point = wintypes.POINT()
                ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
                cursor = (point.x, point.y)
            else:
                pos = QCursor.pos()
                cursor = (pos.x(), pos.y())

            # 优先选择包含鼠标光标的显示器
            target = None
            for mon in monitors[1:]:
                if (
                    mon["left"] <= cursor[0] < mon["left"] + mon["width"]
                    and mon["top"] <= cursor[1] < mon["top"] + mon["height"]
                ):
                    target = mon
                    break

            # 回退：选择与 QScreen 几何最接近的显示器
            if target is None:
                geo = screen.geometry()
                best = None
                for mon in monitors[1:]:
                    score = abs(mon["left"] - geo.x() * self._dpr) + abs(
                        mon["top"] - geo.y() * self._dpr
                    )
                    if best is None or score < best[0]:
                        best = (score, mon)
                target = best[1] if best else monitors[0]

            shot = sct.grab(target)
            return np.array(shot)  # BGRA

    # ---------------- 绘制 ----------------
    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._pixmap)
        # 半透明遮罩（选区外压暗）
        painter.fillRect(self.rect(), QColor(0, 0, 0, 120))
        if self._start is not None and self._current is not None:
            sel = QRect(self._start, self._current).normalized()
            # 把选区内的原图"切回"到蒙层上（QQ 截图效果）
            painter.drawPixmap(sel, self._pixmap, sel)
            # 边框 + 尺寸提示
            pen = QPen(QColor(255, 120, 60), 2)
            painter.setPen(pen)
            painter.drawRect(sel)
            tip = "{} x {}".format(sel.width(), sel.height())
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(sel.left() + 4, max(10, sel.top() - 6), tip)
        painter.end()

    # ---------------- 鼠标 / 键盘 ----------------
    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._start = event.position().toPoint()
            self._current = self._start
            self.update()

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._start is not None:
            self._current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._start is not None:
            self._current = event.position().toPoint()
            sel = QRect(self._start, self._current).normalized()
            self._start = None
            self._current = None
            if sel.width() < 4 or sel.height() < 4:
                # 选区过小视为误操作，取消
                self.cancelled.emit()
                self.close()
                return
            crop = self._crop(sel)
            self.region_selected.emit(crop, sel)
            self.close()

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            self.close()
        else:
            super().keyPressEvent(event)

    def _crop(self, sel: QRect) -> np.ndarray:
        """按选区裁切冻结帧（换算 HiDPI 物理像素并夹紧边界）。"""
        x = int(round(sel.x() * self._dpr))
        y = int(round(sel.y() * self._dpr))
        w = int(round(sel.width() * self._dpr))
        h = int(round(sel.height() * self._dpr))
        frame_h, frame_w = self._frame.shape[:2]
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = max(1, min(w, frame_w - x))
        h = max(1, min(h, frame_h - y))
        return self._frame[y : y + h, x : x + w].copy()


# ============================================================
# 图像预处理（OpenCV）：灰度 → 高斯模糊降噪 → 自适应二值化
# ============================================================
def preprocess_image(img_bgra: np.ndarray) -> np.ndarray:
    """对截图执行 OCR 预处理，返回白底黑字的二值图。

    处理链：BGRA->BGR -> 灰度 -> 低分辨率放大 -> 深色背景反相
    -> 高斯模糊 -> 自适应阈值二值化（必要时退回 Otsu）-> 中值滤波。
    """
    if img_bgra is None or img_bgra.size == 0:
        raise ValueError("截图区域为空，无法识别")

    if img_bgra.ndim == 2:
        gray = img_bgra
    elif img_bgra.ndim == 3 and img_bgra.shape[2] == 4:
        gray = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(img_bgra, cv2.COLOR_BGR2GRAY)

    # 低分辨率放大（提升小字号 / 低清截图的识别率）
    height, width = gray.shape[:2]
    max_side = max(height, width)
    if 0 < max_side < 1200:
        scale = 1200.0 / max_side
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 深色背景（如暗色终端 / 夜间模式网页）自动反相为白底黑字
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    if border.mean() < 127:
        gray = 255 - gray

    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 二值化：Otsu 全局阈值优先（对规整截图更稳、笔画不易断裂）；
    # 若结果异常（黑像素占比过小/过大，说明背景复杂或光照不均），
    # 退回自适应阈值。
    _, threshold = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    black_ratio = np.count_nonzero(threshold == 0) / threshold.size
    if black_ratio < 0.02 or black_ratio > 0.5:
        threshold = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 11
        )

    # 中值滤波去噪
    threshold = cv2.medianBlur(threshold, 3)
    return threshold
