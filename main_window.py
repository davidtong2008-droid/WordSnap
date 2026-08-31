"""WordSnap —— 主窗口 UI

布局：左侧 A-Z/# 首字母索引 + 右侧单词表格 + 底部状态栏 + 右键菜单。
"""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from database import WordDatabase


class EditDialog(QDialog):
    """编辑词性与释义对话框。"""

    def __init__(self, word: str, pos: str, definition: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("编辑释义 - {}".format(word))
        self._pos = QLineEdit(pos)
        self._definition = QLineEdit(definition)
        self._definition.setMinimumWidth(380)

        form = QFormLayout()
        form.addRow("词性", self._pos)
        form.addRow("中文释义", self._definition)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def values(self) -> tuple:
        return self._pos.text().strip(), self._definition.text().strip()


class MainWindow(QMainWindow):
    """主窗口：字母索引 + 单词表格 + 右键菜单。"""

    COLUMNS = ["单词", "词性", "中文释义", "添加时间", "已掌握"]
    LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ#"

    def __init__(self, db: WordDatabase, parent=None):
        super().__init__(parent)
        self.db = db
        self._loading = False          # 刷新表格时的变更抑制标志
        self._current_letter = None    # None 表示"全部"

        self.setWindowTitle("WordSnap - 英语单词截图学习")
        self.resize(980, 620)

        self._build_ui()
        self._refresh_sidebar()
        self.refresh_table()

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QWidget(self)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # 左侧首字母索引
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(110)
        self.sidebar.itemClicked.connect(self._on_letter_clicked)
        root.addWidget(self.sidebar)

        # 右侧单词表格
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        self.table.itemChanged.connect(self._on_item_changed)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setColumnWidth(0, 150)
        self.table.setColumnWidth(3, 150)
        root.addWidget(self.table, 1)

        self.setCentralWidget(central)

        # 底部状态栏
        self.count_label = QLabel("共 0 个单词")
        self.statusBar().addWidget(self.count_label)

    # ---------------- 索引 ----------------
    def _refresh_sidebar(self):
        self.sidebar.clear()
        all_item = QListWidgetItem("全部")
        self.sidebar.addItem(all_item)
        for ch in self.LETTERS:
            self.sidebar.addItem(QListWidgetItem(ch))
        if self._current_letter is None:
            self.sidebar.setCurrentRow(0)

    def _on_letter_clicked(self, item):
        text = item.text()
        self._current_letter = None if text == "全部" else text
        self.refresh_table()

    def refresh_sidebar_selection(self):
        """把侧边栏高亮同步到当前筛选字母。"""
        target = "全部" if self._current_letter is None else self._current_letter
        for i in range(self.sidebar.count()):
            if self.sidebar.item(i).text() == target:
                self.sidebar.setCurrentRow(i)
                break

    # ---------------- 表格 ----------------
    def refresh_table(self):
        """按当前字母重新加载表格。"""
        self._loading = True
        try:
            rows = self.db.get_words(self._current_letter)
            self.table.setRowCount(0)
            self.table.setRowCount(len(rows))
            for i, rec in enumerate(rows):
                word_item = QTableWidgetItem(rec["word"])
                word_item.setData(Qt.ItemDataRole.UserRole, rec["id"])
                pos_item = QTableWidgetItem(rec["part_of_speech"])
                def_item = QTableWidgetItem(rec["definition"])
                time_item = QTableWidgetItem(rec["created_at"])
                mastered_item = QTableWidgetItem("")
                mastered_item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable
                )
                mastered_item.setCheckState(
                    Qt.CheckState.Checked
                    if rec["mastered"]
                    else Qt.CheckState.Unchecked
                )
                items = [word_item, pos_item, def_item, time_item, mastered_item]
                for col, item in enumerate(items):
                    self.table.setItem(i, col, item)
            self.count_label.setText("共 {} 个单词".format(self.db.count_words()))
        finally:
            self._loading = False

    def _on_item_changed(self, item):
        """勾选"已掌握"复选框时同步数据库。"""
        if self._loading or item.column() != 4:
            return
        word_item = self.table.item(item.row(), 0)
        word_id = word_item.data(Qt.ItemDataRole.UserRole) if word_item else None
        if word_id is None:
            return
        self.db.update_mastered(word_id, item.checkState() == Qt.CheckState.Checked)

    # ---------------- 右键菜单 ----------------
    def _show_context_menu(self, pos):
        row = self.table.rowAt(pos.y())
        if row < 0:
            return
        self.table.selectRow(row)
        menu = QMenu(self)
        act_edit = menu.addAction("编辑释义")
        act_toggle = menu.addAction("标记为已掌握 / 取消")
        act_delete = menu.addAction("删除单词")
        chosen = menu.exec(self.table.viewport().mapToGlobal(pos))
        if chosen is act_edit:
            self._edit_definition(row)
        elif chosen is act_toggle:
            self._toggle_mastered(row)
        elif chosen is act_delete:
            self._delete_word(row)

    def _row_id(self, row: int):
        item = self.table.item(row, 0)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _edit_definition(self, row: int):
        rec = self.db.get_word(self._row_id(row))
        if not rec:
            return
        dlg = EditDialog(rec["word"], rec["part_of_speech"], rec["definition"], self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            pos, definition = dlg.values()
            self.db.update_definition(rec["id"], pos, definition)
            self.refresh_table()

    def _toggle_mastered(self, row: int):
        rec = self.db.get_word(self._row_id(row))
        if not rec:
            return
        self.db.update_mastered(rec["id"], not bool(rec["mastered"]))
        self.refresh_table()

    def _delete_word(self, row: int):
        rec = self.db.get_word(self._row_id(row))
        if not rec:
            return
        answer = QMessageBox.question(
            self, "删除单词", "确定删除单词“{}”吗？".format(rec["word"])
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.db.delete_word(rec["id"])
            self.refresh_table()

    # ---------------- 供 main.py 调用 ----------------
    def add_new_words(self, results):
        """OCR 完成后插入新单词：自动跳转到首个新单词所在字母并刷新。"""
        if results:
            self._current_letter = results[0]["first_letter"]
            self.refresh_sidebar_selection()
        self.refresh_table()

    def select_letter(self, letter: str):
        """外部主动切换筛选字母（如托盘菜单）。"""
        self._current_letter = letter
        self.refresh_sidebar_selection()
        self.refresh_table()
