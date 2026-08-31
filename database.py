"""WordSnap —— SQLite 数据持久化层

数据库文件默认存放在项目目录下的 wordsnap.db。
表 words 字段：id / word(唯一) / first_letter / phonetic / part_of_speech /
definition / created_at(默认当前时间) / mastered(默认 0)。
"""
import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "wordsnap.db")


class WordDatabase:
    """单词库：建表、增删查改、按首字母筛选。"""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS words (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    word           TEXT    NOT NULL UNIQUE,
                    first_letter   TEXT    NOT NULL,
                    phonetic       TEXT    NOT NULL DEFAULT '',
                    part_of_speech TEXT    NOT NULL DEFAULT '',
                    definition     TEXT    NOT NULL DEFAULT '',
                    created_at     TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
                    mastered       INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            # 迁移：为旧版本数据库补充 phonetic 列
            columns = [row[1] for row in conn.execute("PRAGMA table_info(words)").fetchall()]
            if "phonetic" not in columns:
                conn.execute(
                    "ALTER TABLE words ADD COLUMN phonetic TEXT NOT NULL DEFAULT ''"
                )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_words_first_letter ON words(first_letter);"
            )

    @staticmethod
    def first_letter_of(word: str) -> str:
        """提取单词首字母（大写）；非 A-Z 开头统一归入 '#'。"""
        ch = (word or "").strip()[:1].upper()
        return ch if "A" <= ch <= "Z" else "#"

    # ---------------- 写入 ----------------
    def add_word(
        self,
        word: str,
        phonetic: str = "",
        part_of_speech: str = "",
        definition: str = "",
    ):
        """插入新单词；已存在（唯一索引冲突）时返回 None。"""
        word = (word or "").strip()
        if not word:
            return None
        first_letter = self.first_letter_of(word)
        try:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO words (word, first_letter, phonetic, part_of_speech, definition) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (word, first_letter, phonetic, part_of_speech, definition),
                )
                return cur.lastrowid
        except sqlite3.IntegrityError:
            return None  # 单词已存在，跳过本次录入

    def word_exists(self, word: str) -> bool:
        """判断单词是否已收录（不区分大小写）。"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM words WHERE word = ? COLLATE NOCASE LIMIT 1",
                ((word or "").strip(),),
            ).fetchone()
        return row is not None

    # ---------------- 查询 ----------------
    def get_words(self, first_letter: str | None = None):
        """按首字母查询；first_letter 为 None 时返回全部，按单词排序。"""
        with self._connect() as conn:
            if first_letter is None:
                rows = conn.execute(
                    "SELECT * FROM words ORDER BY word COLLATE NOCASE"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM words WHERE first_letter = ? "
                    "ORDER BY word COLLATE NOCASE",
                    (first_letter,),
                ).fetchall()
        return [dict(row) for row in rows]

    def get_word(self, word_id: int):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM words WHERE id = ?", (word_id,)
            ).fetchone()
        return dict(row) if row else None

    def count_words(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM words").fetchone()
        return int(row["c"])

    # ---------------- 更新 / 删除 ----------------
    def update_mastered(self, word_id: int, mastered: bool) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE words SET mastered = ? WHERE id = ?",
                (1 if mastered else 0, word_id),
            )

    def update_definition(self, word_id: int, part_of_speech: str, definition: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE words SET part_of_speech = ?, definition = ? WHERE id = ?",
                (part_of_speech, definition, word_id),
            )

    def delete_word(self, word_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM words WHERE id = ?", (word_id,))
