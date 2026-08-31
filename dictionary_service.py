"""WordSnap —— 词典服务

释义优先级：
1. 内置简明英汉字典（dictionary.json，合并代码内置兜底词典）；
2. WordNet（本地离线英文释义，词性 + 英文定义）。

WordNet 语料在首次运行时自动下载到项目本地 nltk_data 目录，
之后完全离线可用（打包时请一并带上 nltk_data 目录）。
"""
import json
import os

import nltk
from nltk.corpus import wordnet as wn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")
JSON_DICT_PATH = os.path.join(BASE_DIR, "dictionary.json")

# 确保 WordNet 语料可用（下载到项目本地目录，保证离线）
nltk.data.path.insert(0, NLTK_DATA_DIR)
if not os.path.isdir(os.path.join(NLTK_DATA_DIR, "corpora", "wordnet")):
    try:
        nltk.download("wordnet", download_dir=NLTK_DATA_DIR, quiet=True)
    except Exception:
        pass  # 离线环境跳过，WordNet 查询将自动降级

# WordNet 词性 -> 缩写
_POS_MAP = {
    wn.NOUN: "n.",
    wn.VERB: "v.",
    wn.ADJ: "a.",
    wn.ADJ_SAT: "a.",
    wn.ADV: "r.",
}

# 代码内置兜底词典（dictionary.json 存在时会被合并/覆盖）
_BUILTIN = {
    "the": {"pos": "art.", "def": "这；那"},
    "be": {"pos": "v.", "def": "是；存在"},
    "to": {"pos": "prep.", "def": "向；到；对于"},
    "of": {"pos": "prep.", "def": "……的；关于"},
    "and": {"pos": "conj.", "def": "和；与"},
    "a": {"pos": "art.", "def": "一（个）；每一"},
    "in": {"pos": "prep.", "def": "在……里；在……期间"},
    "that": {"pos": "pron./conj.", "def": "那；那个；引导从句"},
    "have": {"pos": "v.", "def": "有；拥有；吃"},
    "i": {"pos": "pron.", "def": "我"},
    "it": {"pos": "pron.", "def": "它"},
    "for": {"pos": "prep.", "def": "为了；对于"},
    "not": {"pos": "ad.", "def": "不；没有"},
    "on": {"pos": "prep.", "def": "在……上；关于"},
    "with": {"pos": "prep.", "def": "和……一起；用"},
    "he": {"pos": "pron.", "def": "他"},
    "as": {"pos": "conj./prep.", "def": "如同；作为"},
    "you": {"pos": "pron.", "def": "你；你们"},
    "do": {"pos": "v.", "def": "做；干"},
    "at": {"pos": "prep.", "def": "在（某处/时刻）"},
    "this": {"pos": "pron./a.", "def": "这；这个"},
    "but": {"pos": "conj.", "def": "但是；而是"},
    "his": {"pos": "pron./a.", "def": "他的"},
    "by": {"pos": "prep.", "def": "被；由；在……旁边"},
    "from": {"pos": "prep.", "def": "从；来自"},
    "they": {"pos": "pron.", "def": "他们；她们；它们"},
    "we": {"pos": "pron.", "def": "我们"},
    "say": {"pos": "v.", "def": "说；讲"},
    "her": {"pos": "pron./a.", "def": "她的；她"},
    "she": {"pos": "pron.", "def": "她"},
    "or": {"pos": "conj.", "def": "或者；还是"},
    "an": {"pos": "art.", "def": "一（个）"},
    "will": {"pos": "v./n.", "def": "将；愿意；意志"},
    "my": {"pos": "pron./a.", "def": "我的"},
    "one": {"pos": "num./pron.", "def": "一；一个"},
    "all": {"pos": "a./pron.", "def": "全部的；所有"},
    "would": {"pos": "v.", "def": "将；愿意（will 的过去式）"},
    "there": {"pos": "ad.", "def": "在那里；有（there be）"},
    "their": {"pos": "pron./a.", "def": "他们的"},
    "what": {"pos": "pron./a.", "def": "什么"},
    "so": {"pos": "ad./conj.", "def": "如此；所以"},
    "up": {"pos": "ad./prep.", "def": "向上；起来"},
    "out": {"pos": "ad.", "def": "在外；出去"},
    "if": {"pos": "conj.", "def": "如果；是否"},
    "about": {"pos": "prep./ad.", "def": "关于；大约"},
    "who": {"pos": "pron.", "def": "谁"},
    "get": {"pos": "v.", "def": "得到；变得；到达"},
    "which": {"pos": "pron./a.", "def": "哪一个；哪些"},
    "go": {"pos": "v.", "def": "去；走"},
    "me": {"pos": "pron.", "def": "我（宾格）"},
    "when": {"pos": "ad./conj.", "def": "什么时候；当……时"},
    "make": {"pos": "v.", "def": "制作；使得"},
    "can": {"pos": "v.", "def": "能；可以"},
    "like": {"pos": "v./prep.", "def": "喜欢；像"},
    "time": {"pos": "n.", "def": "时间；次数"},
    "no": {"pos": "ad./a.", "def": "不；没有"},
    "just": {"pos": "ad.", "def": "刚刚；仅仅"},
    "him": {"pos": "pron.", "def": "他（宾格）"},
    "know": {"pos": "v.", "def": "知道；认识"},
    "take": {"pos": "v.", "def": "拿；带走；花费"},
    "people": {"pos": "n.", "def": "人们；人民"},
    "into": {"pos": "prep.", "def": "到……里面"},
    "year": {"pos": "n.", "def": "年"},
    "your": {"pos": "pron./a.", "def": "你的；你们的"},
    "good": {"pos": "a.", "def": "好的"},
    "some": {"pos": "a./pron.", "def": "一些"},
    "could": {"pos": "v.", "def": "能；可以（can 的过去式）"},
    "them": {"pos": "pron.", "def": "他们（宾格）"},
    "see": {"pos": "v.", "def": "看见；明白"},
    "other": {"pos": "a./pron.", "def": "其他的；另一个"},
    "than": {"pos": "conj.", "def": "比"},
    "then": {"pos": "ad.", "def": "然后；那时"},
    "now": {"pos": "ad.", "def": "现在"},
    "look": {"pos": "v.", "def": "看；看起来"},
    "only": {"pos": "ad./a.", "def": "仅仅；唯一的"},
    "come": {"pos": "v.", "def": "来"},
    "its": {"pos": "pron./a.", "def": "它的"},
    "over": {"pos": "prep./ad.", "def": "在……上方；结束"},
    "think": {"pos": "v.", "def": "想；认为"},
    "also": {"pos": "ad.", "def": "也；而且"},
    "back": {"pos": "ad./n.", "def": "回来；背部"},
    "after": {"pos": "prep./conj.", "def": "在……之后"},
    "use": {"pos": "v.", "def": "使用"},
    "two": {"pos": "num.", "def": "二"},
    "how": {"pos": "ad.", "def": "怎样；多么"},
    "our": {"pos": "pron./a.", "def": "我们的"},
    "work": {"pos": "n./v.", "def": "工作"},
    "first": {"pos": "num./a.", "def": "第一；首先"},
    "well": {"pos": "ad./n.", "def": "好；井"},
    "way": {"pos": "n.", "def": "路；方法"},
    "even": {"pos": "ad.", "def": "甚至"},
    "new": {"pos": "a.", "def": "新的"},
    "want": {"pos": "v.", "def": "想要"},
    "because": {"pos": "conj.", "def": "因为"},
    "any": {"pos": "a./pron.", "def": "任何"},
    "these": {"pos": "pron./a.", "def": "这些"},
    "give": {"pos": "v.", "def": "给"},
    "day": {"pos": "n.", "def": "天；白天"},
    "most": {"pos": "a./ad.", "def": "最多的；最"},
    "us": {"pos": "pron.", "def": "我们（宾格）"},
}

_LOCAL_DICT = None  # 延迟加载的本地词典


def _load_local_dict() -> dict:
    """加载 dictionary.json（若存在），并合并代码内置兜底词典。"""
    merged = dict(_BUILTIN)
    try:
        with open(JSON_DICT_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            for key, value in data.items():
                merged[str(key).lower()] = value
    except (OSError, ValueError):
        pass
    return merged


class DictionaryService:
    """单词释义服务：本地英汉词典 -> WordNet 英文释义。"""

    def __init__(self):
        global _LOCAL_DICT
        if _LOCAL_DICT is None:
            _LOCAL_DICT = _load_local_dict()
        self._local = _LOCAL_DICT

    def lookup(self, word: str) -> dict:
        """返回标准释义结构：
        {word, phonetic, part_of_speech, definition, source}
        source 取值：'local'（本地词典）/ 'wordnet' / 'none'。
        """
        key = (word or "").strip().lower()
        if not key:
            return {
                "word": word,
                "phonetic": "",
                "part_of_speech": "",
                "definition": "",
                "source": "none",
            }

        # 1) 本地英汉词典
        entry = self._local.get(key)
        if entry:
            return {
                "word": word,
                "phonetic": entry.get("phonetic", ""),
                "part_of_speech": entry.get("pos", ""),
                "definition": entry.get("def", ""),
                "source": "local",
            }

        # 2) WordNet（离线英文释义）
        try:
            synsets = wn.synsets(key)
            if synsets:
                synset = synsets[0]
                return {
                    "word": word,
                    "phonetic": "",
                    "part_of_speech": _POS_MAP.get(synset.pos(), "?"),
                    "definition": synset.definition(),
                    "source": "wordnet",
                }
        except Exception:
            pass

        # 3) 未收录
        return {
            "word": word,
            "phonetic": "",
            "part_of_speech": "",
            "definition": "（未收录，请右键编辑释义）",
            "source": "none",
        }
