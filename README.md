# WordSnap —— 英语单词截图学习工具

Windows 桌面应用：按 **Alt+E** 框选屏幕任意区域的英文内容，自动 OCR 识别单词、
获取释义，并按单词首字母（A-Z）分类存入本地 SQLite 词库（"错题本"）。

## 功能一览

- 🖼 **QQ 截图式交互**：全屏冻结 + 拖拽矩形选框，松开即截图，`ESC` 取消
- 🔤 **OCR（高精度）**：Tesseract 5.x LSTM + **tessdata_best 高精度英文模型** + OpenCV 预处理（灰度 / 高斯模糊 / 自适应二值化 / 低清放大 / 深色背景反相）
- 🤖 **大模型增强（可选，联网）**：DeepSeek API —— OCR 文本纠错（修正拼写/断词）+ **批量中文释义（音标 + 词性 + 中文解释）**，识别不准与释义缺失问题大幅改善
- 📖 **释义优先级**：大模型中文释义 → 内置英汉词典（`dictionary.json`）→ WordNet 英文释义（离线兜底）
- 🗂 **错题本**：按首字母 A-Z 分类（非字母开头归入 `#`），SQLite 持久化，自动去重
- 🖥 **主界面**：左侧字母索引 + 右侧表格（单词/词性/释义/添加时间/已掌握）+ 右键菜单（编辑释义 / 标记已掌握 / 删除）
- 🔔 **系统托盘**：驻留托盘，提供"显示主窗口 / 截图取词 / 退出程序"

## 环境要求

- Windows 10/11，Python 3.10+
- 安装依赖：

```powershell
pip install -r requirements.txt
```

## 安装 Tesseract OCR（必需）

1. 下载 Tesseract 5.x 安装包（LSTM 版，含英文语言包）：
   <https://github.com/UB-Mannheim/tesseract/wiki>
2. 默认安装到 `C:\Program Files\Tesseract-OCR\`（代码已写死该路径）。
   若自定义路径，请修改 `ocr_engine.py` 顶部的 `TESSERACT_DEFAULT` / `TESSDATA_DEFAULT`。
3. 安装时确保勾选 **English** 语言包（默认已勾选）。

## 运行

```powershell
python main.py
```

- 首次运行会自动下载 WordNet 语料到项目本地 `nltk_data/` 目录（需联网一次，之后完全离线）。
- 之后按 `Alt+E` 即可截图取词。

> 若 `Alt+E` 全局热键注册失败（被其他程序占用），程序会自动退化为"窗口内快捷键"，
> 即主窗口获得焦点时按 Alt+E 也可触发截图。

## 大模型增强配置（可选，推荐）

程序会读取 `config.json` 中的 `llm` 配置（也可用环境变量覆盖）：

```json
{
  "llm": {
    "api_key": "sk-xxxxxxxx",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat",
    "vision": {
      "api_key": "",
      "base_url": "https://ark.cn-beijing.volces.com/api/v3",
      "model": "doubao-1.5-vision-pro-32k-250115"
    }
  }
}
```

**识别链路（按精度优先）**：

1. **Windows 原生 OCR**（首选，本地免费、无需联网）——需系统装有英文（en-US）OCR 语言包：
   `设置 → 时间和语言 → 语言 → 英语 → 语言选项 → 光学字符识别`，或管理员执行
   `Add-WindowsCapability -Online -Name "Language.OCR~~~en-US~0.0.1.0"`；
   无英文包时自动跳过（避免中文 OCR 拖低英文识别）。
2. **Tesseract 多 PSM**（本地兜底，tessdata_best 高精度模型），结果再经 DeepSeek 文本纠错。
3. **视觉大模型直接读图**（可选，`llm.vision`）——对复杂背景/艺术字最强，但消耗 API 额度；
   默认关闭（`api_key` 留空即禁用）。如需启用（任选其一）：
   - **豆包 / 火山方舟**：<https://console.volcengine.com/ark> 创建 API Key 与**推理接入点**（`ep-xxx`），
     `base_url` 填 `https://ark.cn-beijing.volces.com/api/v3`，`model` 填接入点 ID；
   - **硅基流动 SiliconFlow**（免费额度）：`base_url` 填 `https://api.siliconflow.cn/v1`，
     `model` 填 `Qwen/Qwen2.5-VL-72B-Instruct`；
   - **千问百炼**：`base_url` 填 `https://dashscope.aliyuncs.com/compatible-mode/v1`。

- 配置后：截图 → 识别 → 中文释义（音标 + 词性）全链路增强；未配置时自动降级，功能不受影响。
- ⚠️ `config.json` 包含你的 API 密钥，请勿分享该文件；已默认限制为当前用户可读。

## 内置词典说明

- `dictionary.json` 收录约 370 个高频词汇的简明中文释义（词性 + 释义），配合代码内置兜底词典使用。
- 释义优先级：**大模型中文释义 → 内置词典 → WordNet 英文释义（离线兜底）**。
- **扩充到完整 5000 词**：直接向 `dictionary.json` 追加条目即可，格式：

```json
{
  "word": {"pos": "n.", "def": "中文释义"}
}
```

  建议用开源词典（如 ECDICT 的 CSV）生成完整 5000 词文件后整体替换。

## PyInstaller 打包为单文件 .exe

**要点**：把 `tessdata`、`nltk_data`、`dictionary.json` 一并打进 exe，
并在目标电脑上无需联网、无需安装 Python / Tesseract。

### 第 1 步：准备资源目录

```powershell
# 1. 项目内 tessdata 已含高精度 eng 模型（tessdata_best），直接随包打包即可
#    （不要用 Program Files 里的 fast 模型覆盖它）

# 2. 确保本地 WordNet 语料已下载（运行过一次 python main.py 即可）
#    确认项目目录下存在 nltk_data\corpora\wordnet

# 3. （可选）确认 config.json 已配置大模型密钥（未配置则 exe 走离线降级链路）
```

### 第 2 步：执行打包

```powershell
pyinstaller --noconfirm --onefile --windowed --name WordSnap `
  --add-data "tessdata;tessdata" `
  --add-data "nltk_data;nltk_data" `
  --add-data "dictionary.json;." `
  --add-data "config.json;." `
  main.py
```

> Windows 下 `--add-data` 使用分号分隔（`源;目标`），Linux/macOS 使用冒号。

### 第 3 步：验证

- 生成文件：`dist\WordSnap.exe`（单文件，无需 Python 环境）。
- 在未安装 Tesseract 的电脑上测试：截图 OCR 应能正常工作
  （程序会优先使用打包内的 `tessdata`）。
- 首次截图会向 `exe 所在目录` 写入 `wordsnap.db`（词库）与 `nltk_data`（如需）。

### 常见问题

| 问题 | 解决 |
| :--- | :--- |
| 提示"未找到 Tesseract OCR 引擎" | 未打包 tessdata：按第 1 步复制后再打包 |
| OCR 结果为空 / 乱码 | 确认选区包含英文；深色背景会自动反相；低清截图已自动放大 |
| WordNet 释义缺失 | 确认 `nltk_data\corpora\wordnet` 已随包附带 |
| 全局热键无效 | 热键被占用时会自动退化；或用托盘菜单"截图取词 (Alt+E)" |
| HiDPI / 多显示器选区偏移 | 已按屏幕缩放比换算物理像素；多显示器时跟随鼠标所在屏幕 |

## 项目结构

```
WordSnap/
├── main.py                # 程序入口：托盘 + 全局热键 + 截图流程编排
├── screenshot_engine.py   # 截图覆盖层（蒙层+选框）与 OpenCV 预处理
├── ocr_engine.py          # pytesseract 封装 + 正则提取英文单词
├── dictionary_service.py  # WordNet + 内置英汉词典查询
├── database.py            # SQLite 建表 / CRUD / 首字母分类
├── main_window.py         # 主窗口（字母索引 + 表格 + 右键菜单）
├── dictionary.json        # 内置简明英汉字典（可扩充）
├── requirements.txt       # 第三方依赖
└── wordsnap.db            # （运行时自动生成）词库
```
