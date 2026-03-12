# AIGC 文本检测器 (UNAi)

一个基于深度学习的中文 AI 生成文本检测 Web 应用，支持文本粘贴/文件导入、分块检测、阅读器风格界面。

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Flask](https://img.shields.io/badge/Flask-2.0+-green)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red)

## 功能特性

- 📝 **文本输入** - 支持直接粘贴或导入 TXT/Word 文档
- 🔍 **AIGC 检测** - 基于 RoBERTa 中文预训练模型的文本分类
- 📖 **阅读器风格** - 连续文本展示，荧光笔高亮风险段落
- ✏️ **编辑修改** - 点击段落可编辑，实时重新检测
- 📊 **分块检测** - 可调节分块大小 (100/200/500/1000 字)
- 🌐 **Web 界面** - 浏览器直接访问，无需安装

## 在线演示

本地运行后访问: http://localhost:5000

## 快速开始

### 1. 克隆项目

```bash
git clone git@github.com:TJUEZ/unai.git
cd unai
```

### 2. 创建虚拟环境 (推荐)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements_web.txt
```

### 4. 下载模型文件

项目需要以下模型文件（已在仓库中包含）：
- `pytorch_model.bin` - 模型权重
- `config.json` - 模型配置
- `tokenizer_config.json` - 分词器配置
- `vocab.txt` - 词汇表
- `special_tokens_map.json` - 特殊 token 映射

### 5. 启动服务

```bash
python app.py
```

### 6. 访问 Web 界面

打开浏览器访问: http://localhost:5000

## 使用说明

### 基本检测

1. 在右侧文本框粘贴或输入文本
2. 点击"开始检测"按钮
3. 查看左侧阅读区的检测结果

### 分块大小

- 点击顶部"分块大小"按钮选择检测粒度
- 100字：较细，适合短文本
- 200字：默认，推荐使用
- 500字/1000字：较粗，适合长文本

### 编辑修改

1. 点击左侧阅读区任意段落
2. 右侧会显示该段落的编辑框
3. 修改文本后会自动重新检测
4. 观察整体 AIGC 率变化

### 文件导入

支持导入以下格式：
- `.txt` - 文本文件
- `.docx` - Word 文档

## 项目结构

```
AIGC_detector_zhv3/
├── app.py                      # Flask 后端服务
├── requirements_web.txt        # Python 依赖
├── templates/
│   └── index.html             # 前端页面
├── pytorch_model.bin          # 模型权重
├── config.json                # 模型配置
├── tokenizer_config.json      # 分词器配置
├── vocab.txt                  # 词汇表
└── special_tokens_map.json    # 特殊 token
```

## 技术栈

- **后端**: Flask, PyTorch, Transformers
- **前端**: HTML, CSS, JavaScript (原生)
- **模型**: Chinese RoBERTa (chinese-roberta-wwm-ext)

## 常见问题

### Q: 模型加载失败？
A: 确保所有模型文件（.bin, .json, .txt）都在项目根目录

### Q: 检测结果不准确？
A: 当前模型基于特定数据集训练，对某些类型的 AI 文本可能检测效果不佳

### Q: 如何修改端口？
A: 修改 `app.py` 最后一行 `app.run(port=5000, debug=True)` 中的端口号

## 注意事项

1. 首次启动需要加载模型，可能需要等待几秒
2. 长文本检测可能需要更长时间
3. 建议使用 Chrome 或 Edge 浏览器

## License

Apache License 2.0
