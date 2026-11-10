# MathVision 几何解答批改 Demo

基于 Streamlit 和 OpenAI 兼容多模态 API 的几何解答步骤批改原型，当前已配置 CCNU 平台和 `XAI@CCNU/Qwen3-Omni-30B`。支持题目文本、可选题图及逐行解答步骤输入，输出步骤状态、解释、置信度、逻辑关系、模型建议图像区域和 JSON 报告。

页面默认载入 `docs/pgps9k_patent_example.docx` 中的 PGPS9K 几何题样例：题目、三条学生步骤、七类状态概率、逐步反馈、原始题图和 S3 热力图。文档中的预计输出会作为预设参考案例展示；点击“开始智能评估”后显示实时结果。每一步的四类状态、置信度、反馈和七类状态概率由同一次评估请求返回；七类概率作为补充字段，不覆盖原判定。抽取出的样例字段位于 `examples/pgps9k_patent_example.json`，配图位于 `imgs/`。

> 这是 LLM API 驱动的原型，不是专利所述训练模型的完整复现。区域框是模型预测的可视化建议，需人工核验。

## 平台配置

应用从项目根目录的 `config.json` 读取当前平台。首次配置时，将 `config.example.json` 复制为 `config.json`，再填写 API 密钥。密钥只存放在本地配置文件，该文件已加入 `.gitignore`。

配置结构使用 OpenAI 兼容格式。需要接入其他兼容平台时，在 `providers` 中增加平台配置并将 `active_provider` 设置为对应名称：

```json
{
  "active_provider": "ccnu",
  "providers": {
    "ccnu": {
      "display_name": "CCNU",
      "api_style": "openai_compatible",
      "base_url": "http://172.0.163.111:4000/v1",
      "api_key": "你的 API Key",
      "model": "XAI@CCNU/Qwen3-Omni-30B"
    }
  }
}
```

当前代码调用 Chat Completions 兼容接口。其他平台若接口协议不同，需要为该平台增加对应 API adapter。

## 安装与启动

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run main.py
```

API 密钥、地址和模型均不再从环境变量读取。

Streamlit 端口固定配置在 `.streamlit/config.toml`，启动后访问 `http://localhost:8501`。

## 模型训练

多模态训练基线、JSONL 标注格式、训练参数和启动命令见 [training/README.md](training/README.md)。训练依赖单独安装：先依据设备选择匹配的 PyTorch/torchvision，再安装 `requirements-training.txt`。
