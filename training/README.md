# 模型训练代码说明

## 当前实现内容

`training/model.py` 提供专利思路的可训练多任务基线：

- ResNet-101 提取题图的 7×7 空间特征；
- 共享 RoBERTa 编码题干与逐步解答；题干 token 与图像特征进行跨模态注意力，步骤向量对图像区域进行注意力对齐；
- 双向 GRU 建模步骤先后关系，基于标注逻辑边做图消息传播；
- 多任务输出步骤状态、逻辑角色、图像区域注意力和 T5 反馈文本；
- 联合训练损失由状态分类、角色分类、区域注意力 KL 损失及反馈生成损失相加构成。

这是可训练的工程基线，不等同于专利全部层结构的严格复现。词汇级/子句级细粒度实体对齐、关系类型图网络、真实置信度校准等可在有数据后继续迭代。

## 数据目录

在项目根目录创建 `data/`，将图片和 JSONL 清单放在一起：

```text
data/
├── images/
│   ├── sample_001.png
│   └── sample_002.png
├── train.jsonl
└── val.jsonl
```

每行一个样本，`image` 路径相对该 JSONL 文件所在目录。至少准备训练集和验证集，并确保图片路径存在。示例记录：

```json
{"id":"geo-001","image":"images/sample_001.png","question":"如图，已知 AB=AC，AD 平分角 A，证明 BD=DC。","steps":[{"text":"AB=AC（已知）","role":"condition","status":"correct","feedback":"已知条件识别正确。","region":null},{"text":"角 BAD 等于角 CAD","role":"evidence","status":"correct","feedback":"由角平分线定义可得。","region":{"x":0.30,"y":0.18,"width":0.36,"height":0.52}},{"text":"AD=AD（公共边）","role":"evidence","status":"correct","feedback":"公共边相等。","region":null},{"text":"三角形 ABD 全等于三角形 ACD（SAS）","role":"reasoning","status":"correct","feedback":"两边及其夹角对应相等，可用 SAS 判定。","region":null},{"text":"BD=DC","role":"conclusion","status":"correct","feedback":"全等三角形对应边相等，结论成立。","region":null}],"logic_edges":[[1,2],[1,4],[2,4],[3,4],[4,5]]}
```

标签约定：

- `status`: `correct`、`questionable`、`incorrect`、`unsupported`；
- `role`: `condition`、`evidence`、`reasoning`、`conclusion`、`other`；
- `region`: 可选的题图归一化矩形，`x/y/width/height` 均在 0 到 1 范围；不确定时设为 `null`；
- `logic_edges`: 可选的 1 起始步骤编号关系边，省略时自动连接相邻步骤。

反馈 `feedback` 是可选字段，但要训练反馈生成应尽可能为每一步标注反馈。区域定位只从有标注的步骤学习。模型不能从空白字段中学到对应能力。

## 安装与启动

建议使用支持 CUDA 的 PyTorch 安装组合；按机器的 CUDA 版本从 PyTorch 官方安装指引选择匹配的 `torch` 和 `torchvision`，其余依赖安装：

```powershell
pip install -r requirements.txt
python -m pip install torch torchvision
python -m pip install transformers sentencepiece numpy
python -m training.train --config training/train_config.json
```

训练配置在 `training/train_config.json`。默认模型会下载 `roberta-base`、`t5-base` 及 ResNet-101 ImageNet 权重；配置中可换用 Hugging Face 兼容的本地路径/模型 ID。首次下载需网络，训练显存不足时可降低 `batch_size`、冻结视觉 backbone 或使用更小的编码器。

最佳验证集损失检查点写入 `training/checkpoints/geometry_grader/best.pt`，并保存 tokenizer。训练输出不会自动接入当前 Streamlit API 批改页面；需后续增加推理适配器后才能切换到本地训练模型。
