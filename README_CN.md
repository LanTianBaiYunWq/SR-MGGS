# SR-MGGS 补充代码说明

本目录为论文的可复现性补充材料。英文 `README.md` 是提交与公开版本的主说明，
本文件仅用于作者内部核对。

## 快速检查

无需GPU、GIS数据或模型权重：

```bash
python examples/run_minimal_demo.py --input examples/minimal_demo/input.json --output outputs/minimal_demo_result.json
python -m unittest tests/test_minimal_demo.py
```

该示例只验证复合相似度、K模板聚合、独立FAR阈值和认证判定逻辑，不复现论文中的
学习型签名或正式指标。

## 完整复现需要

- 线要素层次缓存；
- 面要素层次缓存；
- 可选点要素层次缓存；
- 已随包提供的三个种子对应的SR-MGGS线面主模型权重；
- 已随包提供的generic辅助模型和point辅助模型权重；
- `splits/source_isolated_split.json` 中列出的数据包。

完整命令见 `REPRODUCIBILITY.md`。其中OpenStreetMap来源数据可按ODbL公开获取，但因
体量较大不在本包重复分发；自主采集数据受所有权和再分发限制，不公开提供。正式公开
前应确认受限数据标识已经匿名化、模型权重允许作为衍生成果发布，并补充软件许可证和
最终引用格式。

