# 设备状态评估与预警（动态阈值 + 决策树）

该项目提供一个基于多源数据的设备状态评估与预警方案：
- **动态阈值**：结合滑动窗口统计与 EWMA 生成自适应阈值。
- **决策树模型**：对设备状态（正常/预警/严重）进行分类。
- **多源数据融合**：对振动、温度、压力等指标进行综合评估。

## 快速开始

```bash
pip install -r requirements.txt
python src/device_monitor.py
```

## 示例用法

```python
import pandas as pd
from src.device_monitor import DeviceStateEvaluator, simulate_device_data

data, labels = simulate_device_data()

evaluator = DeviceStateEvaluator()
evaluator.fit(data, labels)

result = evaluator.evaluate(data)
print(result.tail())
```

输出列说明：
- `state`: 0=正常，1=预警，2=严重
- `risk_score`: 0~1 之间的风险评分
