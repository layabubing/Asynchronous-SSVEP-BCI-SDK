[English](README.en.md) | [中文](README.md)

# 无训练异步 SSVEP-BCI

这是一个面向实时工程集成的 Python 实现：用 CCA 或 FBCCA 完成无需训练的
SSVEP 频率识别，再用独立的异步状态机识别控制/非控制状态。系统不会在每个
分析窗中强制输出命令；只有候选频率同时通过绝对分数、类间裕量和连续驻留判定，
才产生一次 `activated` 事件。

> 本项目是算法与工程组件，不是医疗器械。默认阈值可直接运行，但电极布局、
> 放大器、刺激器、窗口长度和现场噪声都会改变误触率；部署前必须用真实设备的
> 控制/空闲数据做验证。调阈值不涉及模型拟合，算法仍是无训练的。

## 特性

- 标准 CCA：正弦/余弦多谐波参考信号，纯解析计算，无模型文件。
- FBCCA：M3 风格多子带、零相位 Chebyshev-I SOS 滤波和论文权重融合。
- 异步控制：`idle → candidate → active → refractory` 状态机。
- 防误触：绝对分数、第一/第二名相对裕量、驻留、释放和不响应期。
- 通道级容错：剔除含 NaN/Inf 或全平的坏通道；可用通道不足时失败关闭，
  返回非控制且不输出命令。
- 数值稳定：基于 SVD 子空间的 CCA，不显式求协方差逆，支持秩亏通道。
- 流式接口：接受任意长度 `[channels, samples]` 数据块，自动滑窗。
- 时间完整性：可检查设备时间戳连续性；采集断流后明确要求 `reset()`。

## 安装

```bash
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

Linux/macOS 请使用 `.venv/bin/python`。

## 最小实时接口

```python
from async_ssvep import AsyncSSVEPProcessor, DecisionEvent, SSVEPConfig

config = SSVEPConfig(
    sampling_rate=250.0,
    frequencies=(8.0, 10.0, 12.0, 15.0),
    algorithm="fbcca",
    window_seconds=1.0,
    step_seconds=0.1,
    harmonics=3,
    notch_hz=50.0,
)
processor = AsyncSSVEPProcessor(config)


# eeg_chunk 来自采集卡/SDK，必须是 [channels, samples]；这里没有伪造数据。
def on_eeg_chunk(eeg_chunk, first_sample_time):
    for decision in processor.push(
        eeg_chunk,
        first_sample_time=first_sample_time,
    ):
        if decision.event is DecisionEvent.ACTIVATED:
            send_command(decision.command_frequency)
```

输入契约：

- `eeg_chunk`: 二维数值数组，固定通道数，形状为 `[channels, samples]`。
- `first_sample_time`: 可选，当前块首样本的设备或单调时钟秒数。
- 数据幅值单位不限（V、µV 均可），但一个流中必须一致。
- 丢包、重连、换通道或时间跳变后调用 `processor.reset()`。

输出契约：

- `push()` 返回 `list[AsyncDecision]`；数据不足一个分析窗时返回空列表。
- `event == "activated"` 时，`command_frequency` 才是一次性控制命令。
- `state == "active"` 只表示当前锁定状态，不能据此反复执行命令。
- `event == "released"` 表示退出控制并进入可配置的不响应期。
- `decision.as_dict()` 可直接转换成 JSON、消息队列或日志结构。

离线外部数据接口见
[`examples/process_npy.py`](examples/process_npy.py)，采集回调适配见
[`examples/acquisition_callback.py`](examples/acquisition_callback.py)。

## 配置建议

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `window_seconds` | 1.0 s | 越长通常越稳，但延迟越高 |
| `step_seconds` | 0.1 s | 滑窗更新周期 |
| `score_threshold` | 0.18 | 归一化 `rho²` 绝对门限 |
| `margin_threshold` | 0.10 | 第一名相对第二名的最小裕量 |
| `dwell_windows` | 3 | 连续一致多少窗才激活 |
| `release_windows` | 2 | 连续不一致多少窗才释放 |
| `refractory_seconds` | 0.5 s | 释放后的不响应期 |

FBCCA 子带融合权重被归一化，因此输出范围仍为 `[0, 1]`，归一化是所有目标共享
的常数，不改变 FBCCA 的目标排序。默认子带低截止频率
`(6, 14, 22, 30, 38) Hz`，高截止为 `90 Hz`；低采样率下会自动限制到 Nyquist
以下并删除无效高子带。

建议按以下次序做现场验证，而不是直接降低门限：

1. 确认刺激器实际刷新频率、采样率和 `frequencies` 一致。
2. 首选枕区通道（如 O1/Oz/O2、POz 等），检查接触质量和工频噪声。
3. 先检查空闲误触；提高 `score_threshold`、`margin_threshold` 或
   `dwell_windows` 可降低误触。
4. 再检查控制漏检；必要时延长窗口，最后才小幅降低阈值。
5. 在真实任务中统计每分钟误触、激活延迟、释放延迟和分类准确率。

## 测试

```bash
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m ruff check .
```

测试中的确定性合成信号只用于验证算法，不属于生产输入接口。

## 算法来源

1. Lin, Z. et al. “Frequency Recognition Based on Canonical Correlation Analysis
   for SSVEP-Based BCIs.” *IEEE Transactions on Biomedical Engineering* 53(12),
   2610–2614 (2006).
   [DOI: 10.1109/TBME.2006.886577](https://doi.org/10.1109/TBME.2006.886577)
2. Chen, X. et al. “Filter bank canonical correlation analysis for implementing
   a high-speed SSVEP-based brain-computer interface.” *Journal of Neural
   Engineering* 12(4), 046008 (2015).
   [DOI: 10.1088/1741-2560/12/4/046008](https://doi.org/10.1088/1741-2560/12/4/046008)
3. Diez, P. F. et al. “Asynchronous BCI control using high-frequency SSVEP.”
   *Journal of NeuroEngineering and Rehabilitation* 8, 39 (2011). 该工作讨论
   异步阈值、持续 EEG 判定和 refractory time。
   [DOI: 10.1186/1743-0003-8-39](https://doi.org/10.1186/1743-0003-8-39)
4. Kim, H. et al. “Comparison of recognition methods for an asynchronous
   (un-cued) BCI system: an investigation with 40-class SSVEP dataset.”
   *Biomedical Engineering Letters* 14, 617–630 (2024).
   [DOI: 10.1007/s13534-024-00357-4](https://doi.org/10.1007/s13534-024-00357-4)
5. Wang, J. et al. “An Asynchronous Training-Free SSVEP-BCI Detection Algorithm
   for Non-Equal Prior Probability Scenarios.” *IEEE Transactions on Neural
   Systems and Rehabilitation Engineering* 32, 4120–4130 (2024).
   [DOI: 10.1109/TNSRE.2024.3496727](https://doi.org/10.1109/TNSRE.2024.3496727)

本实现直接采用第 1、2 项的无训练频率检测思想；异步层采用第 3–5 项所强调的
非控制状态、门限和时间约束原则。它没有声称复现第 4、5 项的完整实验协议。
