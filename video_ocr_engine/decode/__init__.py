"""decode —— 解码访问层（FrameSource 端口 + decord 适配器）。

port.py 定义 FrameSource 协议；decord_source.py 实现 DecordFrameSource
（打开决策树/降级链/输出格式适配）。driver 消费面为裸 vr。
"""
