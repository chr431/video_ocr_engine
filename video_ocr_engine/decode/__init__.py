"""decode —— 解码访问层（FrameSource 端口 + decord 适配器）。

S3-2 定义 port.py（FrameSource 协议）；R1（2026-09-27 门面拆解轮）落地
decord_source.py（DecordFrameSource：打开决策树/降级链/输出格式适配）。
driver 消费面当前仍为裸 vr（端口收口属后续轮）。
"""
