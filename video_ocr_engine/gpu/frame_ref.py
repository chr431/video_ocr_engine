"""DeviceRef —— 设备帧引用（v2 §6.2/D3；S9-3 取代 4/5/6 元组 dev 槽）。

v1 的 `dev` 槽依产生路径承载三种形状（4/5/6 元组），消费者靠
`len(t) >= 6` 分支——本类型把三态显式化：

  - 未裁切：`x_off=None, crop_w=None`（全宽参与缩放）
  - 已裁切：`x_off/crop_w` 就位（区间由 col_ink/autocrop 给出）
  - 延后裁切标记：`sharp` 非空且裁切字段为空（emit 期只打标记，
    OCR 消费批内一次批量裁切——PI-9 的消费者侧下沉）

`owner` 保活底层缓冲（decord NDArray / 池帧 / 宿主数组）；
调用方必须保持 owner 存活直至消费完成（§9 契约 2）。
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class DeviceRef:
    ptr: int
    h: int
    w: int
    owner: object = None
    x_off: int | None = None
    crop_w: int | None = None
    sharp: float | None = None

    @property
    def deferred(self) -> bool:
        """延后裁切标记态（emit 期未裁，等待批内 autocrop）。"""
        return self.crop_w is None

    @property
    def span(self) -> tuple[int, int]:
        """参与缩放的源列区间（未裁切 = 全宽）。"""
        if self.crop_w is None:
            return 0, self.w
        return int(self.x_off or 0), int(self.crop_w)

    def with_crop(self, x_off: int, crop_w: int) -> "DeviceRef":
        """派生已裁切引用（保留 owner/ptr/sharp，丢弃标记态）。"""
        return replace(self, x_off=int(x_off), crop_w=int(crop_w),
                       sharp=None)
