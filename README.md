# pairwise-patchmatch-inpaint

## 领域

游戏、图形与多媒体

## 主题

PatchMatch 图像补全

## 说明

固定种子、粗层收敛与分块写回校验。

初始状态下 `imaging/patchmatch_inpaint.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
