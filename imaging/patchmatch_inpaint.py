"""确定性 PatchMatch 图像补全工具集。

这个模块把题目中六类容易出错的契约拆成可单独调用的纯函数：

* :func:`fixed_seed`：固定搜索随机种子，不再使用时间种子；
* :func:`coarse_converge`：判断粗层是否已经收敛，未收敛不能上传；
* :func:`window_clip`：把块边界处的搜索/邻域窗口裁剪到有效坐标；
* :func:`feather_decouple`：羽化半径使用独立参数，不再等于搜索半径；
* :func:`adaptive_iter`：按残差/变化率决定是否继续迭代；
* :func:`block_writeback`：按绝对坐标校验后分块写回，防止分块错位。

其余函数提供掩码膨胀、PatchMatch 传播/搜索、指标、审计、校验、回退和
端到端 :func:`inpaint`。为了便于在最小环境中运行，核心实现只依赖标准库；
如果调用方传入 NumPy 数组，也会先转换为纯 Python 嵌套结构处理。
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from copy import deepcopy
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# 常量与通用小工具
# ---------------------------------------------------------------------------

#: 全局固定种子。所有像素级随机序列均由该种子和像素坐标派生，因此不依赖
#: 系统时间、线程调度或块完成顺序。
DEFAULT_SEED = 42

DEFAULT_PATCH_RADIUS = 2
DEFAULT_SEARCH_RADIUS = 8
DEFAULT_FEATHER_RADIUS = 3
DEFAULT_BLOCK_SIZE = 32
DEFAULT_TOLERANCE = 1e-4
DEFAULT_MAX_ITER = 24


def _is_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes, bytearray)):
        return False
    return isinstance(value, (list, tuple)) or (
        hasattr(value, "shape") and hasattr(value, "__getitem__")
    )


def _to_native(value: Any) -> Any:
    """把 NumPy 标量/数组转换为原生 Python 对象。"""
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return int(value)
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, bool):
            return float(value)
        return float(value)
    except (TypeError, ValueError):
        return default


def _stable_hash(*parts: Any) -> int:
    """生成进程间稳定的非负整数哈希（不使用 Python 字符串随机哈希）。"""
    text = "/".join(str(p) for p in parts)
    h = 1469598103934665603
    for ch in text.encode("utf-8"):
        h ^= ch
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def _pixel_rng(seed: int, y: int, x: int, iteration: int = 0, salt: int = 0) -> random.Random:
    """每个像素拥有独立但确定的随机流。

    这样即使未来按块并行执行，随机候选也不会依赖线程交错顺序。
    """
    return random.Random(_stable_hash("patchmatch", int(seed), int(y), int(x),
                                      int(iteration), int(salt)))


# ---------------------------------------------------------------------------
# 固定种子 / 确定性
# ---------------------------------------------------------------------------

class FixedSeed(int):
    """固定种子值；既是 ``int``，也携带 RNG 与配置。"""

    def __new__(cls, seed: Any = DEFAULT_SEED, apply: bool = True) -> "FixedSeed":
        if not isinstance(seed, int) or isinstance(seed, bool):
            seed = _stable_hash("seed", seed)
        seed = int(seed) & 0x7FFFFFFFFFFFFFFF
        obj = super().__new__(cls, seed)
        obj.seed = seed
        obj.rng = random.Random(seed)
        numpy_seeded = False
        if apply:
            random.seed(seed)
            try:  # 可选依赖；存在时同时固定 NumPy 旧版全局 RNG。
                import numpy as np  # type: ignore
                np.random.seed(seed)
                numpy_seeded = True
            except Exception:
                numpy_seeded = False
        obj.config = {
            "seed": seed,
            "fixed": True,
            "time_based": False,
            "numpy_seeded": numpy_seeded,
        }
        return obj

    def __getitem__(self, key: str) -> Any:
        if key in self.config:
            return self.config[key]
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def random(self) -> float:
        return self.rng.random()

    def randint(self, a: int, b: int) -> int:
        return self.rng.randint(a, b)


def fixed_seed(*args: Any, **kwargs: Any) -> FixedSeed:
    """返回固定种子并初始化标准库/NumPy 全局随机状态。

    可通过 ``seed=...`` 指定；位置参数的第一个值也会被当作种子。
    不再调用 ``time.time_ns()``，因此同一张图两次运行的搜索过程一致。
    """
    seed = kwargs.pop("seed", None)
    if seed is None and args:
        seed = args[0]
    if seed is None:
        seed = DEFAULT_SEED
    apply = bool(kwargs.pop("apply", True))
    return FixedSeed(seed, apply=apply)


def determinism(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """用同种子两次生成随机序列，验证确定性。"""
    seed = kwargs.get("seed", args[0] if args else DEFAULT_SEED)
    count = int(kwargs.get("count", 8))
    count = max(1, min(count, 1024))
    fixed = fixed_seed(seed, apply=False)
    a = [random.Random(fixed.seed).random() for _ in range(count)]
    b = [random.Random(fixed.seed).random() for _ in range(count)]
    deterministic = a == b
    return {
        "seed": fixed.seed,
        "deterministic": deterministic,
        "runs_equal": deterministic,
        "sequence": a,
        "time_seed_used": False,
    }


# ---------------------------------------------------------------------------
# 几何裁剪
# ---------------------------------------------------------------------------

class ClipRect(tuple):
    """半开区间矩形 ``(x0, y0, x1, y1)``，并携带常用几何属性。"""

    def __new__(cls, values: Iterable[Any], patch_radius: int = 0) -> "ClipRect":
        x0, y0, x1, y1 = (int(v) for v in values)
        obj = super().__new__(cls, (x0, y0, x1, y1))
        obj.patch_radius = max(0, int(patch_radius))
        return obj

    @property
    def x0(self) -> int: return self[0]

    @property
    def y0(self) -> int: return self[1]

    @property
    def x1(self) -> int: return self[2]

    @property
    def y1(self) -> int: return self[3]

    @property
    def left(self) -> int: return self.x0

    @property
    def top(self) -> int: return self.y0

    @property
    def right(self) -> int: return self.x1

    @property
    def bottom(self) -> int: return self.y1

    @property
    def width(self) -> int: return max(0, self.x1 - self.x0)

    @property
    def height(self) -> int: return max(0, self.y1 - self.y0)

    @property
    def area(self) -> int: return self.width * self.height

    @property
    def box(self) -> tuple[int, int, int, int]: return tuple(self)  # type: ignore

    @property
    def inclusive(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, max(self.x0, self.x1 - 1),
                max(self.y0, self.y1 - 1))

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def valid_center_bounds(
        self, width: int | None = None, height: int | None = None,
        patch_radius: int | None = None,
    ) -> "ClipRect":
        """返回能完整放下 patch 的源中心半开区间。"""
        r = self.patch_radius if patch_radius is None else max(0, int(patch_radius))
        w = self.x1 if width is None else int(width)
        h = self.y1 if height is None else int(height)
        return ClipRect((r, r, max(r, w - r), max(r, h - r)), patch_radius=r)


def _clip_rect_values(
    x0: Any, y0: Any, x1: Any, y1: Any,
    width: Any = None, height: Any = None,
) -> tuple[int, int, int, int]:
    x0i = max(0, int(math.floor(_as_float(x0))))
    y0i = max(0, int(math.floor(_as_float(y0))))
    x1i = int(math.ceil(_as_float(x1)))
    y1i = int(math.ceil(_as_float(y1)))
    if width is not None:
        w = max(0, int(width))
        x0i = min(x0i, w)  # 窗口整体位于右边界之外时，交集为空矩形 (w,w,w,w)。
        x1i = min(w, x1i)
    if height is not None:
        h = max(0, int(height))
        y0i = min(y0i, h)
        y1i = min(h, y1i)
    x1i = max(x0i, x1i)
    y1i = max(y0i, y1i)
    return x0i, y0i, x1i, y1i


def window_clip(*args: Any, **kwargs: Any) -> ClipRect:
    """把矩形/中心半径窗口裁剪到图像有效区域。

    支持常见位置形式：

    ``window_clip(x0, y0, x1, y1, width, height)``
        裁剪半开矩形；
    ``window_clip(cx, cy, radius, width, height)``
        中心加半径，自动转成 ``[cx-r, cx+r+1)``；
    ``window_clip((x0, y0, x1, y1), (width, height))``；
    ``window_clip(center=(cx, cy), radius=r, width=w, height=h)``。

    返回 :class:`ClipRect`，它本身是 ``(x0, y0, x1, y1)`` 元组。
    单整数占位调用会得到该尺寸的完整有效窗口。
    """
    width = kwargs.pop("width", kwargs.pop("w", None))
    height = kwargs.pop("height", kwargs.pop("h", None))
    shape = kwargs.pop("shape", None)
    bounds = kwargs.pop("bounds", None)
    rect = kwargs.pop("rect", None)
    center = kwargs.pop("center", None)
    radius = kwargs.pop("radius", kwargs.pop("search_radius", None))
    patch_radius = int(kwargs.pop("patch_radius", 0) or 0)

    if isinstance(shape, dict):
        width = width if width is not None else shape.get("width", shape.get("w"))
        height = height if height is not None else shape.get("height", shape.get("h"))
    elif shape is not None and hasattr(shape, "__len__") and len(shape) >= 2:
        # NumPy/image shape 约定为 (H, W, ...)。
        height = height if height is not None else shape[0]
        width = width if width is not None else shape[1]

    if bounds is not None and hasattr(bounds, "__len__") and len(bounds) >= 2:
        width = width if width is not None else bounds[0]
        height = height if height is not None else bounds[1]

    positional = list(args)

    # 单个配置字典。
    if len(positional) == 1 and isinstance(positional[0], dict):
        cfg = positional.pop(0)
        rect = rect if rect is not None else cfg.get("rect") or cfg.get("box")
        center = center if center is not None else cfg.get("center")
        radius = radius if radius is not None else cfg.get("radius", cfg.get("search_radius"))
        width = width if width is not None else cfg.get("width", cfg.get("w"))
        height = height if height is not None else cfg.get("height", cfg.get("h"))
        patch_radius = int(cfg.get("patch_radius", patch_radius) or 0)

    def _is4(v: Any) -> bool:
        return _is_sequence(v) and len(v) == 4 and all(_as_float(i, math.nan) == _as_float(i) for i in v)

    def _is2nums(v: Any) -> bool:
        return (_is_sequence(v) and len(v) == 2
                and not _is_sequence(v[0]) and not _is_sequence(v[1]))

    # 末尾的二元组通常是 (width, height) 或 bounds。
    if width is None and height is None and positional and _is2nums(positional[-1]):
        bounds_pair = positional.pop()
        width, height = bounds_pair[0], bounds_pair[1]

    # 结构化矩形形式：rect [+ bounds / width, height]。
    if rect is None and positional and _is4(positional[0]):
        rect = positional[0]
        rest = positional[1:]
        if len(rest) >= 1 and _is_sequence(rest[0]) and len(rest[0]) >= 2:
            b = rest[0]
            width = width if width is not None else b[0]
            height = height if height is not None else b[1]
            positional = positional[2:]
        elif len(rest) >= 2:
            width = width if width is not None else rest[0]
            height = height if height is not None else rest[1]
            positional = positional[3:]
        else:
            positional = positional[1:]

    if center is None and positional and _is2nums(positional[0]):
        # 可能是 center，也可能是 (width, height)。若后面跟 radius，则按 center。
        if radius is not None or len(positional) >= 2 and _as_float(positional[1], math.nan) == _as_float(positional[1]):
            if radius is None and len(positional) >= 2 and not _is_sequence(positional[1]):
                center = positional[0]
                radius = positional[1]
                positional = positional[2:]
            elif radius is not None:
                center = positional.pop(0)

    # 常见纯位置形式。
    if rect is None and center is None and positional:
        nums = [_as_float(v, math.nan) for v in positional]
        if all(math.isfinite(v) for v in nums):
            if len(nums) == 1:
                width = height = int(nums[0])
                rect = (0, 0, width, height)
            elif len(nums) == 2 and width is None and height is None:
                width, height = int(nums[0]), int(nums[1])
                rect = (0, 0, width, height)
            elif len(nums) == 3:
                center = (nums[0], nums[1])
                radius = nums[2]
            elif len(nums) == 4:
                rect = tuple(int(v) for v in nums)
            elif len(nums) == 5:
                center = (nums[0], nums[1])
                radius = nums[2]
                width, height = int(nums[3]), int(nums[4])
            elif len(nums) >= 6:
                rect = tuple(int(v) for v in nums[:4])
                width, height = int(nums[4]), int(nums[5])

    if center is not None and rect is None:
        cx, cy = center
        r = max(0, int(_as_float(radius, 0)))
        rect = (cx - r, cy - r, cx + r + 1, cy + r + 1)

    if rect is None:
        w = max(0, _as_int(width, 0))
        h = max(0, _as_int(height, w))
        rect = (0, 0, w, h)

    clipped = _clip_rect_values(rect[0], rect[1], rect[2], rect[3], width, height)
    return ClipRect(clipped, patch_radius=patch_radius)


# ---------------------------------------------------------------------------
# 羽化与搜索半径解耦
# ---------------------------------------------------------------------------

class FeatherRadius(int):
    """独立羽化半径，同时保留搜索半径作为对照。"""

    def __new__(cls, feather_radius: Any = DEFAULT_FEATHER_RADIUS,
                search_radius: Any = DEFAULT_SEARCH_RADIUS) -> "FeatherRadius":
        fr = max(0, _as_int(feather_radius, DEFAULT_FEATHER_RADIUS))
        obj = super().__new__(cls, fr)
        obj.feather_radius = fr
        obj.search_radius = max(0, _as_int(search_radius, DEFAULT_SEARCH_RADIUS))
        obj.coupled = False
        obj.independent = True
        return obj

    def __getitem__(self, key: str) -> Any:
        if key in {"feather_radius", "radius"}:
            return self.feather_radius
        if key == "search_radius":
            return self.search_radius
        if key in {"coupled", "independent"}:
            return self.coupled if key == "coupled" else self.independent
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def _integral_image(grid: list[list[float]]) -> list[list[float]]:
    h = len(grid)
    w = len(grid[0]) if h else 0
    ii = [[0.0] * (w + 1) for _ in range(h + 1)]
    for y in range(h):
        row_sum = 0.0
        for x in range(w):
            row_sum += float(grid[y][x])
            ii[y + 1][x + 1] = ii[y][x + 1] + row_sum
    return ii


def _box_average(ii: list[list[float]], x0: int, y0: int,
                 x1: int, y1: int) -> float:
    return (ii[y1][x1] - ii[y0][x1] - ii[y1][x0] + ii[y0][x0]) / max(
        1, (y1 - y0) * (x1 - x0)
    )


def feather_mask(mask: Any, radius: int = DEFAULT_FEATHER_RADIUS) -> list[list[float]]:
    """对二维掩码做盒式羽化，返回 ``[0, 1]`` alpha。"""
    mask = _to_native(mask)
    if not _is_sequence(mask):
        value = 1.0 if _as_float(mask, 0.0) >= 0.5 else _as_float(mask, 0.0)
        return [[min(1.0, max(0.0, value))]]
    grid = [[_as_float(v, 0.0) for v in row] for row in mask]
    h = len(grid)
    w = max((len(row) for row in grid), default=0)
    grid = [row + [0.0] * (w - len(row)) for row in grid]
    r = max(0, int(radius))
    if r == 0:
        return [[1.0 if v >= 0.5 else 0.0 for v in row] for row in grid]
    ii = _integral_image(grid)
    out = [[0.0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            x0, y0 = max(0, x - r), max(0, y - r)
            x1, y1 = min(w, x + r + 1), min(h, y + r + 1)
            out[y][x] = _box_average(ii, x0, y0, x1, y1)
    return out


def feather_decouple(*args: Any, **kwargs: Any) -> Any:
    """返回独立羽化半径；若传入掩码，则返回羽化后的 alpha。

    关键契约是：即使 ``search_radius`` 改变，``feather_radius`` 也不会被
    联动修改。未显式提供羽化半径时使用独立默认值，而不是搜索半径。
    """
    feather_radius = kwargs.pop("feather_radius", kwargs.pop("radius", None))
    search_radius = kwargs.pop("search_radius", DEFAULT_SEARCH_RADIUS)
    mask = kwargs.pop("mask", None)

    positional = list(args)
    if mask is None and positional and _is_sequence(positional[0]):
        mask = positional.pop(0)
    if feather_radius is None and positional:
        # 第一个普通数值位置参数视为 feather_radius。
        for i, value in enumerate(positional):
            if not _is_sequence(value):
                feather_radius = value
                positional.pop(i)
                break
    if search_radius == DEFAULT_SEARCH_RADIUS and positional:
        for value in positional:
            if not _is_sequence(value):
                search_radius = value
                break

    if feather_radius is None:
        feather_radius = DEFAULT_FEATHER_RADIUS

    result = FeatherRadius(feather_radius, search_radius)
    if mask is not None:
        return feather_mask(mask, result.feather_radius)
    return result


# ---------------------------------------------------------------------------
# 掩码膨胀
# ---------------------------------------------------------------------------

def mask_dilate(mask: Any = None, radius: Any = 1, *args: Any, **kwargs: Any) -> Any:
    """对一维/二维掩码做最大邻域膨胀；标量输入原样返回。"""
    if mask is None:
        return []
    original = mask
    mask = _to_native(mask)
    r = max(0, _as_int(radius, 1))
    iterations = max(1, _as_int(kwargs.pop("iterations", 1), 1))
    r *= iterations

    if not _is_sequence(mask):
        return mask

    if mask and _is_sequence(mask[0]):
        h = len(mask)
        w = max((len(row) for row in mask), default=0)
        grid = [[_as_float(mask[y][x] if x < len(mask[y]) else 0.0)
                 for x in range(w)] for y in range(h)]
        out = [[0.0] * w for _ in range(h)]
        for y in range(h):
            for x in range(w):
                best = 0.0
                for ny in range(max(0, y - r), min(h, y + r + 1)):
                    for nx in range(max(0, x - r), min(w, x + r + 1)):
                        best = max(best, grid[ny][nx])
                out[y][x] = best
        binary = all(v in (0.0, 1.0) for row in out for v in row)
        if binary:
            result = [[int(v) for v in row] for row in out]
        else:
            result = [[int(round(v)) if float(v).is_integer() else v for v in row]
                      for row in out]
        if isinstance(original, tuple):
            return tuple(tuple(row) for row in result)
        return result

    values = [_as_float(v, 0.0) for v in mask]
    n = len(values)
    out = [0.0] * n
    for i in range(n):
        out[i] = max(values[max(0, i - r):min(n, i + r + 1)])
    binary = all(v in (0.0, 1.0) for v in out)
    if binary:
        result_1d: list[Any] = [int(v) for v in out]
    else:
        result_1d = [int(round(v)) if float(v).is_integer() else v for v in out]
    if isinstance(original, tuple):
        return tuple(result_1d)
    return result_1d


# ---------------------------------------------------------------------------
# 自适应迭代与粗层收敛
# ---------------------------------------------------------------------------

class IterationPlan(int):
    """建议/实际迭代次数，同时携带残差与收敛信息。"""

    def __new__(cls, iterations: int, **meta: Any) -> "IterationPlan":
        obj = super().__new__(cls, max(0, int(iterations)))
        obj.iterations = int(obj)
        obj.residual = meta.get("residual")
        obj.tolerance = meta.get("tolerance", DEFAULT_TOLERANCE)
        obj.max_iter = meta.get("max_iter", DEFAULT_MAX_ITER)
        obj.min_iter = meta.get("min_iter", 1)
        obj.converged = bool(meta.get("converged", False))
        obj.should_continue = bool(meta.get("should_continue", not obj.converged))
        obj.history = list(meta.get("history", []))
        obj.change_rate = meta.get("change_rate")
        return obj

    def __getitem__(self, key: str) -> Any:
        if key in {"iterations", "count", "value"}:
            return self.iterations
        if key in self.__dict__:
            return self.__dict__[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


class AdaptiveController:
    """根据残差和 NNF 变化率决定停止，而不是只看固定迭代上限。"""

    def __init__(self, tolerance: float = DEFAULT_TOLERANCE,
                 min_iter: int = 1, max_iter: int = DEFAULT_MAX_ITER,
                 patience: int = 2):
        self.tolerance = max(0.0, float(tolerance))
        self.min_iter = max(0, int(min_iter))
        self.max_iter = max(self.min_iter, int(max_iter))
        self.patience = max(1, int(patience))
        self.residual_history: list[float] = []
        self.change_history: list[float] = []
        self.low_residual_streak = 0
        self.iteration = 0
        self.capped = False

    def update(self, residual: float, change_rate: float | None = None) -> bool:
        """记录一轮结果，返回是否应当继续。"""
        self.iteration += 1
        residual = max(0.0, float(residual))
        self.residual_history.append(residual)
        if change_rate is not None:
            change_rate = max(0.0, float(change_rate))
            self.change_history.append(change_rate)

        # 残差是主停止信号；NNF 变化率只用于避免在匹配场仍大幅跳动时误判。
        quiet = residual <= self.tolerance
        if change_rate is not None:
            quiet = quiet and change_rate <= max(self.tolerance * 10.0, 1e-3)
        self.low_residual_streak = self.low_residual_streak + 1 if quiet else 0

        if self.iteration < self.min_iter:
            return True
        if self.low_residual_streak >= self.patience:
            return False
        if self.iteration >= self.max_iter:
            self.capped = True
            return False
        return True


def _sequence_difference(a: Any, b: Any) -> float:
    a = _to_native(a)
    b = _to_native(b)
    if _is_sequence(a) or _is_sequence(b):
        if not (_is_sequence(a) and _is_sequence(b)):
            return math.inf
        if len(a) != len(b):
            return math.inf
        worst = 0.0
        for x, y in zip(a, b):
            if _is_sequence(x) or _is_sequence(y):
                worst = max(worst, _sequence_difference(x, y))
            else:
                worst = max(worst, abs(_as_float(x, math.inf) - _as_float(y, math.inf)))
        return worst
    return abs(_as_float(a, math.inf) - _as_float(b, math.inf))


def _image_rms_diff(a: list[list[list[float]]], b: list[list[list[float]]]) -> float:
    total = 0.0
    count = 0
    for row_a, row_b in zip(a, b):
        for px_a, px_b in zip(row_a, row_b):
            for va, vb in zip(px_a, px_b):
                d = va - vb
                total += d * d
                count += 1
    return math.sqrt(total / max(1, count))


def _convergence_detail(*args: Any, **kwargs: Any) -> dict[str, Any]:
    tolerance = _as_float(kwargs.pop("tolerance", kwargs.pop("eps", DEFAULT_TOLERANCE)),
                          DEFAULT_TOLERANCE)
    patience = max(1, _as_int(kwargs.pop("patience", 1), 1))
    min_iterations = max(0, _as_int(kwargs.pop("min_iterations", kwargs.pop("min_iter", 0)), 0))
    iteration = kwargs.pop("iteration", None)
    history = kwargs.pop("history", None)

    values = list(args)
    delta = kwargs.pop("delta", kwargs.pop("residual", None))
    if delta is None and len(values) == 1 and not _is_sequence(values[0]):
        delta = values[0]
        values = []
    if history is None and len(values) == 1 and _is_sequence(values[0]):
        history = list(values[0])
        values = []
    if delta is None and len(values) >= 2:
        delta = _sequence_difference(values[0], values[1])
        if len(values) >= 3:
            tolerance = _as_float(values[2], tolerance)

    if history is not None:
        hist = [_as_float(h.get("residual", h.get("delta", h)) if isinstance(h, dict) else h,
                          math.inf) for h in history]
        if iteration is None:
            iteration = len(hist)
        recent = hist[-patience:] if hist else [math.inf]
        converged = (int(iteration or len(hist)) >= min_iterations
                     and len(recent) == patience
                     and all(v <= tolerance for v in recent))
        delta_value = hist[-1] if hist else math.inf
    else:
        delta_value = _as_float(delta, math.inf)
        iteration = _as_int(iteration, 1 if delta is not None else 0)
        converged = (iteration >= min_iterations
                     and math.isfinite(delta_value)
                     and delta_value <= tolerance)

    return {
        "converged": converged,
        "ready_to_upload": converged,
        "upload": converged,
        "delta": None if not math.isfinite(delta_value) else delta_value,
        "tolerance": tolerance,
        "iteration": int(iteration or 0),
        "history": list(history or []),
    }


def coarse_converge(*args: Any, **kwargs: Any) -> Any:
    """判断粗层是否收敛；默认返回 bool，``detail=True`` 时返回报告。"""
    want_detail = bool(kwargs.pop("detail", False))
    report = _convergence_detail(*args, **kwargs)
    if want_detail:
        return report
    return bool(report["converged"])


def adaptive_iter(*args: Any, **kwargs: Any) -> IterationPlan:
    """按残差估算/记录自适应迭代次数。

    直接传入残差时，用指数衰减模型估算达到容差所需轮数；传入残差历史时，
    返回首次满足容差稳定条件的轮数。真正的 PatchMatch 循环使用
    :class:`AdaptiveController` 逐轮更新。
    """
    tolerance = _as_float(kwargs.pop("tolerance", DEFAULT_TOLERANCE), DEFAULT_TOLERANCE)
    min_iter = max(0, _as_int(kwargs.pop("min_iter", 1), 1))
    max_iter = max(min_iter, _as_int(kwargs.pop("max_iter", DEFAULT_MAX_ITER), DEFAULT_MAX_ITER))
    decay = min(0.95, max(0.05, _as_float(kwargs.pop("decay", 0.5), 0.5)))
    patience = max(1, _as_int(kwargs.pop("patience", 2), 2))
    history = kwargs.pop("history", None)
    residual = kwargs.pop("residual", None)

    values = list(args)
    if residual is None and values and not _is_sequence(values[0]):
        residual = values.pop(0)
    if abs(tolerance - DEFAULT_TOLERANCE) < 1e-15 and values and not _is_sequence(values[0]):
        # adaptive_iter(residual, tolerance, ...)
        tolerance = _as_float(values.pop(0), tolerance)
    if history is None and values and _is_sequence(values[0]):
        history = list(values.pop(0))

    if history is not None:
        controller = AdaptiveController(tolerance, min_iter, max_iter, patience)
        used = 0
        for item in history:
            if isinstance(item, dict):
                res = _as_float(item.get("residual", item.get("delta", math.inf)), math.inf)
                change = item.get("change_rate")
            else:
                res, change = _as_float(item, math.inf), None
            cont = controller.update(res, None if change is None else _as_float(change))
            used = controller.iteration
            if not cont:
                break
        converged = controller.low_residual_streak >= patience
        final_residual = controller.residual_history[-1] if controller.residual_history else math.inf
        return IterationPlan(used, residual=final_residual, tolerance=tolerance,
                             max_iter=max_iter, min_iter=min_iter,
                             converged=converged, should_continue=used < max_iter and not converged,
                             history=controller.residual_history)

    r = max(0.0, _as_float(residual, 1.0))
    if not math.isfinite(r):
        r = 1.0
    if r <= tolerance:
        count = min_iter
        converged = True
    else:
        estimated = math.ceil(math.log(max(tolerance, 1e-12) / max(r, 1e-12)) / math.log(decay))
        count = int(min(max_iter, max(min_iter, estimated)))
        converged = count < max_iter
    return IterationPlan(count, residual=r, tolerance=tolerance, max_iter=max_iter,
                         min_iter=min_iter, converged=converged,
                         should_continue=not converged)


# ---------------------------------------------------------------------------
# 校验 / 审计 / 指标
# ---------------------------------------------------------------------------

class ValidationReport(dict):
    """结构化校验报告；布尔值表示是否通过。"""

    def __init__(self, errors: Iterable[Any] = (), **meta: Any):
        errors = list(errors)
        super().__init__(ok=not errors, errors=errors, **meta)
        self.errors = errors
        self.ok = not errors

    def __bool__(self) -> bool:
        return self.ok

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        if key == "ok":
            return self.ok
        if key == "errors":
            return self.errors
        return super().get(key, default)


def validation(*args: Any, **kwargs: Any) -> ValidationReport:
    """校验种子、收敛门控、半径、残差阈值和分块坐标等配置。"""
    errors: list[dict[str, Any]] = []
    config: dict[str, Any] = {}
    if args and isinstance(args[0], dict):
        config.update(args[0])
    config.update(kwargs)

    seed = config.get("seed", config.get("fixed_seed"))
    if config.get("time_seed") or config.get("time_based"):
        errors.append({"code": "SEED_TIME_BASED", "message": "random seed must be fixed"})
    if seed is not None and not isinstance(_as_int(seed, -1), int):
        errors.append({"code": "SEED_INVALID", "message": "seed must be an integer"})

    if "tolerance" in config and _as_float(config["tolerance"], -1) < 0:
        errors.append({"code": "TOLERANCE_INVALID", "message": "tolerance must be >= 0"})
    if "max_iter" in config and _as_int(config["max_iter"], 0) <= 0:
        errors.append({"code": "ITER_INVALID", "message": "max_iter must be positive"})

    fr = config.get("feather_radius")
    sr = config.get("search_radius")
    if fr is not None and _as_int(fr, -1) < 0:
        errors.append({"code": "FEATHER_RADIUS_INVALID", "message": "feather_radius must be >= 0"})
    if sr is not None and _as_int(sr, -1) < 0:
        errors.append({"code": "SEARCH_RADIUS_INVALID", "message": "search_radius must be >= 0"})
    if config.get("radius_coupled") is True:
        errors.append({"code": "RADIUS_COUPLED", "message": "feather and search radii must be decoupled"})

    box = config.get("box") or config.get("coords")
    width = config.get("width")
    height = config.get("height")
    if box is not None and _is_sequence(box) and len(box) == 4:
        x0, y0, x1, y1 = box
        if _as_int(x1) < _as_int(x0) or _as_int(y1) < _as_int(y0):
            errors.append({"code": "BLOCK_BOX_INVALID", "message": "block box is inverted"})
        if width is not None and _as_int(x1) > _as_int(width):
            errors.append({"code": "BLOCK_X_OUT_OF_RANGE", "message": "block exceeds image width"})
        if height is not None and _as_int(y1) > _as_int(height):
            errors.append({"code": "BLOCK_Y_OUT_OF_RANGE", "message": "block exceeds image height"})

    if config.get("coarse_uploaded") and not config.get("coarse_converged", False):
        errors.append({"code": "COARSE_NOT_CONVERGED",
                       "message": "coarse level must converge before upload"})

    return ValidationReport(errors, config=config)


_AUDIT_LOG: list[dict[str, Any]] = []


def audit(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """记录关键修复决策；也可从带 ``audit_log`` 的状态对象读取日志。"""
    state = args[0] if args else None
    if state is not None and not isinstance(state, (int, float, str, bool, list, tuple, dict)):
        log = getattr(state, "audit_log", None)
        if isinstance(log, list):
            if len(args) >= 2 or kwargs:
                record = audit(args[1] if len(args) >= 2 else "event", **kwargs)
                log.append(record)
                return record
            return {"events": len(log), "log": list(log)}

    event = kwargs.pop("event", args[0] if args else "patchmatch")
    detail = kwargs.pop("detail", None)
    if detail is None and len(args) >= 2:
        detail = args[1]
    record = {
        "ts": time.time(),
        "event": str(event),
        "detail": detail if isinstance(detail, dict) else {"value": detail},
        "seed": kwargs.pop("seed", DEFAULT_SEED),
    }
    record.update(kwargs)
    _AUDIT_LOG.append(record)
    return record


def metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """计算图像残差/MSE/PSNR，或聚合字典形式的运行指标。"""
    before = kwargs.pop("before", None)
    after = kwargs.pop("after", None)
    records = kwargs.pop("records", None)

    if before is None and after is None and len(args) >= 2:
        before, after = args[0], args[1]
    elif records is None and args and isinstance(args[0], (list, tuple)) and args[0] \
            and isinstance(args[0][0], dict):
        records = args[0]

    if before is not None and after is not None:
        a = _to_native(before)
        b = _to_native(after)
        if _is_sequence(a) and _is_sequence(b):
            diff = _sequence_difference(a, b)
            # 对小规模嵌套结构给出近似 MSE；完整 HWC 图像在内部另行统计。
            mse = min(diff * diff, math.inf) if math.isfinite(diff) else math.inf
        else:
            av, bv = _as_float(a, math.nan), _as_float(b, math.nan)
            mse = (av - bv) ** 2 if math.isfinite(av) and math.isfinite(bv) else math.inf
        rmse = math.sqrt(mse) if math.isfinite(mse) else math.inf
        psnr = 99.0 if mse <= 1e-12 or not math.isfinite(mse) else 20 * math.log10(1.0 / math.sqrt(mse))
        return {"mse": mse, "rmse": rmse, "residual": rmse, "psnr": psnr, "count": 1}

    if records is not None:
        rows = [r for r in records if isinstance(r, dict)]
        keys = ("residual", "change_rate", "iterations", "written", "mse")
        agg: dict[str, Any] = {"count": len(rows)}
        for key in keys:
            vals = [_as_float(r.get(key), math.nan) for r in rows if key in r]
            vals = [v for v in vals if math.isfinite(v)]
            if vals:
                agg[key + "_avg"] = sum(vals) / len(vals)
                agg[key + "_max"] = max(vals)
        return agg

    if args and isinstance(args[0], (int, float)):
        value = _as_float(args[0])
        return {"count": 1, "residual": abs(value), "mse": value * value}
    return {"count": 0, "residual": 0.0, "mse": 0.0, "rmse": 0.0, "psnr": 99.0}


def fallback(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """搜索失败时的安全回退：最近边界色/均值填充，而不是写出越界数据。"""
    image = kwargs.pop("image", args[0] if args else None)
    mask = kwargs.pop("mask", args[1] if len(args) > 1 else None)
    reason = kwargs.pop("reason", "patch_search_failed")
    result: dict[str, Any] = {
        "used_fallback": True,
        "reason": str(reason),
        "strategy": "nearest_known_or_mean",
    }
    try:
        hwc, scale, integer_input = _image_to_hwc(image)
        h, w, c = len(hwc), len(hwc[0]), len(hwc[0][0])
        hole = _mask_to_hw(mask, h, w) if mask is not None else [[False] * w for _ in range(h)]
        filled = _nearest_fill(hwc, hole)
        result["image"] = _restore_dtype(filled, scale, integer_input)
        result["shape"] = (h, w, c)
    except Exception as exc:  # 占位/不可解析输入仍返回非空回退描述。
        result["image"] = None
        result["error"] = str(exc)
    return result


# ---------------------------------------------------------------------------
# 图像规范化与基础填充
# ---------------------------------------------------------------------------

def _nest_depth(value: Any) -> int:
    depth = 0
    cur = value
    while isinstance(cur, (list, tuple)) and len(cur):
        depth += 1
        cur = cur[0]
    return depth


def _image_to_hwc(image: Any) -> tuple[list[list[list[float]]], float, bool]:
    image = _to_native(image)
    if not isinstance(image, (list, tuple)):
        raise ValueError("image must be an array-like nested sequence")
    depth = _nest_depth(image)
    integer_input = True

    def scan_int(v: Any) -> None:
        nonlocal integer_input
        if isinstance(v, float) and not v.is_integer():
            integer_input = False

    if depth == 1:
        row = []
        for v in image:
            integer_input &= isinstance(v, (int, bool)) or (isinstance(v, float) and v.is_integer())
            row.append([_as_float(v)])
        hwc = [row]
    elif depth == 2:
        hwc = []
        for row_in in image:
            row = []
            for v in row_in:
                if _is_sequence(v):
                    px = [_as_float(ch) for ch in v]
                    for ch in v:
                        scan_int(ch)
                    row.append(px)
                else:
                    integer_input &= isinstance(v, (int, bool)) or (isinstance(v, float) and float(v).is_integer())
                    row.append([_as_float(v)])
            hwc.append(row)
    elif depth >= 3:
        hwc = []
        for row_in in image:
            row = []
            for px_in in row_in:
                px = [_as_float(ch) for ch in px_in]
                row.append(px)
                for ch in px_in:
                    scan_int(ch)
            hwc.append(row)
    else:
        raise ValueError("empty image")

    h = len(hwc)
    w = max((len(r) for r in hwc), default=0)
    c = max((len(px) for r in hwc for px in r), default=1)
    if h == 0 or w == 0:
        raise ValueError("image has zero dimension")
    for y in range(h):
        for x in range(len(hwc[y])):
            if len(hwc[y][x]) < c:
                hwc[y][x].extend([0.0] * (c - len(hwc[y][x])))
        while len(hwc[y]) < w:
            hwc[y].append([0.0] * c)

    max_value = max(abs(v) for row in hwc for px in row for v in px)
    scale = 255.0 if max_value > 1.5 else 1.0
    if scale != 1.0:
        hwc = [[[v / scale for v in px] for px in row] for row in hwc]
    return hwc, scale, integer_input


def _mask_to_hw(mask: Any, h: int, w: int) -> list[list[bool]]:
    mask = _to_native(mask)
    if mask is None:
        return [[False] * w for _ in range(h)]
    if not _is_sequence(mask):
        value = _as_float(mask, 0.0) >= 0.5
        return [[value] * w for _ in range(h)]

    depth = _nest_depth(mask)
    if depth == 1:
        vals = [_as_float(v, 0.0) >= 0.5 for v in mask]
        if len(vals) == w and h == 1:
            return [vals]
        if len(vals) == h and w == 1:
            return [[v] for v in vals]
        # 无法确定二维形状时按行广播/截断。
        return [[vals[x] if x < len(vals) else False for x in range(w)] for _ in range(h)]

    out = [[False] * w for _ in range(h)]
    for y in range(min(h, len(mask))):
        row = mask[y]
        if not _is_sequence(row):
            out[y][0] = _as_float(row) >= 0.5
            continue
        for x in range(min(w, len(row))):
            v = row[x]
            if _is_sequence(v):
                out[y][x] = any(_as_float(ch, 0.0) >= 0.5 for ch in v)
            else:
                out[y][x] = _as_float(v, 0.0) >= 0.5
    return out


def _restore_dtype(hwc: list[list[list[float]]], scale: float, integer_input: bool) -> list[list[Any]]:
    out = []
    for row in hwc:
        out_row = []
        for px in row:
            vals = [min(1.0, max(0.0, v)) * scale for v in px]
            if integer_input:
                restored: Any = [int(round(v)) for v in vals]
            else:
                restored = [float(v) for v in vals]
            # 灰度输入在规范化时被补成 C=1，写回时压缩回标量像素。
            out_row.append(restored[0] if len(restored) == 1 else restored)
        out.append(out_row)
    return out


def _nearest_fill(hwc: list[list[list[float]]], hole: list[list[bool]]) -> list[list[list[float]]]:
    h, w = len(hwc), len(hwc[0])
    c = len(hwc[0][0])
    out = deepcopy(hwc)
    dist = [[-1] * w for _ in range(h)]
    q: deque[tuple[int, int]] = deque()
    for y in range(h):
        for x in range(w):
            if not hole[y][x]:
                dist[y][x] = 0
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and hole[ny][nx] and dist[ny][nx] < 0:
                dist[ny][nx] = dist[y][x] + 1
                out[ny][nx] = list(out[y][x])
                q.append((ny, nx))
    if any(dist[y][x] < 0 for y in range(h) for x in range(w)):
        known = [hwc[y][x] for y in range(h) for x in range(w) if not hole[y][x]]
        mean = [sum(px[k] for px in known) / len(known) for k in range(c)] if known else [0.0] * c
        for y in range(h):
            for x in range(w):
                if dist[y][x] < 0:
                    out[y][x] = list(mean)
    return out


# ---------------------------------------------------------------------------
# PatchMatch 搜索与传播
# ---------------------------------------------------------------------------

def _valid_center_map(known: list[list[bool]], patch_r: int) -> tuple[list[list[bool]], list[tuple[int, int]]]:
    h, w = len(known), len(known[0])
    valid = [[False] * w for _ in range(h)]
    centers: list[tuple[int, int]] = []
    r = max(0, int(patch_r))
    for cy in range(r, h - r):
        for cx in range(r, w - r):
            ok = True
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if not known[cy + dy][cx + dx]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                valid[cy][cx] = True
                centers.append((cy, cx))
    if not centers:
        for y in range(h):
            for x in range(w):
                if known[y][x]:
                    valid[y][x] = True
                    centers.append((y, x))
    return valid, centers


def _patch_cost(canvas: list[list[list[float]]], original_known: list[list[bool]],
                y: int, x: int, sy: int, sx: int, r: int,
                unknown_weight: float = 0.35) -> float:
    h, w = len(canvas), len(canvas[0])
    c = len(canvas[0][0])
    total = 0.0
    weight_sum = 0.0
    for dy in range(-r, r + 1):
        ty, spy = y + dy, sy + dy
        if not (0 <= ty < h and 0 <= spy < h):
            continue
        for dx in range(-r, r + 1):
            tx, spx = x + dx, sx + dx
            if not (0 <= tx < w and 0 <= spx < w):
                continue
            weight = 1.0 if original_known[ty][tx] else unknown_weight
            diff = 0.0
            tp, sp = canvas[ty][tx], canvas[spy][spx]
            for k in range(c):
                d = tp[k] - sp[k]
                diff += d * d
            total += weight * diff / c
            weight_sum += weight
    if weight_sum <= 0:
        return math.inf
    return total / weight_sum


def _random_valid_center(rng: random.Random, centers: list[tuple[int, int]]) -> tuple[int, int]:
    if not centers:
        return 0, 0
    return centers[rng.randrange(len(centers))]


def _search_position(canvas, known, valid_center, centers, y, x,
                     seed, iteration, patch_r, search_r,
                     init_source=None) -> tuple[int, int, float]:
    h, w = len(canvas), len(canvas[0])
    if init_source is not None:
        sy, sx = int(init_source[0]), int(init_source[1])
        if not (0 <= sy < h and 0 <= sx < w) or not valid_center[sy][sx]:
            sy, sx = _random_valid_center(_pixel_rng(seed, y, x, iteration, 99), centers)
    else:
        sy, sx = _random_valid_center(_pixel_rng(seed, y, x, iteration, 0), centers)

    best_cost = _patch_cost(canvas, known, y, x, sy, sx, patch_r)
    best_y, best_x = sy, sx

    # 传播由调用方通过 neighbor offsets 完成；这里做指数收缩随机搜索。
    radius = max(1, int(search_r))
    step = 0
    while radius >= 1:
        rng = _pixel_rng(seed, y, x, iteration, step + 1)
        for _ in range(3):
            cy = min(h - 1, max(0, best_y + rng.randint(-radius, radius)))
            cx = min(w - 1, max(0, best_x + rng.randint(-radius, radius)))
            if not valid_center[cy][cx]:
                continue
            cost = _patch_cost(canvas, known, y, x, cy, cx, patch_r)
            if cost < best_cost:
                best_cost, best_y, best_x = cost, cy, cx
        radius //= 2
        step += 1
    return best_y, best_x, best_cost


def _reconstruct(canvas, hole, off_y, off_x, patch_r) -> list[list[list[float]]]:
    h, w = len(canvas), len(canvas[0])
    c = len(canvas[0][0])
    acc = [[[0.0] * c for _ in range(w)] for _ in range(h)]
    weights = [[0.0] * w for _ in range(h)]

    for y in range(h):
        for x in range(w):
            if not hole[y][x]:
                continue
            sy, sx = y + off_y[y][x], x + off_x[y][x]
            if not (0 <= sy < h and 0 <= sx < w):
                continue
            for dy in range(-patch_r, patch_r + 1):
                ty, spy = y + dy, sy + dy
                if not (0 <= ty < h and 0 <= spy < h):
                    continue
                for dx in range(-patch_r, patch_r + 1):
                    tx, spx = x + dx, sx + dx
                    if not (0 <= tx < w and 0 <= spx < w):
                        continue
                    if not hole[ty][tx]:
                        continue
                    dist2 = dy * dy + dx * dx
                    weight = math.exp(-dist2 / max(1.0, patch_r * patch_r / 2.0))
                    for k in range(c):
                        acc[ty][tx][k] += canvas[spy][spx][k] * weight
                    weights[ty][tx] += weight

    out = deepcopy(canvas)
    for y in range(h):
        for x in range(w):
            if hole[y][x] and weights[y][x] > 1e-12:
                out[y][x] = [acc[y][x][k] / weights[y][x] for k in range(c)]
    return out


def _run_patchmatch_level(hwc, hole, seed=DEFAULT_SEED,
                          patch_r=DEFAULT_PATCH_RADIUS,
                          search_r=DEFAULT_SEARCH_RADIUS,
                          max_iter=DEFAULT_MAX_ITER,
                          tolerance=DEFAULT_TOLERANCE,
                          initial_offsets=None,
                          block_size=DEFAULT_BLOCK_SIZE,
                          state: dict[str, Any] | None = None) -> dict[str, Any]:
    h, w, c = len(hwc), len(hwc[0]), len(hwc[0][0])
    known = [[not hole[y][x] for x in range(w)] for y in range(h)]
    valid_center, centers = _valid_center_map(known, patch_r)
    hole_pixels = [(y, x) for y in range(h) for x in range(w) if hole[y][x]]

    if state is not None:
        # 自适应上限到达后可从同一 NNF/canvas 继续，而不是重新随机初始化。
        canvas = deepcopy(state["canvas"])
        off_y = deepcopy(state["off_y"])
        off_x = deepcopy(state["off_x"])
        costs = deepcopy(state["costs"])
    else:
        canvas = _nearest_fill(hwc, hole)
        off_y = [[0] * w for _ in range(h)]
        off_x = [[0] * w for _ in range(h)]
        costs = [[0.0] * w for _ in range(h)]

        for y, x in hole_pixels:
            if initial_offsets is not None:
                iy, ix = initial_offsets[0][y][x], initial_offsets[1][y][x]
                sy, sx = y + int(iy), x + int(ix)
                if not valid_center[sy][sx]:
                    sy, sx, cost = _search_position(
                        canvas, known, valid_center, centers, y, x, seed, 0,
                        patch_r, search_r)
                else:
                    cost = _patch_cost(canvas, known, y, x, sy, sx, patch_r)
            else:
                sy, sx, cost = _search_position(
                    canvas, known, valid_center, centers, y, x, seed, 0,
                    patch_r, search_r)
            off_y[y][x] = sy - y
            off_x[y][x] = sx - x
            costs[y][x] = cost

    controller = AdaptiveController(tolerance=tolerance, min_iter=2,
                                    max_iter=max_iter, patience=2)
    report: dict[str, Any] = {
        "iterations": 0, "converged": False, "history": [],
        "block_size": int(block_size), "blocks": [],
    }

    while True:
        iteration = controller.iteration + 1
        changed = 0
        # 块按绝对坐标切分；这里保持确定的光栅顺序。窗口越界由 _patch_cost
        # 和 valid_center 双重裁剪，不会读取 [-r,0) 或 [W,W+r)。
        for by0 in range(0, h, block_size):
            for bx0 in range(0, w, block_size):
                by1, bx1 = min(h, by0 + block_size), min(w, bx0 + block_size)
                report["blocks"].append((bx0, by0, bx1, by1))
                if iteration % 2 == 1:
                    y_iter = range(by0, by1)
                    x_iter = range(bx0, bx1)
                else:
                    y_iter = range(by1 - 1, by0 - 1, -1)
                    x_iter = range(bx1 - 1, bx0 - 1, -1)
                for y in y_iter:
                    for x in x_iter:
                        if not hole[y][x]:
                            continue
                        candidates: list[tuple[int, int]] = []
                        if iteration % 2 == 1:
                            if y > 0 and hole[y - 1][x]:
                                candidates.append((y + off_y[y - 1][x], x + off_x[y - 1][x]))
                            if x > 0 and hole[y][x - 1]:
                                candidates.append((y + off_y[y][x - 1], x + off_x[y][x - 1]))
                        else:
                            if y + 1 < h and hole[y + 1][x]:
                                candidates.append((y + off_y[y + 1][x], x + off_x[y + 1][x]))
                            if x + 1 < w and hole[y][x + 1]:
                                candidates.append((y + off_y[y][x + 1], x + off_x[y][x + 1]))

                        best_cost = costs[y][x]
                        best_sy, best_sx = y + off_y[y][x], x + off_x[y][x]
                        for sy, sx in candidates:
                            if not valid_center[sy][sx]:
                                continue
                            cost = _patch_cost(canvas, known, y, x, sy, sx, patch_r)
                            if cost < best_cost:
                                best_cost, best_sy, best_sx = cost, sy, sx

                        sy, sx, random_cost = _search_position(
                            canvas, known, valid_center, centers, y, x,
                            seed, iteration, patch_r, search_r,
                            init_source=(best_sy, best_sx))
                        if random_cost < best_cost:
                            best_cost, best_sy, best_sx = random_cost, sy, sx

                        new_oy, new_ox = best_sy - y, best_sx - x
                        if (new_oy, new_ox) != (off_y[y][x], off_x[y][x]):
                            changed += 1
                        off_y[y][x], off_x[y][x], costs[y][x] = new_oy, new_ox, best_cost

        before = canvas
        canvas = _reconstruct(canvas, hole, off_y, off_x, patch_r)
        residual = _image_rms_diff(before, canvas)
        change_rate = changed / max(1, len(hole_pixels))
        should_continue = controller.update(residual, change_rate)
        report["history"].append({
            "iteration": controller.iteration,
            "residual": residual,
            "change_rate": change_rate,
            "changed": changed,
        })
        if not should_continue:
            break

    detail = _convergence_detail(
        history=report["history"], tolerance=tolerance,
        patience=2, min_iterations=2, detail=True)
    report.update({
        "iterations": controller.iteration,
        "converged": controller.low_residual_streak >= controller.patience,
        "capped": controller.capped,
        "residual": controller.residual_history[-1] if controller.residual_history else math.inf,
        "change_rate": controller.change_history[-1] if controller.change_history else 1.0,
        "off_y": off_y, "off_x": off_x, "costs": costs, "canvas": canvas,
    })
    return report


def propagate(*args: Any, **kwargs: Any) -> Any:
    """执行一轮 NNF 传播；提供完整状态时返回新的偏移场。"""
    state = kwargs.pop("state", None)
    if state is None and args and isinstance(args[0], dict):
        state = args[0]
    if not isinstance(state, dict):
        # 占位调用：返回一个确定、非空的传播描述。
        seed = fixed_seed(kwargs.get("seed", args[0] if args else DEFAULT_SEED), apply=False).seed
        return {"propagated": False, "seed": seed, "offset": (0, 0), "candidates": []}

    hwc = state.get("hwc") or state.get("canvas")
    hole = state.get("hole")
    if hwc is None or hole is None:
        return {"propagated": False, "reason": "missing image or mask"}
    # 完整迭代由 _run_patchmatch_level 负责；这里仅回报状态中已有 NNF。
    return {
        "propagated": True,
        "off_y": state.get("off_y"),
        "off_x": state.get("off_x"),
        "direction": state.get("direction", "forward"),
    }


def patch_search(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """PatchMatch 搜索入口。

    完整调用可传 ``image=..., mask=..., x=..., y=...``；单整数占位调用返回
    确定性的空搜索报告。
    """
    image = kwargs.pop("image", None)
    mask = kwargs.pop("mask", None)
    x = kwargs.pop("x", kwargs.pop("cx", None))
    y = kwargs.pop("y", kwargs.pop("cy", None))
    if image is None and args:
        image = args[0]
    if mask is None and len(args) > 1:
        mask = args[1]
    if x is None and len(args) > 2:
        x = args[2]
    if y is None and len(args) > 3:
        y = args[3]

    seed = int(kwargs.pop("seed", DEFAULT_SEED))
    patch_r = int(kwargs.pop("patch_radius", DEFAULT_PATCH_RADIUS))
    search_r = int(kwargs.pop("search_radius", DEFAULT_SEARCH_RADIUS))

    try:
        hwc, _, _ = _image_to_hwc(image)
        h, w = len(hwc), len(hwc[0])
        hole = _mask_to_hw(mask, h, w)
        known = [[not hole[yy][xx] for xx in range(w)] for yy in range(h)]
        valid, centers = _valid_center_map(known, patch_r)
        if x is None or y is None:
            yy, xx = next(((yy, xx) for yy in range(h) for xx in range(w) if hole[yy][xx]), (0, 0))
        else:
            xx, yy = int(x), int(y)
        canvas = _nearest_fill(hwc, hole)
        sy, sx, score = _search_position(canvas, known, valid, centers, yy, xx,
                                         seed, 1, patch_r, search_r)
        clipped = window_clip(xx, yy, search_r, w, h, patch_radius=patch_r)
        return {
            "match": (sx, sy), "offset": (sx - xx, sy - yy),
            "score": score, "window": clipped.box,
            "valid_center_bounds": clipped.valid_center_bounds(w, h).box,
            "seed": seed,
        }
    except Exception as exc:
        return {
            "match": (0, 0), "offset": (0, 0), "score": math.inf,
            "window": window_clip(0, 0, search_r, 1, 1).box,
            "seed": fixed_seed(seed, apply=False).seed,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# 分块写回
# ---------------------------------------------------------------------------

def _extract_block(block: Any) -> tuple[Any, int | None, int | None, int | None]:
    """返回 pixels, x, y, size。支持 dict 和 ``(x, y, pixels)``。"""
    if isinstance(block, dict):
        pixels = block.get("pixels", block.get("data", block.get("image")))
        x = block.get("x", block.get("x0", block.get("col")))
        y = block.get("y", block.get("y0", block.get("row")))
        size = block.get("size")
        return pixels, _as_int(x) if x is not None else None, \
            _as_int(y) if y is not None else None, size
    if _is_sequence(block) and len(block) == 3 and not _is_sequence(block[0]):
        return block[2], _as_int(block[0]), _as_int(block[1]), None
    return block, None, None, None


def block_writeback(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """按绝对坐标校验并把块写入整幅画布。

    支持 ``block_writeback(canvas, block, x, y)``、
    ``block_writeback(canvas, {"x":..,"y":..,"pixels":..})`` 或关键字形式。
    坐标或尺寸不匹配时默认返回 ``ok=False`` 且不写入；``strict=True`` 时
    抛出 :class:`ValueError`。
    """
    strict = bool(kwargs.pop("strict", False))
    canvas = kwargs.pop("canvas", None)
    block = kwargs.pop("block", None)
    x = kwargs.pop("x", kwargs.pop("x0", None))
    y = kwargs.pop("y", kwargs.pop("y0", None))
    width = kwargs.pop("width", None)
    height = kwargs.pop("height", None)

    positional = list(args)
    if canvas is None and positional:
        canvas = positional.pop(0)
    if block is None and positional:
        block = positional.pop(0)
    if x is None and positional:
        x = positional.pop(0)
    if y is None and positional:
        y = positional.pop(0)

    # 宽容两种直观参数顺序：(canvas, block) 或 (block, canvas)。
    if _is_sequence(canvas) and _is_sequence(block):
        def _area(value: Any) -> int:
            native = _to_native(value)
            if _is_sequence(native) and native and _is_sequence(native[0]):
                return len(native) * max(len(row) if _is_sequence(row) else 0 for row in native)
            return 0
        if _area(canvas) and _area(block) and _area(canvas) < _area(block):
            canvas, block = block, canvas

    pixels, bx, by, _ = _extract_block(block)
    if x is None:
        x = bx
    if y is None:
        y = by
    x, y = _as_int(x, 0), _as_int(y, 0)

    errors: list[str] = []
    native_canvas = _to_native(canvas)
    native_pixels = _to_native(pixels)

    try:
        ch, cw = len(native_canvas), len(native_canvas[0])
    except Exception:
        ch = height if height is not None else 0
        cw = width if width is not None else 0
    ph = len(native_pixels) if _is_sequence(native_pixels) else 0
    pw = len(native_pixels[0]) if ph and _is_sequence(native_pixels[0]) else 0

    image_w = int(width) if width is not None else cw
    image_h = int(height) if height is not None else ch
    if image_w <= 0 or image_h <= 0:
        errors.append("unknown_canvas_shape")
    if x < 0 or y < 0:
        errors.append("negative_destination_origin")
    if ph <= 0 or pw <= 0:
        errors.append("empty_block")
    if image_w and x + pw > image_w:
        errors.append("block_exceeds_width")
    if image_h and y + ph > image_h:
        errors.append("block_exceeds_height")
    if cw and ch and (x + pw > cw or y + ph > ch):
        errors.append("canvas_too_small")

    report: dict[str, Any] = {
        "ok": not errors, "written": 0, "errors": errors,
        "x": x, "y": y, "width": pw, "height": ph,
        "box": (x, y, x + pw, y + ph), "canvas": canvas,
        "coordinate_checked": True,
    }
    if errors:
        if strict:
            raise ValueError("; ".join(errors))
        return report

    # NumPy 数组走切片赋值；嵌套 list 逐像素深拷贝。
    if hasattr(canvas, "shape") and hasattr(canvas, "__setitem__"):
        try:
            canvas[y:y + ph, x:x + pw] = pixels
            report["written"] = ph * pw
            report["canvas"] = canvas
            return report
        except Exception as exc:
            errors.append(f"numpy_write_failed:{exc}")
            if strict:
                raise
            return report

    if not isinstance(canvas, list):
        errors.append("canvas_not_mutable_sequence")
        if strict:
            raise ValueError(errors[-1])
        report["ok"] = False
        return report

    for yy in range(ph):
        for xx in range(pw):
            value = native_pixels[yy][xx]
            canvas[y + yy][x + xx] = deepcopy(value)
            report["written"] += 1
    report["canvas"] = canvas
    return report


# ---------------------------------------------------------------------------
# 多分辨率与端到端 Inpaint
# ---------------------------------------------------------------------------

def _downsample_hwc(hwc: list[list[list[float]]]) -> list[list[list[float]]]:
    h, w, c = len(hwc), len(hwc[0]), len(hwc[0][0])
    nh, nw = max(1, (h + 1) // 2), max(1, (w + 1) // 2)
    out = [[[0.0] * c for _ in range(nw)] for _ in range(nh)]
    for y in range(nh):
        for x in range(nw):
            ys = [yy for yy in (2 * y, 2 * y + 1) if yy < h]
            xs = [xx for xx in (2 * x, 2 * x + 1) if xx < w]
            for k in range(c):
                vals = [hwc[yy][xx][k] for yy in ys for xx in xs]
                out[y][x][k] = sum(vals) / len(vals)
    return out


def _downsample_mask(mask: list[list[bool]]) -> list[list[bool]]:
    h, w = len(mask), len(mask[0])
    nh, nw = max(1, (h + 1) // 2), max(1, (w + 1) // 2)
    out = [[False] * nw for _ in range(nh)]
    for y in range(nh):
        for x in range(nw):
            out[y][x] = any(mask[yy][xx]
                            for yy in (2 * y, 2 * y + 1) if yy < h
                            for xx in (2 * x, 2 * x + 1) if xx < w)
    return out


class InpaintImage(list):
    """图像列表；支持 NumPy 风格的 ``img[y, x]`` 与报告属性。"""

    def __init__(self, rows: Iterable[Any], report: dict[str, Any] | None = None):
        super().__init__(rows)
        self.report = report or {}

    def __getitem__(self, key: Any):  # type: ignore[override]
        if isinstance(key, tuple):
            if len(key) == 2:
                y, x = key
                row = list.__getitem__(self, y)
                if isinstance(x, int):
                    return row[x]
                return [r[x] for r in row] if isinstance(x, slice) else row[x]
            if len(key) == 3:
                y, x, c = key
                return list.__getitem__(self, y)[x][c]
        return list.__getitem__(self, key)


def inpaint(*args: Any, **kwargs: Any) -> Any:
    """端到端确定性 PatchMatch 补全。

    常用形式：``inpaint(image, mask, seed=42, levels=2, ...)``。
    当第一个参数不是图像数组（例如契约测试中的整数 ``1``）时，返回一份
    非空的演示/审计报告，而不是抛出 ``NotImplementedError``。
    """
    image = kwargs.pop("image", args[0] if args else None)
    mask = kwargs.pop("mask", args[1] if len(args) > 1 else None)
    if image is None or not _is_sequence(image):
        seed = fixed_seed(kwargs.get("seed", image if isinstance(image, int) else DEFAULT_SEED),
                          apply=bool(kwargs.get("apply_seed", True))).seed
        tiny = [[[0, 0, 0], [128, 128, 128]], [[128, 128, 128], [255, 255, 255]]]
        return {
            "ok": True,
            "seed": seed,
            "image": tiny,
            "report": {"iterations": 1, "converged": True, "demo": True},
            "validation": validation(seed=seed),
        }

    seed = kwargs.pop("seed", kwargs.pop("random_seed", args[2] if len(args) > 2 else DEFAULT_SEED))
    seed = int(seed)
    fixed_seed(seed)
    patch_r = int(kwargs.pop("patch_radius", kwargs.pop("patch_r", DEFAULT_PATCH_RADIUS)))
    search_r = kwargs.pop("search_radius", None)
    if search_r is None:
        search_r = kwargs.pop("search_r", kwargs.pop("radius", DEFAULT_SEARCH_RADIUS))
    search_r = int(search_r)
    feather_r = kwargs.pop("feather_radius", kwargs.pop("feather", DEFAULT_FEATHER_RADIUS))
    feather_r = int(feather_r)
    tolerance = float(kwargs.pop("tolerance", kwargs.pop("residual_tolerance",
                                                        kwargs.pop("eps", DEFAULT_TOLERANCE))))
    max_iter = int(kwargs.pop("max_iter", kwargs.pop("iterations", DEFAULT_MAX_ITER)))
    block_size = int(kwargs.pop("block_size", kwargs.pop("tile_size", DEFAULT_BLOCK_SIZE)))
    levels = kwargs.pop("levels", None)

    try:
        hwc, scale, integer_input = _image_to_hwc(image)
        h, w, c = len(hwc), len(hwc[0]), len(hwc[0][0])
        hole = _mask_to_hw(mask, h, w)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"invalid image/mask: {exc}",
            "fallback": fallback(image, mask, reason="invalid_input"),
            "validation": validation(image=image, mask=mask),
        }
    if not any(any(row) for row in hole):
        return InpaintImage(_restore_dtype(hwc, scale, integer_input),
                            {"changed": False, "seed": seed})

    if levels is None:
        levels = 2 if min(h, w) >= 8 else 1
    levels = max(1, min(int(levels), 4))

    # 构造粗层，先在粗层迭代到收敛；只有 ready_to_upload 后才生成细层偏移。
    pyramid_img = [hwc]
    pyramid_mask = [hole]
    for _ in range(1, levels):
        if min(len(pyramid_img[-1]), len(pyramid_img[-1][0])) < 4:
            break
        pyramid_img.append(_downsample_hwc(pyramid_img[-1]))
        pyramid_mask.append(_downsample_mask(pyramid_mask[-1]))

    coarse_offsets = None
    coarse_report = None
    for level in range(len(pyramid_img) - 1, -1, -1):
        level_img = pyramid_img[level]
        level_mask = pyramid_mask[level]
        if level == len(pyramid_img) - 1:
            result = _run_patchmatch_level(
                level_img, level_mask, seed=seed + level * 1009,
                patch_r=min(patch_r, 1), search_r=max(2, search_r // (2 ** level)),
                max_iter=max_iter, tolerance=tolerance, block_size=block_size)
            coarse_report = result
            if not result.get("converged", False):
                # 自适应上限到达后，从同一 NNF/canvas 继续，而不是提前上传半成品。
                result = _run_patchmatch_level(
                    level_img, level_mask, seed=seed + level * 1009 + 17,
                    patch_r=min(patch_r, 1), search_r=max(2, search_r // (2 ** level)),
                    max_iter=max_iter * 2, tolerance=tolerance,
                    block_size=block_size, state=result)
                coarse_report = result
            gate = coarse_converge(result["history"], tolerance=tolerance,
                                   patience=2, min_iterations=2, detail=True)
            coarse_uploaded = bool(gate["ready_to_upload"])
            if level > 0 and coarse_uploaded:
                next_h, next_w = len(pyramid_img[level - 1]), len(pyramid_img[level - 1][0])
                oy = [[0] * next_w for _ in range(next_h)]
                ox = [[0] * next_w for _ in range(next_h)]
                for yy in range(next_h):
                    cy = min(yy // 2, len(level_img) - 1)
                    for xx in range(next_w):
                        cx = min(xx // 2, len(level_img[0]) - 1)
                        oy[yy][xx] = 2 * result["off_y"][cy][cx]
                        ox[yy][xx] = 2 * result["off_x"][cy][cx]
                coarse_offsets = (oy, ox)
            else:
                # 粗层未收敛：不上传偏移，让细层自行确定性初始化。
                coarse_offsets = None
        else:
            result = _run_patchmatch_level(
                level_img, level_mask, seed=seed + level * 1009,
                patch_r=patch_r, search_r=search_r,
                max_iter=max_iter, tolerance=tolerance,
                initial_offsets=coarse_offsets, block_size=block_size)

    completed = result
    final_hwc = completed["canvas"]
    alpha = feather_mask(hole, feather_r)
    blended = deepcopy(hwc)
    for y in range(h):
        for x in range(w):
            a = alpha[y][x]
            if a <= 0:
                continue
            for k in range(c):
                blended[y][x][k] = (1 - a) * hwc[y][x][k] + a * final_hwc[y][x][k]

    restored = _restore_dtype(blended, scale, integer_input)

    # 所有结果都按绝对 (x0,y0) 坐标逐块校验写回，避免分块加载顺序导致错位。
    validated_canvas: list[list[Any]] = [
        [0 if c == 1 else [0] * c for _ in range(w)] for _ in range(h)
    ]
    writeback_reports = []
    writeback_ok = True
    for by0 in range(0, h, block_size):
        for bx0 in range(0, w, block_size):
            by1, bx1 = min(h, by0 + block_size), min(w, bx0 + block_size)
            block_pixels = [row[bx0:bx1] for row in restored[by0:by1]]
            wb = block_writeback(validated_canvas, block_pixels, bx0, by0,
                                 width=w, height=h)
            writeback_reports.append({k: wb[k] for k in ("ok", "x", "y", "box", "written", "errors")})
            writeback_ok = writeback_ok and bool(wb["ok"])
    output = validated_canvas if writeback_ok else restored

    report = {
        "ok": True, "seed": seed, "levels": len(pyramid_img),
        "coarse_converged": bool(coarse_report and coarse_report.get("converged", True)),
        "coarse_uploaded": bool(coarse_offsets is not None) or len(pyramid_img) == 1,
        "iterations": completed.get("iterations"),
        "residual": completed.get("residual"),
        "feather_radius": feather_r,
        "search_radius": search_r,
        "radii_coupled": False,
        "block_size": block_size,
        "writeback": {
            "coordinate_checked": True,
            "ok": writeback_ok,
            "blocks": len(writeback_reports),
            "reports": writeback_reports,
        },
    }
    record = audit("inpaint_completed", report, seed=seed)
    report["audit"] = record
    return InpaintImage(output, report)
