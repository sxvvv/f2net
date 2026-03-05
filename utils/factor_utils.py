# utils/factor_utils.py
# 退化因子解析工具

import torch

# ============================================================================
# 因子定义
# ============================================================================

FACTORS = ["low", "haze", "rain", "snow"]

FACTOR2IDX = {
    "low": 0,
    "haze": 1,
    "rain": 2,
    "snow": 3,
}

IDX2FACTOR = {idx: factor for factor, idx in FACTOR2IDX.items()}


# ============================================================================
# 因子解析函数
# ============================================================================

def parse_factors(deg_name):
    """
    从退化名称字符串解析出因子列表
    
    Args:
        deg_name: 退化名称，如 "low_haze_rain" 或 "low"
    
    Returns:
        list: 因子列表，如 ["low", "haze", "rain"]
    """
    if deg_name is None or deg_name == "":
        return []
    
    # 按 "_" 分割
    factors = deg_name.split("_")
    
    # 过滤掉无效因子
    valid_factors = [f for f in factors if f in FACTOR2IDX]
    
    return valid_factors


def factors_to_present(factors):
    """
    将因子列表转换为4维present向量
    
    Args:
        factors: 因子列表，如 ["low", "haze"]
    
    Returns:
        torch.Tensor: (4,) 的present向量，如 [1, 1, 0, 0]
    """
    present = torch.zeros(4, dtype=torch.float32)
    
    for factor in factors:
        if factor in FACTOR2IDX:
            idx = FACTOR2IDX[factor]
            present[idx] = 1.0
    
    return present


def build_name(factors):
    """
    从因子列表构建退化名称
    
    Args:
        factors: 因子列表，如 ["low", "haze", "rain"]
    
    Returns:
        str: 退化名称，如 "low_haze_rain"
    """
    # 按索引顺序排序，确保一致性
    sorted_factors = sorted(factors, key=lambda f: FACTOR2IDX.get(f, 999))
    return "_".join(sorted_factors)


def get_leave_one_out_name(deg_name):
    """
    获取leave-one-out的退化名称列表
    
    Args:
        deg_name: 原始退化名称，如 "low_haze_rain"
    
    Returns:
        list: leave-one-out名称列表，如 ["haze_rain", "low_rain", "low_haze"]
    """
    factors = parse_factors(deg_name)
    
    if len(factors) <= 1:
        return []
    
    loo_names = []
    for i, factor in enumerate(factors):
        remaining = factors[:i] + factors[i+1:]
        if remaining:
            loo_names.append(build_name(remaining))
    
    return loo_names


def present_to_factors(present):
    """
    从present向量转换为因子列表
    
    Args:
        present: (4,) 的present向量
    
    Returns:
        list: 因子列表
    """
    factors = []
    for idx, val in enumerate(present):
        if val > 0.5:  # 阈值
            factor = IDX2FACTOR.get(idx)
            if factor:
                factors.append(factor)
    return factors
