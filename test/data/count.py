#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计文件夹中每个“数据集”（按文件）包含的：
- 数值个数
- 小数精度（decimal places）的平均值/最大值（会去掉小数末尾0；支持科学计数法）
- 有效位数（significant digits）的平均值/最大值（按常见教科书规则；支持科学计数法）

用法举例：
    python dataset_digit_stats.py "D:/data" --exts .csv .txt .tsv .dat

注意：
1) 建议以文本形式存储数据；二进制浮点(.npy/.npz等)无法可靠恢复“原始小数位数”。
2) 我们以“字符串”来解析每个数值，以保留它在文件中的原始书写格式（含小数位与科学计数法）。
3) 小数精度定义为：去除小数部分末尾的0后，实际需要的小数位数；对科学计数法，会先根据指数移动小数点。
4) 有效位数规则：
   - 先去掉正负号；若有小数点，所有数字(除去小数点)从第一个非零开始到末尾全计为有效位（包括尾随0）。
   - 若无小数点（纯整数书写），尾随0不计入有效位。
   - 特例 “0”、“0.0”等，记作 1 位有效数字。
"""

from __future__ import annotations
import argparse
import math
import os
import re
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Optional, Tuple

NUMBER_RE = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')
TRIM_CHARS = ' \t\r\n,;:[](){}"\'`'

def is_number_token(tok: str) -> bool:
    s = tok.strip(TRIM_CHARS)
    if not s:
        return False
    low = s.lower()
    if low in ('nan', '+nan', '-nan', 'inf', '+inf', '-inf'):
        return False
    return bool(NUMBER_RE.match(s))

def count_decimal_places(num_str: str) -> int:
    """
    返回最少需要的小数位数（去掉小数末尾0）。
    规则：
      - 若为科学计数，如 a.bcd e±k，则先计算 mantissa 的有效小数位 dec，
        再用 dec - k，最后与 0 取最大值。
      - 若无科学计数法，直接统计小数点后的位数，去掉末尾0。
    """
    s = num_str.strip(TRIM_CHARS).lower()
    # 去掉正负号
    if s.startswith(('+','-')):
        s = s[1:]
    # 科学计数法分离
    if 'e' in s:
        mantissa, exp = s.split('e', 1)
        try:
            k = int(exp.strip())
        except ValueError:
            return 0
    else:
        mantissa, k = s, 0

    if '.' in mantissa:
        int_part, frac = mantissa.split('.', 1)
        frac = frac.rstrip('0')  # 去掉小数末尾的0
        dec_in_mantissa = len(frac)
    else:
        dec_in_mantissa = 0

    # 负指数会增加小数位，正指数会减少小数位
    dec = dec_in_mantissa - k
    return dec if dec > 0 else 0

def count_significant_digits(num_str: str) -> int:
    """
    计算有效位数（Significant Digits）。规则：
      - 去掉正负号；若字符串包含小数点，则尾随0视为有效（例如 1.2300 有 5 位有效数字）。
      - 若不含小数点（纯整数），则尾随0不算有效（例如 1200 有 2 位有效数字）。
      - 科学计数法仅根据“尾数部分”判断（例如 1.230e3 => 4 位有效数字）。
      - 全零（如 "0"、"0.0"、"000"）视为 1 位有效数字。
    """
    s = num_str.strip(TRIM_CHARS).lower()
    if s.startswith(('+','-')):
        s = s[1:]
    # 排除无穷/NaN等
    if s in ('nan', 'inf'):
        return 0

    if 'e' in s:
        mantissa = s.split('e', 1)[0]
    else:
        mantissa = s

    has_decimal = '.' in mantissa
    digits = [ch for ch in mantissa if ch.isdigit()]
    if not digits:
        return 0

    # 检测是否全为0
    if all(d == '0' for d in digits):
        return 1

    # 去掉前导0
    first_non_zero = 0
    while first_non_zero < len(digits) and digits[first_non_zero] == '0':
        first_non_zero += 1
    if first_non_zero >= len(digits):
        # 形如 "0.0001200" 这种情况，上面全零判断会提前返回；这里防御
        return 1

    if has_decimal:
        # 有小数点：从第一个非零到结尾，全部算有效（包括尾随0）
        return len(digits) - first_non_zero
    else:
        # 无小数点：尾随0不算有效
        end = len(digits)
        while end > first_non_zero and digits[end - 1] == '0':
            end -= 1
        return max(0, end - first_non_zero)

def iter_number_tokens_from_text_file(path: Path) -> Iterable[str]:
    """
    从纯文本文件中迭代出所有“看起来像数字”的 token（字符串形式）。
    我们使用正则与简单分割的结合来鲁棒提取。
    """
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # 先用常见分隔符切分
            # 注意：这是一种启发式方法，若你的文件包含千位分隔符等，请先清洗。
            for raw in re.split(r'[,\s;]+', line.strip()):
                tok = raw.strip(TRIM_CHARS)
                if tok and is_number_token(tok):
                    yield tok

def analyze_file(path: Path) -> Tuple[int, Optional[float], Optional[int], Optional[float], Optional[int]]:
    """
    返回：
      (count, avg_decimal_places, max_decimal_places, avg_sig_digits, max_sig_digits)
    若文件中没有数字，返回 (0, None, None, None, None)。
    """
    decimals: List[int] = []
    sigs: List[int] = []

    try:
        for tok in iter_number_tokens_from_text_file(path):
            dp = count_decimal_places(tok)
            sd = count_significant_digits(tok)
            decimals.append(dp)
            sigs.append(sd)
    except Exception as e:
        # 若某文件有编码/格式问题，不中断整体流程
        print(f"[WARN] 解析失败: {path} -> {e}")

    if not decimals or not sigs:
        return 0, None, None, None, None

    return (
        len(decimals),
        float(mean(decimals)),
        int(max(decimals)),
        float(mean(sigs)),
        int(max(sigs)),
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计每个数据集文件的数值个数、平均/最大 小数精度 与 有效位数（文本解析方式）。"
    )
    parser.add_argument("folder", help="待扫描的文件夹路径")
    parser.add_argument(
        "--exts",
        nargs="+",
        default=[".csv", ".tsv", ".txt", ".dat", ".log"],
        help="需要统计的文件扩展名（默认：.csv .tsv .txt .dat .log）"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描子目录（默认不递归）"
    )
    parser.add_argument(
        "--save",
        default="dataset_digit_stats_summary.csv",
        help="将结果另存为 CSV 的文件名（默认：dataset_digit_stats_summary.csv）"
    )
    args = parser.parse_args()

    root = Path(args.folder).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"路径不存在或不是文件夹: {root}")

    chosen_exts = set(e.lower() for e in args.exts)
    files: List[Path] = []
    if args.recursive:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in chosen_exts:
                files.append(p)
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in chosen_exts:
                files.append(p)

    files.sort()

    rows = []
    print(f"扫描路径: {root}")
    print(f"文件数量: {len(files)}")
    for fp in files:
        count, avg_dp, max_dp, avg_sd, max_sd = analyze_file(fp)
        rows.append({
            "dataset": fp.stem,
            "file": str(fp),
            "count": count,
            "avg_decimal_places": None if avg_dp is None else round(avg_dp, 6),
            "max_decimal_places": max_dp,
            "avg_sig_digits": None if avg_sd is None else round(avg_sd, 6),
            "max_sig_digits": max_sd,
        })

    # 打印表格（简版）
    if rows:
        # 对齐打印
        cols = ["dataset", "count", "avg_decimal_places", "max_decimal_places", "avg_sig_digits", "max_sig_digits", "file"]
        header = "\t".join(cols)
        print("\n" + header)
        print("-" * len(header.expandtabs()))
        for r in rows:
            print("\t".join(str(r.get(c, "")) for c in cols))
    else:
        print("未找到匹配的文件或未解析到任何数值。")

    # 保存 CSV
    try:
        import csv
        out_csv = (Path.cwd() / args.save).resolve()
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "file", "count", "avg_decimal_places", "max_decimal_places", "avg_sig_digits", "max_sig_digits"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"\n统计结果已保存：{out_csv}")
    except Exception as e:
        print(f"[WARN] 保存 CSV 失败: {e}")

if __name__ == "__main__":
    main()
