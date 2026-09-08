"""Presentation-only checks for human-readable reports; never alter decisions."""

from __future__ import annotations

import json
import re


REPORT_WRITING_RULES = """
面向用户的报告写作与自检要求：
- 正文、标题、摘要、风险说明及供下游使用的自然语言字段全部使用简体中文。
  股票代码、来源名称、链接及必要指标缩写可保留原文，指标首次出现时说明中文含义。
- 使用 Markdown，先写“信息概括”（已获取的数据、时间范围和来源），再写“结论”，
  随后按需要写“关键依据”“风险与失效条件”“待补充信息”。短段落和列表优先，避免宽表格。
- 正文禁止任何 JSON、对象转储、工具原始输出、内部字段名和调试过程。
  后文要求的结构化对象仅用于内部解析，必须独立放在末尾，不属于报告正文。
- 数据缺失、过期、工具失败或无法核实的部分明确标注“【待补充】”，说明缺什么、
  对结论有何限制；零分不等于中性证据，未发现风险不等于已排除风险，不得编造数据。
- 按金融研究标准区分事实、分析判断与行动结论。数字注明单位、时点和统计口径；
  不混淆百分比与百分点、行情日期与财报期间、潜在催化与已发生事件。
- 技术分析区分个股、行业与大盘，说明支撑、阻力和风险收益比适用的价格假设；
  基本面区分盈利、现金流、资产负债与估值；新闻保留来源、发布日期和链接。
  只分析实际取得证据的维度，不强行补齐所有栏目。
- 行动结论交代方向、触发条件与失效条件；计划、风控批准、待确认和已成交必须区分。
  不得把机会数量称为已执行交易，不得承诺收益或夸大确定性。
- 提交前静默自检并直接修正：中文和 Markdown、无正文 JSON、信息概括及结论齐全、
  缺口标记完整、数值与单位一致、结论与结构化决策/风险约束一致、引用确有来源。
  若证据互相冲突，明确指出分歧并限制结论，不得自行虚构调和理由。不输出自检过程。
""".strip()


def strip_report_json(content: str) -> str:
    """Remove fenced or inline JSON, including arrays and braces inside strings."""
    def clean_fence(match: re.Match) -> str:
        label, body = match.group(1).strip().lower(), match.group(2).strip()
        if label in {"json", "jsonc", "json5"} or body.startswith(("{", "[")):
            return ""
        return match.group(0)

    content = re.sub(r"```([^\n`]*)\n(.*?)(?:```|\Z)", clean_fence, content, flags=re.S)
    decoder = json.JSONDecoder()
    result: list[str] = []
    index = 0
    while index < len(content):
        if content[index] in "{[":
            try:
                value, end = decoder.raw_decode(content[index:])
            except ValueError:
                pass
            else:
                if isinstance(value, (dict, list)):
                    index += end
                    continue
        result.append(content[index])
        index += 1
    return re.sub(r"\n{3,}", "\n\n", "".join(result)).strip()


def prepare_report(content: str) -> str:
    """Final publication gate, also used for deterministic fallback reports."""
    content = strip_report_json(content)
    # Replace missing-value placeholders only in display cells or list values.
    content = re.sub(r"(?<=\|)(\s*)(?:-|None|N/A|unknown)(\s*)(?=\|)",
                     r"\1【待补充】\2", content)
    content = re.sub(r"([：:]\s*)(?:None|N/A|unknown|-)(?=\s*$)",
                     r"\1【待补充】", content, flags=re.M)
    if not content:
        return "## 信息概括\n\n【待补充】未取得可展示的分析信息。\n\n## 结论\n\n【待补充】证据不足，无法形成可靠结论。\n"
    # Deterministic reports may have only facts. Do not invent a conclusion.
    if "信息概括" not in content:
        lines = content.splitlines()
        insert_at = 1 if lines[0].startswith("# ") else 0
        lines[insert_at:insert_at] = ["", "## 信息概括", ""]
        content = "\n".join(lines)
    if not re.search(r"结论|综合判断|技术立场", content):
        conclusions = [line for line in content.splitlines()
                       if re.search(r"(?:状态|方向|原因|决策依据|投资逻辑)[：:]", line)]
        content += "\n\n## 结论\n\n" + (
            "\n".join(conclusions[:3]) if conclusions
            else "【待补充】本阶段仅提供已获取的信息，尚不足以形成独立的行动结论。"
        )
    return re.sub(r"\n{3,}", "\n\n", content).rstrip() + "\n"
