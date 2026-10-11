"""评测运行器：一键跑出四项离线指标，并可选启用 LLM-as-Judge 评估结论质量。

用法::

    python knowledge/eval/run_eval.py                 # 离线指标（无需任何三方服务）
    python knowledge/eval/run_eval.py --json out.json  # 同时写出机器可读结果
    python knowledge/eval/run_eval.py --judge          # 追加 LLM-as-Judge（无可用 LLM 时自动跳过）
    python knowledge/eval/run_eval.py --judge --reports gen.json
                                                      # 用真实生成的报告文本送评（{qa_id: report}）

设计取舍：
- **离线优先**：规则命中 / 风险等级 / HITL 是确定性结论，离线严格校验；
  意图识别离线只有关键词兜底，因此把准确率作为「参考值」如实报出（低分本身就是有用信息：
  它说明哪类问题必须走 LLM）。
- **LLM-as-Judge 可跳过硬失败**：没有 API Key / LLM 不可用时打印「已跳过」并正常退出，
  不让离线环境因为缺少外部依赖而变红。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from knowledge.eval.dataset import (  # noqa: E402
    EvalReport, dataset_meta, evaluate_dataset, load_samples, predict_sample,
)

JUDGE_SYSTEM_PROMPT = """你是电商平台合规审查的质量评审员。
给定「审查结论」（命中的合规规则与证据）与「参考期望」，请判断结论是否**被证据充分支撑**。
只输出 JSON：{"score": 1-5 的整数, "reason": "简短理由", "unsupported": ["缺乏证据支撑的结论"]}
评分口径：5=结论与证据完全一致且无遗漏；3=主要结论正确但证据不足或表述含糊；1=结论与证据矛盾。
不要因为参考期望里有的规则而给分——只看证据是否真的支撑结论。"""


# ==================== LLM-as-Judge ====================

def _judge_prompt(sample: Dict[str, Any], prediction: Dict[str, Any]) -> str:
    expect = sample.get("expect") or {}
    findings = prediction.get("findings") or []
    findings_text = "\n".join(
        f"- {f['rule_id']} {f['rule_name']}（{f['severity']}）：{f['message']}\n  证据：{f.get('evidence')}"
        for f in findings
    ) or "（无命中规则）"
    return (
        f"审查主体：{sample.get('company')}\n"
        f"审查类型：{sample.get('review_type')}\n"
        f"问题：{sample.get('question')}\n\n"
        f"【系统给出的审查结论】\n风险等级：{prediction.get('risk_level')}\n{findings_text}\n\n"
        f"【参考期望】\n规则：{expect.get('rules')} 风险等级：{expect.get('risk_level')}\n"
    )


def run_judge(samples: List[Dict[str, Any]], reports: Optional[Dict[str, str]] = None,
              limit: Optional[int] = None) -> Dict[str, Any]:
    """LLM-as-Judge：可选能力，任何不可用情况都返回 skipped 而不是抛异常。"""
    try:
        from knowledge.utils.llm_client import get_llm_client
        # 不强制 json_mode：统一由 parse_json_response 做兜底解析，避免服务端 400 直接失败
        client = get_llm_client()
    except Exception as e:                                    # 依赖缺失
        return {"status": "skipped", "reason": f"LLM 客户端不可用：{e}", "items": []}
    if client is None:
        return {"status": "skipped",
                "reason": "未安装 langchain_deepseek 或无可用模型配置（设置 MODEL / DEEPSEEK_API_KEY 后可启用）",
                "items": []}

    items: List[Dict[str, Any]] = []
    for sample in samples[: limit or len(samples)]:
        content = _judge_prompt(sample, predict_sample(sample))
        if reports and sample["id"] in reports:
            content += f"\n【系统生成的审查报告】\n{reports[sample['id']]}\n"
        try:
            data = client.invoke_json([
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ])
        except Exception as e:
            items.append({"id": sample["id"], "score": None, "reason": f"调用失败：{e}"})
            continue
        if not isinstance(data, dict) or data.get("score") is None:
            items.append({"id": sample["id"], "score": None, "reason": "返回结构不合法"})
            continue
        items.append({
            "id": sample["id"],
            "score": int(data.get("score") or 0),
            "reason": str(data.get("reason") or "")[:200],
            "unsupported": data.get("unsupported") or [],
        })

    scored = [i["score"] for i in items if isinstance(i.get("score"), int)]
    failed = [i["id"] for i in items if i.get("score") is None]
    return {
        "status": "ok" if scored else "skipped",
        "reason": "" if scored else "全部样本评分失败",
        "scored": len(scored),
        "failed_calls": failed,
        "mean_score": round(sum(scored) / len(scored), 3) if scored else 0.0,
        "pass_rate_ge4": round(sum(1 for s in scored if s >= 4) / len(scored), 4) if scored else 0.0,
        "items": items,
    }


# ==================== 输出 ====================

def _render(report: EvalReport, judge: Optional[Dict[str, Any]] = None) -> str:
    lines = [
        "# 电商合规审查 · 评测结果",
        "",
        "## 离线指标（确定性，可复现）",
        "",
        report.to_markdown(),
        "",
    ]
    if report.false_positive_samples:
        lines += [f"- 误报样本（期望无规则却命中）：{', '.join(report.false_positive_samples)}", ""]
    if report.missed_samples:
        lines += [f"- 漏检样本（期望有规则却全未命中）：{', '.join(report.missed_samples)}", ""]
    if judge is not None:
        lines += ["## LLM-as-Judge（在线能力，离线自动跳过）", ""]
        if judge.get("status") != "ok":
            lines += [f"已跳过：{judge.get('reason')}", ""]
        else:
            lines += [
                f"- 已评分样本：**{judge['scored']}**",
                f"- 平均分（1–5）：**{judge['mean_score']}**",
                f"- 达标率（≥4 分）：**{judge['pass_rate_ge4']:.1%}**",
                f"- 调用失败：{judge.get('failed_calls') or '无'}",
                "",
            ]
    if report.mismatches:
        lines += ["## 不一致明细", ""] + [f"- {m}" for m in report.mismatches] + [""]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="run_eval", description="电商合规审查评测集运行器")
    parser.add_argument("--judge", action="store_true", help="启用 LLM-as-Judge（无可用 LLM 时自动跳过）")
    parser.add_argument("--reports", default="", help="可选：{qa_id: 生成的报告文本} JSON 文件，供 Judge 评报告质量")
    parser.add_argument("--json", default="", help="把结果写入指定 JSON 文件")
    parser.add_argument("--limit", type=int, default=0, help="仅评测前 N 条（调试用）")
    args = parser.parse_args(argv)

    samples = load_samples()
    if not samples:
        print("评测集为空", file=sys.stderr)
        return 1
    if args.limit:
        samples = samples[: args.limit]

    report = evaluate_dataset(samples)

    judge_result: Optional[Dict[str, Any]] = None
    if args.judge:
        reports = {}
        if args.reports and os.path.exists(args.reports):
            with open(args.reports, "r", encoding="utf-8") as f:
                reports = json.load(f)
        judge_result = run_judge(samples, reports=reports)

    print(_render(report, judge_result))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({
                "offline": report.to_dict(),
                "llm_judge": judge_result or {"status": "not_requested"},
                "dataset": dataset_meta(samples),
            }, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 结果已写入 {args.json}")

    # 确定性指标必须全绿：规则命中集合、风险等级、HITL 判定
    assert not report.mismatches, "评测存在不一致：\n" + "\n".join(report.mismatches)
    return 0


if __name__ == "__main__":
    sys.exit(main())
