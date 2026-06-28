"""
Utility tools — small helpers like current time.
"""

import json
from datetime import datetime


TOOL_SCHEMAS = {
    "get_current_time": {
        "name": "get_current_time",
        "description": "获取当前日期和时间。用于判断政策时效性、计算截止日期等。当你需要知道「今天是什么日期」时调用此工具。",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


def _get_current_time() -> str:
    """获取当前时间"""
    now = datetime.now()
    return json.dumps({
        "current_time": now.strftime("%Y年%m月%d日 %H:%M"),
        "date": now.strftime("%Y-%m-%d"),
        "day_of_week": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()],
        "year": now.year,
        "month": now.month,
    }, ensure_ascii=False)


EXECUTORS = {
    "get_current_time": _get_current_time,
}
