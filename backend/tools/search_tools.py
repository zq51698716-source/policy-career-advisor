"""
Tavily search tools — web search, policy search, and job market search.

Migrated from agent/tools.py with the same execution logic and schemas.
"""

import json
from datetime import datetime
from typing import Any

from tavily import TavilyClient

from backend.config import TAVILY_API_KEY

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


# ============================================================
# Tool JSON Schemas (for Claude API)
# ============================================================
TOOL_SCHEMAS = {
    "search_web": {
        "name": "search_web",
        "description": "通用联网搜索工具。搜索互联网获取最新信息，适合查询新闻、百科、一般性问题。返回结果包含标题、URL、内容摘要。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，建议使用多个关键词组合以提高精确度。例如：'2026年深圳人才补贴政策 申请条件'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "返回结果数量，默认5条，最多10条",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    "search_policy": {
        "name": "search_policy",
        "description": "政策专项搜索工具。专门用于搜索中国各级政府的政策文件，会自动优先搜索政府官方网站（gov.cn）。适合查询就业政策、人才政策、补贴政策、社保政策等。",
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "政策关键词，例如：'人才引进补贴'、'灵活就业社保补贴'、'高校毕业生就业扶持'",
                },
                "region": {
                    "type": "string",
                    "description": "地区限定，如'全国'、'深圳'、'北京'、'上海'、'广东'等。如用户未指定地区，填'全国'",
                    "default": "全国",
                },
                "policy_type": {
                    "type": "string",
                    "description": "政策类型：employment（就业）、talent（人才引进）、subsidy（补贴）、social_security（社保）、entrepreneurship（创业）。如不确定，填'general'",
                    "enum": ["employment", "talent", "subsidy", "social_security", "entrepreneurship", "general"],
                    "default": "general",
                },
            },
            "required": ["keyword"],
        },
    },
    "search_job_market": {
        "name": "search_job_market",
        "description": "岗位市场搜索工具。专门用于查询行业薪资水平、岗位需求趋势、招聘要求等职业市场数据。",
        "input_schema": {
            "type": "object",
            "properties": {
                "position": {
                    "type": "string",
                    "description": "目标岗位，例如：'前端开发'、'Java工程师'、'产品经理'、'数据分析师'",
                },
                "location": {
                    "type": "string",
                    "description": "工作城市，如'北京'、'上海'、'深圳'、'杭州'等",
                    "default": "全国",
                },
                "industry": {
                    "type": "string",
                    "description": "所属行业，如'互联网'、'金融'、'制造业'、'新能源'等。如不确定，填'general'",
                    "default": "general",
                },
            },
            "required": ["position"],
        },
    },
}


# ============================================================
# Tool Execution Functions
# ============================================================
def _search_web(query: str, max_results: int = 5) -> str:
    """通用联网搜索"""
    try:
        response = tavily_client.search(
            query=query,
            max_results=min(max_results, 10),
            search_depth="advanced",
            include_domains=[],
            exclude_domains=[],
        )
        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0),
            })
        return json.dumps({"query": query, "results": results}, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({"error": f"搜索失败: {str(e)}"}, ensure_ascii=False)


def _search_policy(keyword: str, region: str = "全国", policy_type: str = "general") -> str:
    """政策专项搜索 - 优先搜索政府网站"""
    policy_query = f"{region} {keyword} 政策 site:gov.cn"

    try:
        response = tavily_client.search(
            query=policy_query,
            max_results=5,
            search_depth="advanced",
            include_domains=["gov.cn"],
        )

        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "source": "政府官方网站" if "gov.cn" in r.get("url", "") else "其他来源",
            })

        # 如果政府网站结果不足，补充通用搜索
        if len(results) < 3:
            response2 = tavily_client.search(
                query=f"{region} {keyword} 最新政策 官方发布",
                max_results=3,
                search_depth="advanced",
            )
            for r in response2.get("results", []):
                if not any(existing["url"] == r.get("url") for existing in results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                        "source": "新闻/资讯来源",
                    })

        return json.dumps({
            "query": f"{region} - {keyword}",
            "policy_type": policy_type,
            "results": results[:8],
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": f"政策搜索失败: {str(e)}"}, ensure_ascii=False)


def _search_job_market(position: str, location: str = "全国", industry: str = "general") -> str:
    """岗位市场搜索 - 查询薪资、需求、趋势"""
    queries = [
        f"{location} {position} 薪资水平 2026",
        f"{location} {position} 招聘要求 岗位需求",
        f"{industry} {position} 行业发展趋势 2026",
    ]

    all_results = []

    try:
        for q in queries[:2]:  # 只搜前两个维度，节省时间
            response = tavily_client.search(
                query=q,
                max_results=3,
                search_depth="advanced",
            )
            for r in response.get("results", []):
                if not any(existing["url"] == r.get("url") for existing in all_results):
                    all_results.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "content": r.get("content", ""),
                    })

        return json.dumps({
            "position": position,
            "location": location,
            "industry": industry,
            "results": all_results[:6],
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": f"市场搜索失败: {str(e)}"}, ensure_ascii=False)


# ============================================================
# Executor registry: tool_name → callable
# ============================================================
EXECUTORS = {
    "search_web": _search_web,
    "search_policy": _search_policy,
    "search_job_market": _search_job_market,
}
