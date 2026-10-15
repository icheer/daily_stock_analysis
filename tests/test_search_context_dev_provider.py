# -*- coding: utf-8 -*-
"""
Context.dev 搜索引擎测试套件

测试覆盖范围:
1. 请求构造测试 - 验证 API URL、鉴权头、payload 字段（numResults 上下限、freshness 映射）
2. 结果映射测试 - 验证 description -> snippet、域名提取、URL 内嵌日期提取
3. 错误处理测试 - 验证 HTTP 错误、JSON 解析失败时的降级
4. 服务初始化测试 - 验证 SearchService 是否正确注册 ContextDev provider

运行方式:
python -m pytest tests/test_search_context_dev_provider.py -v
"""

import os
import sys
import unittest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

# 添加项目根目录到 Python 路径，解决模块导入问题
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mock newspaper before search_service import (optional dependency)
if "newspaper" not in sys.modules:
    mock_np = MagicMock()
    mock_np.Article = MagicMock()
    mock_np.Config = MagicMock()
    sys.modules["newspaper"] = mock_np

from src.search_service import (
    ContextDevSearchProvider,
    SearchService,
    _post_with_retry,
)


class _FakeResponse:
    """模拟 HTTP 响应对象（json_data 传 _NO_JSON 表示响应体不是合法 JSON）"""

    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        if self._json_data is None:
            raise ValueError("no json")
        return self._json_data


def _sample_payload():
    return {
        "query": "贵州茅台 最新新闻",
        "results": [
            {
                "url": "https://finance.eastmoney.com/a/202610133893317988.html",
                "title": "贵州茅台重回第一高价股",
                "description": "截至当天收盘，贵州茅台涨1.98%报1288元。" + "x" * 600,
                "relevance": "high",
            },
            {
                "url": "https://www.zaobao.com.sg/news/china/story20261012-9823128",
                "title": "贵州茅台重新成为A股市值最高股票",
                "description": "联讯仪器星期一跌超7.5%。",
                "relevance": "high",
            },
            {
                "url": "https://example.com/no-date-in-url",
                "title": "无日期链接",
                "description": "没有内嵌日期。",
                "relevance": "low",
            },
        ],
    }


class TestContextDevFreshnessMapping(unittest.TestCase):
    """days -> freshness 枚举映射"""

    def test_freshness_mapping(self):
        cases = {
            1: "last_24_hours",
            3: "last_week",
            7: "last_week",
            14: "last_month",
            31: "last_month",
            90: "last_year",
        }
        for days, expected in cases.items():
            self.assertEqual(ContextDevSearchProvider._freshness(days), expected)


class TestContextDevSearchProvider(unittest.TestCase):
    """Context.dev provider 请求构造与结果映射"""

    def setUp(self):
        self.provider = ContextDevSearchProvider(["dummy_key"])

    def _search(self, payload, **kwargs):
        fake = _FakeResponse(status_code=200, json_data=payload)
        with patch("src.search_service._post_with_retry", return_value=fake) as mock_post:
            response = self.provider.search("贵州茅台 最新新闻", **kwargs)
        return response, mock_post

    def test_request_url_auth_and_payload(self):
        response, mock_post = self._search(_sample_payload(), max_results=5, days=3)
        self.assertTrue(response.success)

        self.assertEqual(mock_post.call_count, 1)
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], ContextDevSearchProvider.API_URL)
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer dummy_key")
        self.assertEqual(kwargs["json"]["query"], "贵州茅台 最新新闻")
        # max_results=5 低于 API 下限 10，应抬升到 10
        self.assertEqual(kwargs["json"]["numResults"], 10)
        self.assertEqual(kwargs["json"]["freshness"], "last_week")

    def test_num_results_upper_bound(self):
        _, mock_post = self._search(_sample_payload(), max_results=150, days=1)
        kwargs = mock_post.call_args.kwargs
        self.assertEqual(kwargs["json"]["numResults"], 100)

    def test_result_mapping_and_snippet_truncation(self):
        response, _ = self._search(_sample_payload(), max_results=10, days=7)
        self.assertTrue(response.success)
        self.assertEqual(response.provider, "ContextDev")
        self.assertEqual(len(response.results), 3)

        first = response.results[0]
        self.assertEqual(first.title, "贵州茅台重回第一高价股")
        self.assertEqual(first.source, "finance.eastmoney.com")
        # description 截断到 500 字符
        self.assertLessEqual(len(first.snippet), 500)
        # URL 内嵌日期 /20261013/ 应被提取
        self.assertEqual(first.published_date, "2026-10-13")
        # story20261012-xxx 形式也应被提取
        self.assertEqual(response.results[1].published_date, "2026-10-12")
        # 无日期 URL 保持 None
        self.assertIsNone(response.results[2].published_date)

    def test_invalid_url_date_rejected(self):
        self.assertIsNone(
            ContextDevSearchProvider._extract_date_from_url("https://example.com/20991301/x")
        )

    def test_http_error_returns_failed_response(self):
        fake = _FakeResponse(status_code=401, json_data={"message": "invalid api key"})
        with patch("src.search_service._post_with_retry", return_value=fake):
            response = self.provider.search("query", max_results=10)
        self.assertFalse(response.success)
        self.assertIn("401", response.error_message)
        self.assertIn("invalid api key", response.error_message)
        self.assertEqual(response.results, [])

    def test_invalid_json_returns_failed_response(self):
        fake = _FakeResponse(status_code=200, json_data=None)
        with patch("src.search_service._post_with_retry", return_value=fake):
            response = self.provider.search("query", max_results=10)
        self.assertFalse(response.success)
        self.assertIn("JSON", response.error_message)

    def test_base_class_key_rotation(self):
        """多 Key 时按基类轮换使用"""
        provider = ContextDevSearchProvider(["key_a", "key_b"])
        fake = _FakeResponse(status_code=200, json_data={"results": []})
        with patch("src.search_service._post_with_retry", return_value=fake) as mock_post:
            provider.search("q1", max_results=10)
            provider.search("q2", max_results=10)
        used_keys = [call.kwargs["headers"]["Authorization"] for call in mock_post.call_args_list]
        self.assertEqual(
            sorted(used_keys),
            ["Bearer key_a", "Bearer key_b"],
        )


class TestSearchServiceContextDevRegistration(unittest.TestCase):
    """SearchService 集成：Context.dev 注册与优先级"""

    def test_provider_registered_between_tavily_and_brave(self):
        service = SearchService(
            tavily_keys=["tvly_key"],
            context_dev_keys=["ctxt_key"],
            brave_keys=["brave_key"],
        )
        names = [p.name for p in service._providers]
        self.assertIn("ContextDev", names)
        self.assertLess(names.index("Tavily"), names.index("ContextDev"))
        self.assertLess(names.index("ContextDev"), names.index("Brave"))

    def test_no_keys_no_provider(self):
        service = SearchService()
        names = [p.name for p in service._providers]
        self.assertNotIn("ContextDev", names)

    def test_positional_call_signature_unchanged(self):
        """旧位置传参兼容：既有参数相对顺序必须保持不变（review OR-COM-5e9eee92）。"""
        service = SearchService(
            ["bocha_k"],    # bocha_keys
            ["tavily_k"],   # tavily_keys
            ["anspire_k"],  # anspire_keys
            ["brave_k"],    # brave_keys
            ["serpapi_k"],  # serpapi_keys
            ["minimax_k"],  # minimax_keys
            [],             # searxng_base_urls
            False,          # searxng_public_instances_enabled
            None,           # searxng_timeout_seconds
            3,              # news_max_age_days
            "short",        # news_strategy_profile
        )
        by_name = {p.name: p._api_keys for p in service._providers}
        self.assertEqual(by_name.get("Bocha"), ["bocha_k"])
        self.assertEqual(by_name.get("Tavily"), ["tavily_k"])
        self.assertEqual(by_name.get("Anspire"), ["anspire_k"])
        self.assertEqual(by_name.get("Brave"), ["brave_k"])
        self.assertEqual(by_name.get("SerpAPI"), ["serpapi_k"])
        self.assertEqual(by_name.get("MiniMax"), ["minimax_k"])
        self.assertNotIn("ContextDev", by_name)

    def test_context_dev_keys_never_leak_to_other_providers(self):
        """context_dev_keys 只能经关键字或末尾位置参数传入，不得错位到其他 provider。"""
        service = SearchService(
            ["bocha_k"], ["tavily_k"], ["anspire_k"], ["brave_k"], ["serpapi_k"],
            ["minimax_k"], [], False, None, 3, "short", ["ctxt_k"],
        )
        by_name = {p.name: p._api_keys for p in service._providers}
        self.assertEqual(by_name.get("ContextDev"), ["ctxt_k"])
        self.assertEqual(by_name.get("Anspire"), ["anspire_k"])
        # Anspire Key 不应被传给 Context.dev（避免第三方凭据错误披露）
        self.assertNotIn("anspire_k", by_name.get("ContextDev", []))
        self.assertNotIn("ctxt_k", by_name.get("Anspire", []))

    def test_search_capability_via_config(self):
        from src.config import Config

        config = Config(context_dev_api_keys=["ctxt_key"])
        self.assertTrue(config.has_search_capability_enabled())


if __name__ == "__main__":
    unittest.main()
