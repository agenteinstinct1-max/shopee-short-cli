import json
import os
import unittest
from unittest.mock import patch

import requests

from shopee_short.audit import AuditLog
from shopee_short.discovery import ImageCandidate
from shopee_short.errors import ProductNotFound, ProviderError
from shopee_short.llm import (
    DEFAULT_BASE_URL,
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_MODEL,
    generate_script,
)
from shopee_short.provenance import (
    PageEvidence,
    model_codes,
    primary_model_code,
    verify_image_sources,
    verify_product_identity,
)


class ProductIdentityTests(unittest.TestCase):
    def test_model_code_keeps_plus_variant_distinct(self):
        self.assertEqual(
            primary_model_code("MAGCUBIC HY300ProPlus Projector 290 ANSI"),
            "hy300proplus",
        )
        self.assertEqual(primary_model_code("MAGCUBIC HY300 Pro Projector 200 ANSI"), "hy300pro")
        self.assertIn("hy300proplus", model_codes("Magcubic HY300Pro+ Smart Mini Projector"))
        self.assertNotIn("hy300proplus", model_codes("Magcubic HY300Pro Smart Mini Projector"))

    def test_wrong_model_page_is_rejected(self):
        page = PageEvidence(
            url="https://example.org/product",
            status=200,
            title="Replacement Pads MDR-ZX110",
            h1="Replacement Pads MDR-ZX110",
            meta_title="",
            identity_codes=model_codes("Replacement Pads MDR-ZX110"),
        )
        with self.assertRaises(ProductNotFound):
            verify_product_identity("JBL Tune 510BT Ear Pads", page, label="foto")

    def test_photo_must_be_bound_to_model_labelled_gallery_image(self):
        product_name = "MAGCUBIC HY300ProPlus Projector 290 ANSI"
        page_url = "https://www.magcubic.com/products/miniprojector-hy300proplus"
        image_url = "https://www.magcubic.com/cdn/shop/files/hy300_pro.png?v=1&width=600"
        page = PageEvidence(
            url=page_url,
            status=200,
            title="Magcubic HY300Pro+ Smart Mini Projector",
            h1="Magcubic HY300Pro+ Smart Mini Projector",
            meta_title="",
            identity_codes=model_codes("Magcubic HY300Pro+ Smart Mini Projector"),
            images=[{
                "alt": "Magcubic HY300Pro+ Smart Mini Projector",
                "sources": "//www.magcubic.com/cdn/shop/files/hy300_pro.png?v=1&width=600",
            }],
        )
        candidate = ImageCandidate(image_url, page_url, "hy300_pro.png", "www.magcubic.com", False)
        with patch("shopee_short.provenance.fetch_page_evidence", return_value=page):
            evidence = verify_image_sources([candidate], product_name)
        self.assertTrue(evidence[0]["identity_match"])
        self.assertTrue(evidence[0]["image_bound_to_page"])
        self.assertEqual(evidence[0]["expected_model_code"], "hy300proplus")

    def test_correct_model_page_does_not_authorize_unlisted_image(self):
        product_name = "MAGCUBIC HY300ProPlus Projector 290 ANSI"
        page_url = "https://www.magcubic.com/products/miniprojector-hy300proplus"
        image_url = "https://images.example.org/unrelated.jpg"
        page = PageEvidence(
            url=page_url,
            status=200,
            title="Magcubic HY300Pro+ Smart Mini Projector",
            h1="Magcubic HY300Pro+ Smart Mini Projector",
            meta_title="",
            identity_codes=model_codes("Magcubic HY300Pro+ Smart Mini Projector"),
            images=[],
        )
        candidate = ImageCandidate(image_url, page_url, "unrelated.jpg", "images.example.org", False)
        with patch("shopee_short.provenance.fetch_page_evidence", return_value=page):
            with self.assertRaises(ProductNotFound):
                verify_image_sources([candidate], product_name)


class PollinationsTests(unittest.TestCase):
    @staticmethod
    def response(status, body):
        result = requests.Response()
        result.status_code = status
        result.headers["content-type"] = "application/json; charset=utf-8"
        result._content = body.encode("utf-8")
        result.url = DEFAULT_BASE_URL + "/chat/completions"
        return result

    def test_defaults_use_requested_route_and_models(self):
        self.assertEqual(DEFAULT_BASE_URL, "https://gen.pollinations.ai/v1")
        self.assertEqual(DEFAULT_MODEL, "community/YoannDev90/muse-glimmer-30b:free")
        self.assertEqual(DEFAULT_FALLBACK_MODEL, "openai/gpt-5.4-nano")

    def test_primary_401_falls_back_to_requested_model_over_post_only(self):
        fallback_body = json.dumps({
            "choices": [{"message": {"content": "Conheça o Magcubic HY300Pro+. Um projetor portátil para seu espaço. Aproveite o cinema em casa."}}]
        })
        responses = [self.response(401, '{"error":"primary denied"}'), self.response(200, fallback_body)]
        calls = []

        def fake_request(session, method, url, **kwargs):
            calls.append((method, url, kwargs))
            return responses.pop(0)

        audit = AuditLog()
        with patch.dict(os.environ, {"POLLINATIONS_API_KEY": ""}):
            with patch("shopee_short.llm.safe_request", side_effect=fake_request):
                script, meta = generate_script("MAGCUBIC HY300ProPlus Projector", audit=audit)
        self.assertIn("HY300Pro+", script)
        self.assertTrue(meta["fallback_used"])
        self.assertEqual(meta["model"], DEFAULT_FALLBACK_MODEL)
        self.assertEqual([call[0] for call in calls], ["POST", "POST"])
        self.assertTrue(all(call[1] == "https://gen.pollinations.ai/v1/chat/completions" for call in calls))
        self.assertEqual(calls[0][2]["json"]["model"], DEFAULT_MODEL)
        self.assertEqual(calls[1][2]["json"]["model"], DEFAULT_FALLBACK_MODEL)
        self.assertTrue(all("Authorization" not in call[2]["headers"] for call in calls))
        attempt_events = [e for e in audit.events if e["event"] == "pollinations_model_attempt"]
        self.assertEqual(attempt_events[0]["body"], '{"error":"primary denied"}')

    def test_double_401_preserves_exact_status_and_response_body(self):
        primary_body = '{"success":false,"error":{"message":"primary denied","code":"UNAUTHORIZED"},"status":401}'
        fallback_body = '{"success":false,"error":{"message":"fallback denied","code":"UNAUTHORIZED"},"status":401}'
        responses = [self.response(401, primary_body), self.response(401, fallback_body)]

        def fake_request(session, method, url, **kwargs):
            self.assertEqual(method, "POST")
            self.assertEqual(url, "https://gen.pollinations.ai/v1/chat/completions")
            return responses.pop(0)

        audit = AuditLog()
        with patch.dict(os.environ, {"POLLINATIONS_API_KEY": ""}):
            with patch("shopee_short.llm.safe_request", side_effect=fake_request):
                with self.assertRaises(ProviderError):
                    generate_script("MAGCUBIC HY300ProPlus Projector", audit=audit)
        attempts = [e for e in audit.events if e["event"] == "pollinations_model_attempt"]
        self.assertEqual([(a["status"], a["body"]) for a in attempts], [(401, primary_body), (401, fallback_body)])
        failed = [e for e in audit.events if e["event"] == "pollinations_generation_failed"]
        self.assertEqual(failed[-1]["attempts"][1]["body"], fallback_body)


if __name__ == "__main__":
    unittest.main()
