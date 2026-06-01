import datetime
import os
import tempfile
import unittest
from unittest.mock import patch

import ai_summary


def create_config(**overrides):
    values = {
        "base_url": "https://api.example.com/v1",
        "api_key": "sk-test",
        "model": "paper-model",
        "prompt": "1. solid electrolytes\n2. catalysts",
        "interval_hours": 24,
        "max_candidates": 100,
        "max_output_tokens": 4096,
        "screening_batch_size": 10,
        "requests_per_minute": 5,
        "max_prompt_title_chars": 240,
        "max_prompt_abstract_chars": 1200,
        "retry_attempts_per_round": 3,
        "retry_rounds": 2,
        "retry_sleep_seconds": 600,
    }
    values.update(overrides)
    return ai_summary.AiSummaryConfig(**values)


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, messages):
        self.calls.append(messages[-1]["content"])
        if isinstance(self.responses[0], Exception):
            raise self.responses.pop(0)
        return self.responses.pop(0)


class AlwaysFailClient:
    def __init__(self):
        self.calls = 0

    def complete(self, _messages):
        self.calls += 1
        raise RuntimeError("HTTP 403")


class FinalHtmlFailClient:
    def __init__(self):
        self.calls = []

    def complete(self, messages):
        content = messages[-1]["content"]
        self.calls.append(content)
        if "Paper Batch" in content:
            return '[{"id":1,"matched_direction":"solid electrolytes","importance":"high","summary":"本文相关。"}]'
        raise RuntimeError("HTTP 524")


class AiSummaryTest(unittest.TestCase):
    def test_load_ai_config_skips_when_required_env_is_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(ai_summary.load_ai_config())

    def test_load_ai_config_reads_multiline_api_secret(self):
        with patch.dict(
            os.environ,
            {
                "AI_API_CONFIG": "https://api.example.com/v1\nsk-test\npaper-model",
                "AI_SUMMARY_PROMPT": "1. solid electrolytes",
            },
            clear=True,
        ):
            config = ai_summary.load_ai_config()

        self.assertIsNotNone(config)
        self.assertEqual(config.base_url, "https://api.example.com/v1")
        self.assertEqual(config.api_key, "sk-test")
        self.assertEqual(config.model, "paper-model")
        self.assertEqual(config.requests_per_minute, 5)
        self.assertEqual(config.screening_batch_size, 10)

    def test_load_ai_config_reads_public_config_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with open(ai_summary.CONFIG_FILE, "w", encoding="utf-8") as handle:
                    handle.write(
                        '{"ai_summary":{"interval_hours":12,"max_candidates":30,'
                        '"screening_batch_size":8,"requests_per_minute":3,'
                        '"max_output_tokens":2048,"retry_sleep_seconds":120}}'
                    )

                with patch.dict(
                    os.environ,
                    {
                        "AI_API_CONFIG": "https://api.example.com/v1\nsk-test\npaper-model",
                        "AI_SUMMARY_PROMPT": "1. solid electrolytes",
                    },
                    clear=True,
                ):
                    config = ai_summary.load_ai_config()

                self.assertEqual(config.interval_hours, 12)
                self.assertEqual(config.max_candidates, 30)
                self.assertEqual(config.screening_batch_size, 8)
                self.assertEqual(config.requests_per_minute, 3)
                self.assertEqual(config.max_output_tokens, 2048)
                self.assertEqual(config.retry_sleep_seconds, 120)
            finally:
                os.chdir(old_cwd)

    def test_generate_ai_summary_report_batches_then_wraps_final_html(self):
        papers = [
            {
                "id": "paper-1",
                "title": "Fast lithium conduction in solid electrolytes",
                "abstract": "This paper studies lithium ion transport.",
                "journal": "Advanced Materials",
                "url": "https://example.com/paper",
                "pubDate": "2026-05-22T00:00:00+00:00",
            }
        ]
        client = FakeClient(
            [
                '[{"id":1,"matched_direction":"solid electrolytes","importance":"high","summary":"本文研究固态电解质中的锂离子输运。"}]',
                "<section><h3>固态电解质</h3><p><strong>VASP</strong> summary.</p></section>",
            ]
        )

        report = ai_summary.generate_ai_summary_report(
            create_config(),
            papers,
            client,
            datetime.datetime(2026, 5, 22, 8, tzinfo=datetime.timezone.utc),
        )

        self.assertEqual(len(client.calls), 2)
        self.assertIn("Classified Paper Summaries", client.calls[1])
        self.assertEqual(report["matched_count"], 1)
        self.assertIn("Daily AI Literature Insights", report["html"])
        self.assertIn("<strong>VASP</strong>", report["html"])

    def test_run_ai_summary_writes_outputs_and_marks_candidates_submitted(self):
        feed_xml = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/"><channel>
<title>Paper Feed</title>
<item>
<title>Fast lithium conduction in solid electrolytes</title>
<link>https://example.com/paper</link>
<description>This paper studies lithium ion transport.</description>
<guid isPermaLink="false">paper-1</guid>
<pubDate>Fri, 22 May 2026 00:00:00 GMT</pubDate>
<dc:source>Advanced Materials</dc:source>
</item>
</channel></rss>"""

        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with open(ai_summary.INPUT_FEED_FILE, "w", encoding="utf-8") as handle:
                    handle.write(feed_xml)

                client = FakeClient(["[]"])
                with patch.dict(
                    os.environ,
                    {"GITHUB_REPOSITORY": "Jarvis-Towne/paper-feed"},
                    clear=False,
                ):
                    changed = ai_summary.run_ai_summary(
                        create_config(),
                        client,
                        datetime.datetime(2026, 5, 22, 8, tzinfo=datetime.timezone.utc),
                    )

                self.assertTrue(changed)
                self.assertTrue(os.path.exists(ai_summary.OUTPUT_FEED_FILE))
                self.assertTrue(os.path.exists(ai_summary.OUTPUT_HTML_FILE))
                with open(ai_summary.OUTPUT_FEED_FILE, "r", encoding="utf-8") as handle:
                    ai_feed_xml = handle.read()
                self.assertIn(
                    "https://Jarvis-Towne.github.io/paper-feed/ai_summary.html",
                    ai_feed_xml,
                )
                state = ai_summary.read_state()
                self.assertIn("paper-1", state["submitted_ids"])
                self.assertEqual(state["last_success_at"], "2026-05-22T08:00:00Z")
            finally:
                os.chdir(old_cwd)

    def test_run_ai_summary_failure_does_not_mark_candidates_submitted(self):
        feed_xml = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/"><channel>
<title>Paper Feed</title>
<item>
<title>Fast lithium conduction in solid electrolytes</title>
<link>https://example.com/paper</link>
<description>This paper studies lithium ion transport.</description>
<guid isPermaLink="false">paper-1</guid>
<pubDate>Fri, 22 May 2026 00:00:00 GMT</pubDate>
<dc:source>Advanced Materials</dc:source>
</item>
</channel></rss>"""

        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                with open(ai_summary.INPUT_FEED_FILE, "w", encoding="utf-8") as handle:
                    handle.write(feed_xml)

                client = AlwaysFailClient()
                sleeps = []
                changed = ai_summary.run_ai_summary(
                    create_config(),
                    client,
                    datetime.datetime(2026, 5, 22, 8, tzinfo=datetime.timezone.utc),
                    sleep_fn=sleeps.append,
                )

                self.assertFalse(changed)
                self.assertEqual(client.calls, 6)
                self.assertIn(600, sleeps)
                self.assertIn(12.0, sleeps)
                self.assertFalse(os.path.exists(ai_summary.STATE_FILE))
                self.assertFalse(os.path.exists(ai_summary.OUTPUT_FEED_FILE))
            finally:
                os.chdir(old_cwd)

    def test_final_html_retry_does_not_rerun_successful_screening_batch(self):
        client = FinalHtmlFailClient()
        sleeps = []

        with self.assertRaisesRegex(RuntimeError, "final HTML failed after 6 attempts"):
            ai_summary.generate_ai_summary_report(
                create_config(),
                [
                    {
                        "id": "paper-1",
                        "title": "Fast lithium conduction in solid electrolytes",
                        "abstract": "This paper studies lithium ion transport.",
                        "journal": "Advanced Materials",
                        "url": "https://example.com/paper",
                        "pubDate": "2026-05-22T00:00:00+00:00",
                    }
                ],
                client,
                datetime.datetime(2026, 5, 22, 8, tzinfo=datetime.timezone.utc),
                sleep_fn=sleeps.append,
            )

        screening_calls = [call for call in client.calls if "Paper Batch" in call]
        final_calls = [call for call in client.calls if "Classified Paper Summaries" in call]
        self.assertEqual(len(screening_calls), 1)
        self.assertEqual(len(final_calls), 6)
        self.assertIn(600, sleeps)
        self.assertIn(12.0, sleeps)


if __name__ == "__main__":
    unittest.main()
