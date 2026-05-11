from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest

from llm.llm_client import LLMClient, LLMConfig


class _JsonResponder(BaseHTTPRequestHandler):
    response_payload: dict[str, object] = {}
    last_request: dict[str, object] | None = None

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(content_length)
        payload = json.loads(body.decode("utf-8"))
        type(self).last_request = {
            "path": self.path,
            "headers": dict(self.headers),
            "payload": payload,
        }
        response_bytes = json.dumps(type(self).response_payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response_bytes)))
        self.end_headers()
        self.wfile.write(response_bytes)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


class _ServerHarness:
    def __init__(self, response_payload: dict[str, object]) -> None:
        handler = type(
            "DynamicJsonResponder",
            (_JsonResponder,),
            {"response_payload": response_payload, "last_request": None},
        )
        self._handler = handler
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    @property
    def last_request(self) -> dict[str, object] | None:
        return self._handler.last_request

    def __enter__(self) -> "_ServerHarness":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class LLMClientTests(unittest.TestCase):
    def test_ollama_provider_calls_generate_endpoint_and_returns_response_text(self) -> None:
        with _ServerHarness({"response": "OLLAMA_OK"}) as harness:
            client = LLMClient(
                LLMConfig(
                    provider="ollama_cloud",
                    model="gpt-oss:120b-cloud",
                    base_url=harness.base_url,
                    temperature=0.1,
                )
            )

            result = client.generate("hello ollama", system_prompt="system")

            self.assertEqual(result, "OLLAMA_OK")
            request = harness.last_request
            self.assertIsNotNone(request)
            self.assertEqual(request["path"], "/api/generate")
            self.assertEqual(request["payload"]["model"], "gpt-oss:120b-cloud")
            self.assertEqual(request["payload"]["prompt"], "hello ollama")
            self.assertEqual(request["payload"]["system"], "system")
            self.assertEqual(request["payload"]["stream"], False)

    def test_custom_cloud_provider_calls_openai_compatible_chat_endpoint(self) -> None:
        with _ServerHarness(
            {
                "choices": [
                    {
                        "message": {
                            "content": "CUSTOM_CLOUD_OK",
                        }
                    }
                ]
            }
        ) as harness:
            client = LLMClient(
                LLMConfig(
                    provider="custom_cloud_llm",
                    model="example-cloud-model",
                    base_url=harness.base_url,
                    api_key="test-key",
                    temperature=0.3,
                )
            )

            result = client.generate("hello custom", system_prompt="system")

            self.assertEqual(result, "CUSTOM_CLOUD_OK")
            request = harness.last_request
            self.assertIsNotNone(request)
            self.assertEqual(request["path"], "/v1/chat/completions")
            self.assertEqual(request["payload"]["model"], "example-cloud-model")
            self.assertEqual(request["payload"]["messages"][0]["role"], "system")
            self.assertEqual(request["payload"]["messages"][1]["role"], "user")
            self.assertEqual(request["headers"]["Authorization"], "Bearer test-key")

    def test_default_ollama_base_url_is_localhost(self) -> None:
        config = LLMConfig(provider="ollama_self_hosted", model="qwen3:8b")
        client = LLMClient(config)
        self.assertEqual(client.resolved_base_url, "http://127.0.0.1:11434")


if __name__ == "__main__":
    unittest.main()
