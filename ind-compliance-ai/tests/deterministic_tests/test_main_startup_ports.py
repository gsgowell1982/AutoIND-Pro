from __future__ import annotations

import unittest
from unittest.mock import patch

import main


class MainStartupPortTests(unittest.TestCase):
    def test_resolve_startup_ports_moves_api_and_ui_when_requested_ports_are_reachable(self) -> None:
        occupied = {("localhost", 8000), ("localhost", 8001), ("localhost", 5173)}

        with patch.object(main, "_port_is_reachable", side_effect=lambda host, port: (host, port) in occupied):
            resolved_api_port, resolved_ui_port = main._resolve_startup_ports(
                api_host="0.0.0.0",
                requested_api_port=8000,
                requested_ui_port=5173,
                auto_port=True,
            )

        self.assertEqual(resolved_api_port, 8002)
        self.assertEqual(resolved_ui_port, 5174)

    def test_resolve_startup_ports_keeps_requested_ports_when_auto_port_disabled(self) -> None:
        with patch.object(main, "_port_is_reachable", return_value=True):
            resolved_api_port, resolved_ui_port = main._resolve_startup_ports(
                api_host="0.0.0.0",
                requested_api_port=8000,
                requested_ui_port=5173,
                auto_port=False,
            )

        self.assertEqual(resolved_api_port, 8000)
        self.assertEqual(resolved_ui_port, 5173)

    def test_api_client_host_uses_localhost_for_wildcard_bind_address(self) -> None:
        self.assertEqual(main._api_client_host("0.0.0.0"), "localhost")
        self.assertEqual(main._api_client_host("127.0.0.1"), "127.0.0.1")


if __name__ == "__main__":
    unittest.main()
