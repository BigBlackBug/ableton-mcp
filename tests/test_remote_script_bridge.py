"""Regression tests for the Live-side Remote Script's TCP surface.

Each test pins a defect this change fixes:

* the socket bound every interface, exposing an unauthenticated command
  surface to the whole network,
* each recv() chunk was decoded on its own, so a multi-byte UTF-8 character
  split across a chunk boundary desynchronised the connection,
* an incomplete command grew the buffer without bound,
* get_browser_tree never recursed and never reported a folder count,
* disconnect() left client sockets open, so after Live swapped documents the
  MCP server kept talking to the dead instance and its stale Song.

The script normally runs inside Live's interpreter, so ``_Framework`` is
stubbed and the class is instantiated without ``__init__`` (which would open a
socket). No Ableton, no network.
"""

import importlib.util
import json
import socket
import sys
import threading
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SCRIPT = REPO_ROOT / "AbletonMCP_Remote_Script" / "__init__.py"
BUNDLED_SCRIPT = (REPO_ROOT / "MCP_Server" / "bundled_ableton_remote_script"
                  / "AbletonMCP_init.py")


def _load_script_module(fresh=False):
    """Import the Live-side script with _Framework stubbed out.

    Each call re-executes the module, so ``fresh`` is only a readability hint
    at call sites that depend on module-level state being re-evaluated (the
    ABLETON_MCP_HOST lookup).
    """
    framework = types.ModuleType("_Framework")
    control_surface_mod = types.ModuleType("_Framework.ControlSurface")

    class _StubControlSurface(object):
        def __init__(self, *args, **kwargs):
            pass

        def log_message(self, *args, **kwargs):
            pass

        def show_message(self, *args, **kwargs):
            pass

    control_surface_mod.ControlSurface = _StubControlSurface
    framework.ControlSurface = control_surface_mod
    sys.modules.setdefault("_Framework", framework)
    sys.modules.setdefault("_Framework.ControlSurface", control_surface_mod)

    spec = importlib.util.spec_from_file_location(
        "ableton_mcp_remote_script_under_test", BUNDLED_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script_module():
    return _load_script_module()


def make_instance(script_module, application=None):
    inst = script_module.AbletonMCP.__new__(script_module.AbletonMCP)
    inst.application = lambda: application
    inst.log_message = lambda *a, **k: None
    inst.show_message = lambda *a, **k: None
    inst.client_sockets = []
    return inst


# ------------------------------------------------------------------ binding

def test_socket_binds_loopback_by_default(script_module):
    """The command surface is unauthenticated: it must not listen publicly
    unless the user opts in via ABLETON_MCP_HOST."""
    assert script_module.HOST == "127.0.0.1"


def test_bind_host_is_overridable_for_remote_setups(monkeypatch):
    """Running Live and the MCP server on separate machines stays possible,
    but only as a deliberate choice."""
    monkeypatch.setenv("ABLETON_MCP_HOST", "0.0.0.0")
    assert _load_script_module(fresh=True).HOST == "0.0.0.0"


def test_repo_and_bundled_copies_are_identical():
    """The installer ships the bundled copy; drift would install stale code."""
    assert REPO_SCRIPT.read_bytes() == BUNDLED_SCRIPT.read_bytes()


# ------------------------------------------------------------------ framing

class FakeClient(object):
    """Hands out prepared chunks, then EOF. Records what was written back."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.sent = []
        self.closed = False

    def settimeout(self, _timeout):
        pass

    def recv(self, _size):
        return self._chunks.pop(0) if self._chunks else b""

    def sendall(self, data):
        self.sent.append(data)

    def close(self):
        self.closed = True


def _serve(script_module, chunks):
    """Run one client session and decode whatever it wrote back."""
    inst = make_instance(script_module)
    inst.running = True
    received = []

    def _process(command):
        received.append(command)
        return {"status": "success", "result": {"echo": command.get("type")}}

    inst._process_command = _process
    client = FakeClient(chunks)
    inst._handle_client(client)
    replies = [json.loads(frame.decode("utf-8")) for frame in client.sent]
    return received, replies


def test_command_split_mid_utf8_character_still_parses(script_module):
    """Regression: decoding each chunk alone raised on the split, dropped the
    chunk, and wrote an error frame into a stream still awaiting the real
    response — leaving request and response permanently out of step.

    ensure_ascii=False is what puts raw multi-byte on the wire.
    """
    payload = json.dumps(
        {"type": "set_clip_name", "params": {"name": "café été"}},
        ensure_ascii=False,
    ).encode("utf-8")
    split = payload.index(b"\xc3") + 1  # between the two bytes of 'é'
    assert payload[:split].endswith(b"\xc3")

    received, replies = _serve(script_module, [payload[:split], payload[split:]])

    assert len(received) == 1
    assert received[0]["params"]["name"] == "café été"
    assert replies == [{"status": "success", "result": {"echo": "set_clip_name"}}]


def test_command_split_across_many_chunks_parses_once(script_module):
    payload = json.dumps({"type": "get_session_info", "params": {}}).encode("utf-8")
    chunks = [payload[i:i + 3] for i in range(0, len(payload), 3)]

    received, replies = _serve(script_module, chunks)

    assert len(received) == 1
    assert len(replies) == 1


def test_back_to_back_commands_each_get_one_reply(script_module):
    first = json.dumps({"type": "start_playback", "params": {}}).encode("utf-8")
    second = json.dumps({"type": "stop_playback", "params": {}}).encode("utf-8")

    received, replies = _serve(script_module, [first, second])

    assert [c["type"] for c in received] == ["start_playback", "stop_playback"]
    assert len(replies) == 2


def test_oversized_request_is_refused_instead_of_buffered_forever(
        script_module, monkeypatch):
    """A client that never completes a command must not grow the buffer."""
    monkeypatch.setattr(script_module, "MAX_REQUEST_BYTES", 1024)
    junk = b'{"type": "' + b"x" * 4096

    received, replies = _serve(script_module, [junk, junk])

    assert received == []
    assert len(replies) == 1
    assert replies[0]["status"] == "error"


# ------------------------------------------------------------- browser tree

class FakeBrowserItem(object):
    def __init__(self, name, children=(), loadable=False, uri=None):
        self.name = name
        self.children = list(children)
        self.is_device = False
        self.is_loadable = loadable
        self.uri = uri


class FakeBrowser(object):
    def __init__(self, instruments):
        self.instruments = instruments


class FakeApplication(object):
    def __init__(self, browser=None):
        self.browser = browser


def test_browser_tree_returns_children_and_a_truthful_folder_count(script_module):
    """Regression: process_item never recursed (children was always []) and
    total_folders was never returned, so the server's header always read
    'showing 0 folders'."""
    kits = FakeBrowserItem("Kits", [FakeBrowserItem("808 Core Kit", loadable=True)])
    browser = FakeBrowser(FakeBrowserItem("Instruments", [kits]))
    inst = make_instance(script_module, application=FakeApplication(browser=browser))

    tree = inst.get_browser_tree("instruments")

    assert tree["total_folders"] >= 2  # Instruments and Kits both have children
    category = tree["categories"][0]
    assert [child["name"] for child in category["children"]] == ["Kits"]
    # Depth is capped one level below the category, so Kits reports more.
    assert category["children"][0]["has_more"] is True


def test_browser_tree_caps_a_huge_folder_and_flags_it(script_module):
    many = [FakeBrowserItem("Preset %d" % i, loadable=True) for i in range(200)]
    browser = FakeBrowser(FakeBrowserItem("Instruments", many))
    inst = make_instance(script_module, application=FakeApplication(browser=browser))

    category = inst.get_browser_tree("instruments")["categories"][0]

    assert len(category["children"]) == 64
    assert category["has_more"] is True


# ------------------------------------------------------------------ disconnect

def _disconnectable(script_module, monkeypatch):
    """An instance whose disconnect() runs without Live or a listening server."""
    monkeypatch.setattr(script_module.ControlSurface, "disconnect",
                        lambda self: None, raising=False)
    inst = make_instance(script_module)
    inst.running = True
    inst.server = None
    inst.server_thread = None
    inst.client_threads = []
    inst._teardown_passive_listeners = lambda: None
    return inst


def test_disconnect_closes_client_so_server_reconnects(script_module, monkeypatch):
    """Regression: Live replaces the control surface when the user opens
    another set. The old instance left its client socket open, its handler
    stayed blocked in recv(), and the MCP server's liveness check kept that
    socket, so every later command ran against the closed set's Song
    ("did not match C++ signature: None(TPyHandle<ASong>)") until Live was
    restarted. Closing the socket gives the server an EOF to reconnect on.
    """
    inst = _disconnectable(script_module, monkeypatch)
    processed = []
    inst._process_command = lambda command: processed.append(command) or {}

    live_end, server_end = socket.socketpair()
    server_end.settimeout(5.0)
    inst.client_sockets.append(live_end)
    handler = threading.Thread(target=inst._handle_client, args=(live_end,))
    handler.daemon = True
    handler.start()

    inst.disconnect()

    assert server_end.recv(1) == b""  # the MCP server sees EOF
    handler.join(5.0)
    assert not handler.is_alive()
    assert inst.client_sockets == []
    assert processed == []
    server_end.close()


def test_command_reaching_a_disconnected_instance_is_not_run(script_module):
    """A command read after disconnect() must not touch the stale Song."""
    inst = make_instance(script_module)
    inst.running = True
    processed = []
    inst._process_command = lambda command: processed.append(command) or {}
    payload = json.dumps({"type": "get_session_info", "params": {}}).encode("utf-8")

    class DisconnectDuringRecv(FakeClient):
        def recv(self, size):
            inst.running = False
            return super(DisconnectDuringRecv, self).recv(size)

    client = DisconnectDuringRecv([payload])
    inst._handle_client(client)

    assert processed == []
    assert client.sent == []
    assert client.closed


def test_socket_closed_by_disconnect_is_not_reported_as_error(script_module):
    """disconnect() closing the socket makes recv() fail with EBADF; that is
    the intended shutdown, not an error to log or answer."""
    inst = make_instance(script_module)
    inst.running = True
    logged = []
    inst.log_message = logged.append

    class ClosedUnderneath(FakeClient):
        def recv(self, size):
            inst.running = False
            raise OSError(9, "Bad file descriptor")

    client = ClosedUnderneath([])
    inst._handle_client(client)

    assert client.sent == []
    assert not any("Error handling client data" in m for m in logged)
