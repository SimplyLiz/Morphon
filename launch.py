#!/usr/bin/env python3
"""
Morphon local development launcher.

Usage:
    python3 launch.py

Starts the WASM web demo server. Keyboard shortcuts let you rebuild,
restart, or open the browser while the server is running.

Prerequisites: wasm-pack in PATH (for builds); web/pkg/ built.
"""

import os
import select
import shutil
import signal
import socket
import subprocess
import sys
import termios
import time
import tty

# ── Constants ────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(ROOT, "web")
DEFAULT_PORT = 8080

# ── State ────────────────────────────────────────────────────────────────────

port = None
processes = []
original_terminal_settings = None


# ── Colors ───────────────────────────────────────────────────────────────────

class Colors:
    HEADER = '\033[95m'
    BLUE   = '\033[94m'
    CYAN   = '\033[96m'
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    END    = '\033[1m\033[0m'
    BOLD   = '\033[1m'
    DIM    = '\033[2m'


def print_section(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}▶ {title}{Colors.END}")


def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.END}")


def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.END}")


def print_info(msg):
    print(f"{Colors.YELLOW}ℹ {msg}{Colors.END}")


# ── Terminal ─────────────────────────────────────────────────────────────────

def setup_terminal():
    global original_terminal_settings
    if sys.stdin.isatty():
        original_terminal_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())


def restore_terminal():
    global original_terminal_settings
    if original_terminal_settings and sys.stdin.isatty():
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_terminal_settings)


# ── Helpers ──────────────────────────────────────────────────────────────────

def is_port_in_use(p):
    for family, addr in [(socket.AF_INET, '127.0.0.1'), (socket.AF_INET6, '::1')]:
        try:
            with socket.socket(family, socket.SOCK_STREAM) as s:
                if s.connect_ex((addr, p)) == 0:
                    return True
        except OSError:
            continue
    return False


def find_free_port(start, count=20):
    for p in range(start, start + count):
        if not is_port_in_use(p):
            return p
    return None


def wait_for_server(p, timeout=15):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_port_in_use(p):
            return True
        time.sleep(0.3)
    return False


def check_prereqs():
    issues = []
    if not os.path.exists(os.path.join(WEB_DIR, "pkg")):
        issues.append("web/pkg/ not found — run B to build WASM first")
    if issues:
        print_section("Warnings")
        for issue in issues:
            print_info(issue)
    return True  # warnings only, don't block launch


def open_browser(url):
    try:
        applescript = f'''
        tell application "System Events"
            set chromeRunning to (name of processes) contains "Google Chrome"
            set safariRunning to (name of processes) contains "Safari"
            if chromeRunning then
                tell application "Google Chrome"
                    set found to false
                    repeat with w in windows
                        set tabIndex to 1
                        repeat with t in tabs of w
                            if URL of t contains "{url}" then
                                set active tab index of w to tabIndex
                                set index of w to 1
                                activate
                                set found to true
                                exit repeat
                            end if
                            set tabIndex to tabIndex + 1
                        end repeat
                        if found then exit repeat
                    end repeat
                    if not found then
                        open location "{url}"
                        activate
                    end if
                end tell
            else if safariRunning then
                tell application "Safari"
                    set found to false
                    repeat with w in windows
                        repeat with t in tabs of w
                            if URL of t contains "{url}" then
                                set current tab of w to t
                                set index of w to 1
                                activate
                                set found to true
                                exit repeat
                            end if
                        end repeat
                        if found then exit repeat
                    end repeat
                    if not found then
                        open location "{url}"
                        activate
                    end if
                end tell
            else
                do shell script "open {url}"
            end if
        end tell
        '''
        subprocess.run(['osascript', '-e', applescript], capture_output=True, timeout=5)
        print_success("Browser opened")
    except Exception as e:
        print_info(f"Could not open browser: {e}")


# ── Server ───────────────────────────────────────────────────────────────────

def start_server():
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c",
             f"import os, http.server\n"
             f"class H(http.server.SimpleHTTPRequestHandler):\n"
             f"  def end_headers(self):\n"
             f"    self.send_header('Cache-Control','no-store')\n"
             f"    super().end_headers()\n"
             f"  def log_message(self, *a): pass\n"
             f"os.chdir(r'{WEB_DIR}')\n"
             f"http.server.HTTPServer(('',{port}),H).serve_forever()\n"
            ],
            cwd=WEB_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        processes.append(("Web Server", proc))
        return proc
    except Exception as e:
        print_error(f"Failed to start server: {e}")
        return None


def stop_all():
    for name, proc in processes:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    processes.clear()


def cleanup_and_exit():
    print(f"\n{Colors.YELLOW}Shutting down...{Colors.END}")
    stop_all()
    print_success("Stopped")


def signal_handler(sig, frame):
    restore_terminal()
    cleanup_and_exit()
    sys.exit(0)


# ── Build ────────────────────────────────────────────────────────────────────

def build_wasm():
    print(f"\n{Colors.BOLD}{Colors.YELLOW}▶ Building WASM{Colors.END}")
    restore_terminal()
    result = subprocess.run(
        ["bash", "web/build.sh"],
        cwd=ROOT,
    )
    setup_terminal()
    if result.returncode == 0:
        print_success("WASM build complete")
    else:
        print_error("WASM build failed — check output above")
    print_running()


# ── UI ───────────────────────────────────────────────────────────────────────

def print_banner():
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}")
    print("  Morphon — Morphogenic Intelligence Engine")
    print(f"{'='*60}{Colors.END}\n")


def print_running():
    print_section("Running")
    print(f"\n{Colors.BOLD}WASM Demo:{Colors.END}")
    print(f"  {Colors.CYAN}http://localhost:{port}{Colors.END}")
    print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}  {Colors.CYAN}R{Colors.END}{Colors.BOLD}      Restart server{Colors.END}")
    print(f"{Colors.BOLD}  {Colors.CYAN}B{Colors.END}{Colors.BOLD}      Build WASM (wasm-pack){Colors.END}")
    print(f"{Colors.BOLD}  {Colors.CYAN}O{Colors.END}{Colors.BOLD}      Open in browser{Colors.END}")
    print(f"{Colors.BOLD}  {Colors.DIM}Cmd+C{Colors.END}{Colors.BOLD}  Stop{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'='*60}{Colors.END}\n")


def relaunch():
    print(f"\n{Colors.BOLD}{Colors.YELLOW}▶ Restarting server{Colors.END}")
    stop_all()
    proc = start_server()
    if proc and wait_for_server(port):
        print_success(f"Server restarted on port {port}")
    else:
        print_error("Failed to restart")
    print_running()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global port

    sys.stdout.reconfigure(line_buffering=True)
    os.chdir(ROOT)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print_banner()
    check_prereqs()

    print_section("Checking Port")
    p = find_free_port(DEFAULT_PORT)
    if not p:
        print_error(f"No free port near {DEFAULT_PORT}")
        sys.exit(1)
    port = p
    if port != DEFAULT_PORT:
        print_info(f"Port {DEFAULT_PORT} in use, using {port}")
    else:
        print_success(f"Port {port} available")

    print_section("Starting Server")
    proc = start_server()
    if not proc:
        sys.exit(1)

    if wait_for_server(port):
        print_success(f"Server ready on port {port}")
    else:
        print_error(f"Server did not come up on port {port}")
        cleanup_and_exit()
        sys.exit(1)

    print_running()
    setup_terminal()

    try:
        while True:
            for name, p in list(processes):
                if p.poll() is not None:
                    print_error(f"{name} exited unexpectedly (code {p.returncode})")
                    restore_terminal()
                    cleanup_and_exit()
                    return

            if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0.5)[0]:
                key = sys.stdin.read(1)
                if key.lower() == 'r':
                    relaunch()
                elif key.lower() == 'b':
                    build_wasm()
                elif key.lower() == 'o':
                    open_browser(f"http://localhost:{port}")
    except KeyboardInterrupt:
        pass
    finally:
        restore_terminal()
        cleanup_and_exit()


if __name__ == "__main__":
    main()
