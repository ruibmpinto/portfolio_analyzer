"""Launch the dashboard sidecar + Vite dev server.

Spawns the FastAPI sidecar on port 8000 (matches the
VITE_DASHBOARD_API_PORT default in dashboard/.env), then runs
``npm run dev`` in the foreground. Sidecar is terminated when
Vite exits or Ctrl+C is pressed. Stale processes holding the
required ports are killed before launch.

Run from the project root:
    python -m src.user_scripts.run_dashboard_dev
"""

import os
import pathlib
import shutil
import signal
import subprocess
import sys


api_port = 8000
vite_port = 5173
root_dir = pathlib.Path(__file__).resolve().parents[2]


def _clear_pyc(base: pathlib.Path):
    """Remove every __pycache__ under `base`; stale bytecode bites."""
    for path in base.rglob('__pycache__'):
        shutil.rmtree(path, ignore_errors=True)


def _free_port(port):
    """Kill anything listening on `port` so binding succeeds."""
    try:
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True, text=True, check=False)
    except FileNotFoundError:
        # lsof missing -> skip; the server will fail loudly
        return
    for pid in result.stdout.split():
        try:
            os.kill(int(pid), signal.SIGTERM)
        except (ProcessLookupError, ValueError, PermissionError):
            pass


def main():
    """Spawn sidecar, run Vite in foreground, clean up on exit."""
    _clear_pyc(root_dir / 'src')
    _free_port(api_port)
    _free_port(vite_port)

    env = os.environ.copy()
    env['DASHBOARD_PORT'] = str(api_port)
    sidecar = subprocess.Popen(
        [sys.executable, '-m', 'src.dashboard_api.server'],
        cwd=str(root_dir), env=env)

    print(f'  sidecar  : http://127.0.0.1:{api_port}/api/health')
    print(f'  frontend : http://localhost:{vite_port}')
    print()

    try:
        subprocess.run(
            ['npm', 'run', 'dev', '--', '--open'],
            cwd=str(root_dir / 'dashboard'))
    finally:
        sidecar.terminate()
        sidecar.wait()


if __name__ == '__main__':
    main()
