#!/usr/bin/env python3
"""Daemonize a command via double-fork.

Processes spawned directly by bash tool sessions are reaped when the session
ends. A double-forked, session-detached daemon (reparented to init) survives.
Usage: daemonize.py <logfile> <cmd> [args...]
Env: DAEMONIZE_CWD — working directory for the command.
"""
import os
import sys


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: daemonize.py <logfile> <cmd> [args...]", file=sys.stderr)
        sys.exit(1)
    logfile = os.path.abspath(sys.argv[1])
    cmd = sys.argv[2:]
    cwd = os.environ.get("DAEMONIZE_CWD", os.getcwd())

    pid = os.fork()
    if pid == 0:
        try:
            os.setsid()
        except OSError:
            pass
        pid2 = os.fork()
        if pid2 == 0:
            try:
                os.umask(0o022)
                os.chdir(cwd)
                devnull = os.open("/dev/null", os.O_RDONLY)
                os.dup2(devnull, 0)
                lf = os.open(logfile, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
                os.dup2(lf, 1)
                os.dup2(lf, 2)
                if lf > 2:
                    os.close(lf)
                os.execvp(cmd[0], cmd)
            except Exception as e:  # noqa: BLE001
                print(f"daemonize exec failed: {e}", file=sys.stderr)
                os._exit(127)
        os._exit(0)
    os.waitpid(pid, 0)
    print(f"daemonized: {' '.join(cmd)} | log={logfile} | cwd={cwd}")


if __name__ == "__main__":
    main()
