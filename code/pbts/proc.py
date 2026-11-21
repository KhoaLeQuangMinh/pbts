"""Is a recorded process still the one we started? (Linux /proc; used by the run lock and the local job queue.)

A pid alone is not enough: after a reboot (blackwell, 2026-11-21) the same pid can belong to an unrelated process,
which would make a dead run look alive, so it would never be resumed. A recorded process therefore counts as alive
only if a process with that pid exists, is not a zombie, and STARTED BEFORE the record was written (the file's mtime).
Every record here is written after its process started: the queue starts the job and then writes its file; a run
writes its lock itself.
"""
import os

SLACK_S = 5.0   # /proc/stat btime is whole seconds and can jitter by about a second


def start_time(pid):
    """Start time (epoch seconds) of a live, non-zombie process on this machine, or None."""
    try:
        fields = open(f"/proc/{int(pid)}/stat").read().rsplit(")", 1)[1].split()
        if fields[0] == "Z":                                  # field 3: state
            return None
        ticks = int(fields[19])                               # field 22: starttime, clock ticks after boot
        btime = next(int(l.split()[1]) for l in open("/proc/stat") if l.startswith("btime"))
        return btime + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, StopIteration, TypeError):
        return None


def alive(pid, recorded_at=None):
    """True if `pid` runs on this machine and (when `recorded_at`, epoch seconds, is given) started before then."""
    t = start_time(pid)
    return t is not None and (recorded_at is None or t <= recorded_at + SLACK_S)
