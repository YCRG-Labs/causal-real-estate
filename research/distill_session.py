"""Distill a Claude Code session .jsonl into a compact handoff.

Keeps the signal (user messages, assistant prose, commits, files written)
and drops the noise (tool inputs + tool_result payloads, which are ~95% of
the bytes in a long coding session). Reads the transcript line by line so it
never holds the whole 48MB in memory.
"""
import json
import re
import sys
from pathlib import Path

src = Path(sys.argv[1])
out = Path(sys.argv[2])

users, assistant_turns, commits, writes = [], [], [], []
COMMIT_RE = re.compile(r"\[main [0-9a-f]{7,}\][^\n]*|shipped as [0-9a-f]{7}[^\n]*")


def texts(content):
    """Yield only human/assistant prose text blocks, never tool payloads."""
    if isinstance(content, str):
        yield content
    elif isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                yield b["text"]


with src.open() as f:
    for line in f:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        msg = rec.get("message") or {}
        role = msg.get("role") or rec.get("type")
        content = msg.get("content", rec.get("content"))
        if content is None:
            continue
        # Skip tool_result messages (role user but content is tool output).
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        ):
            continue
        blob = "\n".join(t for t in texts(content)).strip()
        # Mine tool_use inputs only for file writes (high-signal, tiny).
        if isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    name = b.get("name", "")
                    inp = b.get("input", {}) or {}
                    if name in ("Write", "Edit", "NotebookEdit") and inp.get("file_path"):
                        writes.append(inp["file_path"])
        if not blob:
            continue
        if role == "user":
            # Drop harness/system-reminder noise and giant pastes.
            if blob.startswith("<") or "system-reminder" in blob[:120]:
                continue
            users.append(blob)
        elif role == "assistant":
            for m in COMMIT_RE.findall(blob):
                commits.append(m.strip())
            assistant_turns.append(blob[:600])

seen = set()
commits = [c for c in commits if not (c in seen or seen.add(c))]
seen = set()
writes = [w for w in writes if not (w in seen or seen.add(w))]

lines = []
lines.append("# Session handoff — distilled from %s\n" % src.name)
lines.append("Source: %d-byte transcript. %d user msgs, %d assistant turns, "
             "%d commits, %d files written.\n"
             % (src.stat().st_size, len(users), len(assistant_turns),
                len(commits), len(writes)))

lines.append("\n## Every message you typed (verbatim, in order)\n")
for i, u in enumerate(users, 1):
    snip = u if len(u) <= 800 else u[:800] + " …[truncated]"
    lines.append("%d. %s" % (i, snip.replace("\n", " ")))

lines.append("\n## Commits shipped (in order)\n")
for c in commits:
    lines.append("- %s" % c)

lines.append("\n## Files written / edited\n")
for w in writes:
    lines.append("- %s" % w)

out.write_text("\n".join(lines))
print("WROTE", out, "%.1f KB" % (out.stat().st_size / 1024))
print("users=%d turns=%d commits=%d writes=%d"
      % (len(users), len(assistant_turns), len(commits), len(writes)))
