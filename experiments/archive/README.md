# Archived experiments

These scripts had no whole-word module-name mention in any other Git-tracked
text file on 2026-09-26, including documentation and results. Their own content
was ignored; private names followed the same rule, and __init__.py was excluded.

They are unmaintained and are not covered by CI. Some need optional dependencies
or historical datasets; some run computations even when passed --help.

To restore a script, use `git mv experiments/archive/<name>.py experiments/`
and undo the extra parent-directory level in its archive path setup.

Regenerate the current top-level orphan list from the repository root:

```bash
python scripts/maint/find_orphan_experiments.py
```

The scanner reads tracked paths in the working tree. Until archive moves are
staged, references in the new, untracked archive paths are not counted. After
staging, those references retain the scripts they mention. This command lists
current candidates, not the historical contents of this archive.
