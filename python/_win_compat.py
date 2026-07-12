"""Windows compatibility patches applied process-wide at engine boot.

Currently:
- os.symlink → transparent copy fallback when the caller lacks
  SeCreateSymbolicLinkPrivilege (the WinError 1314 case). HuggingFace's
  cache layout uses symlinks to dedup blobs across snapshots, and a
  standard non-admin user without Developer Mode hits this on every
  download. POSIX users keep the real os.symlink (symlinks always work
  there).

Import this module before any other library that may call os.symlink.
On non-Windows platforms the import is a no-op.
"""
from __future__ import annotations

import os
import shutil
import sys


def _install_symlink_copy_fallback() -> None:
    if sys.platform != "win32":
        return

    _orig = os.symlink

    def _resolve(src, dst):
        if os.path.isabs(src):
            return src
        return os.path.normpath(os.path.join(os.path.dirname(dst), src))

    def _symlink(src, dst, target_is_directory=False, *, dir_fd=None):
        try:
            return _orig(src, dst, target_is_directory=target_is_directory, dir_fd=dir_fd)
        except OSError as e:
            is_privilege_error = (
                getattr(e, "winerror", None) == 1314
                or "privilege" in str(e).lower()
            )
            if not is_privilege_error:
                raise
            real_src = _resolve(src, dst)
            if target_is_directory or os.path.isdir(real_src):
                shutil.copytree(real_src, dst, dirs_exist_ok=True)
            else:
                shutil.copyfile(real_src, dst)
            return None

    os.symlink = _symlink


_install_symlink_copy_fallback()
