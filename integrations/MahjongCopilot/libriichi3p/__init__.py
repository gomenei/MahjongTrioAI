""" Import libriichi3p module """
import sys
import os
import platform
import importlib.util
from pathlib import Path

assert sys.version_info >= (3, 10), "Python version must be 3.10 or higher"
assert sys.version_info <= (3, 12), "Python version must be 3.12 or lower"

def load_module():
    """ Determine system specifics and load the appropriate module file"""
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    if platform.processor() == "arm":
        proc_str = "aarch64"
    else:
        proc_str = "x86_64"

    if platform.system() == "Windows":
        os_ext_str = "pc-windows-msvc.pyd"
    elif platform.system() == "Darwin":
        os_ext_str = "apple-darwin.so"
    elif platform.system() == "Linux":
        os_ext_str = "unknown-linux-gnu.so"
    else:
        raise EnvironmentError(f"Unsupported OS: {platform.system()}")

    # Resolve relative to this package rather than the process working
    # directory.  This allows tools outside MahjongCopilot's root directory to
    # load the native three-player library reliably.
    filename = f"libriichi3p-{python_version}-{proc_str}-{os_ext_str}"
    file_path = Path(__file__).resolve().parent / filename
    if not file_path.is_file():
        raise ImportError(f"Could not find file: {file_path}")
    
    # Attempt to load the .pyd file
    spec = importlib.util.spec_from_file_location("libriichi3p", str(file_path))
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    else:
        raise ImportError(f"Could not import: {file_path}")
    
libriichi3p = load_module()

# Re-export the native extension's public objects while retaining this module
# as a package.  In particular, engine3p needs ``mjai`` and ``consts``.
for _name in dir(libriichi3p):
    if not _name.startswith("_"):
        globals()[_name] = getattr(libriichi3p, _name)

__doc__ = libriichi3p.__doc__
if hasattr(libriichi3p, "__all__"):
    __all__ = libriichi3p.__all__
