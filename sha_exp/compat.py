"""Make the official `diffusha` code importable on a plain CPU machine.

The official repo (ripl/diffusion-for-shared-autonomy) targets a CUDA docker image
with TensorFlow / tf-agents / d4rl installed. For BlockPush evaluation none of these
are actually used at runtime, so we stub them out instead of installing them, and
patch the two places that hard-code CUDA (SAC `gpu=0`, `torch.load` without map_location).

Import this module before importing anything from `diffusha`.
"""
import importlib.abc
import importlib.machinery
import os
import sys
import types
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIFFUSHA_DIR = Path(os.environ.get("DIFFUSHA_DIR", REPO_ROOT / "third_party" / "diffusion-for-shared-autonomy"))
DATA_DIR = Path(os.environ.get("DIFFUSHA_DATA_DIR", DIFFUSHA_DIR / "data-dir"))

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
warnings.filterwarnings("ignore")

if not (DIFFUSHA_DIR / "diffusha").is_dir():
    raise RuntimeError(
        f"Official repo not found at {DIFFUSHA_DIR}. Run `bash setup.sh` first "
        "or set DIFFUSHA_DIR to your clone of ripl/diffusion-for-shared-autonomy."
    )
if str(DIFFUSHA_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFUSHA_DIR))
# The BlockPush env loads its URDF assets via paths relative to the official repo root.
# Scripts therefore resolve their own output paths to absolute before importing this module.
os.chdir(DIFFUSHA_DIR)


class _StubModule(types.ModuleType):
    """Any attribute access returns a sub-module (lowercase) or an empty class (CamelCase)."""

    __path__: list = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if name[0].islower():
            return _StubModule(f"{self.__name__}.{name}")
        return type(name, (), {})


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    roots = ("tensorflow", "tf_agents", "d4rl")

    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in self.roots:
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        pass


def _install_stubs():
    import importlib.util

    missing = []
    for root in _StubFinder.roots:
        try:
            found = importlib.util.find_spec(root) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(root)
    if missing:
        _StubFinder.roots = tuple(missing)
        sys.meta_path.insert(0, _StubFinder())


def _patch_sac_device():
    import pfrl
    import torch

    base = pfrl.agents.SoftActorCritic
    if getattr(base, "_sha_exp_patched", False):
        return

    class SoftActorCritic(base):
        _sha_exp_patched = True

        def __init__(self, *args, **kwargs):
            kwargs["gpu"] = 0 if torch.cuda.is_available() else None
            super().__init__(*args, **kwargs)

    pfrl.agents.SoftActorCritic = SoftActorCritic


_install_stubs()
_patch_sac_device()
