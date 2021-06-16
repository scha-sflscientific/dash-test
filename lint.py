import sys
import shlex
import subprocess

def call(cmd=["string", "string"], **kwargs):  # pragma: no cover
    """Run a subprocess command and exit if it fails."""
    print(" ".join(shlex.quote(c) for c in cmd))
    # pylint: disable=subprocess-run-check
    res = subprocess.run(cmd, **kwargs).returncode
    if res:
        sys.exit(res)


def python_call(
    module="", arguments=["string", "string"], **kwargs
):  # pragma: no cover
    """Run a subprocess command that invokes a Python module."""
    call([sys.executable, "-m", module] + list(arguments), **kwargs)


def lint(files=("src", "test")):
    """Run flake8, isort and (on Python >=3.6) black."""
    # pylint: disable=unused-import
    if not files:
        files = ("src", "test")

    try:
        import flake8
        import isort
    except ImportError as exc:
        raise KedroCliError(NO_DEPENDENCY_MESSAGE.format(exc.name))

    python_call("flake8", ("--max-line-length=88",) + files)
    python_call("isort", ("-rc", "-tc", "-up", "-fgw=0", "-m=3", "-w=88") + files)

    if sys.version_info[:2] >= (3, 6):
        try:
            import black
        except ImportError:
            raise KedroCliError(NO_DEPENDENCY_MESSAGE.format("black"))
        python_call("black", files)


lint()
