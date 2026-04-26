"""Entry point for `python -m litreviewrag`.

Delegates to the typer-based CLI defined in cli.py.
"""

from litreviewrag.cli import app

if __name__ == "__main__":
    app()