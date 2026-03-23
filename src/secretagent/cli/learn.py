from pathlib import Path
from typing import Optional

import typer

from secretagent.learn.baselines import RoteLearner

app = typer.Typer()
_EXTRA_ARGS = {"allow_extra_args": True, "allow_interspersed_args": False}

@app.callback()
def main():
    """Learn implementations from recorded interface calls."""

@app.command(context_settings=_EXTRA_ARGS)
def rote(
    ctx: typer.Context,
    interface: str = typer.Option(..., help="Interface name to extract, e.g. 'consistent_sports'"),
    latest: int = typer.Option(1, help='Keep latest k dirs per tag; 0 for all'),
    check: Optional[list[str]] = typer.Option(None, help='Config constraint like key=value'),
    train_dir: str = typer.Option('/tmp/rote_train', help='Directory to store collected data'),
):
    learner = RoteLearner(interface_name=interface, train_dir=train_dir)
    learner.learn([Path(a) for a in ctx.args], latest=latest, check=check)

if __name__ == '__main__':
    app()
