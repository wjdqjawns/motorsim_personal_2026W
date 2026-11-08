"""Command-line interface:  py -3.13 run.py <command> -c config/xxx.yaml [...]"""
from __future__ import annotations

import argparse

from motorsim.pipeline import executor  # noqa: F401  (registers the commands)
from motorsim.pipeline.registry import COMMANDS

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="motorsim", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in COMMANDS.values():
        p = sub.add_parser(cmd.name, help=cmd.help)

        if cmd.needs_config:
            p.add_argument("-c", "--config", required=True)

        p.add_argument("-o", "--out", default=None)

        for flags, kwargs in cmd.args:
            p.add_argument(*flags, **kwargs)

        p.set_defaults(fn=cmd.fn)
    
    return ap

def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)