"""Command registry: executor functions register themselves; the CLI builds its parser from here."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

COMMANDS: dict[str, "Command"] = {}


@dataclass
class Command:
    name: str
    fn: Callable
    help: str
    needs_config: bool = True
    args: list = field(default_factory=list)      # [(flags tuple, kwargs dict), ...]


def command(name: str, help: str, needs_config: bool = True, args=()):
    def deco(fn):
        COMMANDS[name] = Command(name, fn, help, needs_config, list(args))
        return fn
    return deco


def arg(*flags, **kwargs):
    return (flags, kwargs)
