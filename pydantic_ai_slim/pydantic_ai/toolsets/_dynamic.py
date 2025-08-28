from __future__ import annotations

import inspect
from collections.abc import Awaitable
from typing import Any, Callable, Union

from typing_extensions import Self, TypeAlias

from .._run_context import AgentDepsT, RunContext
from .abstract import AbstractToolset, ToolsetTool

ToolsetFunc: TypeAlias = Callable[
    [RunContext[AgentDepsT]],
    Union[AbstractToolset[AgentDepsT], None, Awaitable[Union[AbstractToolset[AgentDepsT], None]]],
]
"""A sync/async function which takes a run context and returns a toolset."""


class DynamicToolset(AbstractToolset[AgentDepsT]):
    """A toolset that dynamically builds a toolset using a function that takes the run context.

    It should only be used during a single agent run as it stores the generated toolset.
    To use it multiple times, copy it using `dataclasses.replace`.
    """

    toolset_func: ToolsetFunc[AgentDepsT]
    per_run_step: bool
    _id: str | None

    _toolset: AbstractToolset[AgentDepsT] | None
    _run_step: int | None

    def __init__(self, toolset_func: ToolsetFunc[AgentDepsT], per_run_step: bool = True, id: str | None = None):
        self.toolset_func = toolset_func
        self.per_run_step = per_run_step
        self._id = id

        self._toolset = None
        self._run_step = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DynamicToolset):
            return False
        return (
            self.toolset_func == other.toolset_func  # pyright: ignore[reportUnknownMemberType]
            and self.per_run_step == other.per_run_step
            and self._id == other._id
            and self._toolset == other._toolset  # pyright: ignore[reportUnknownMemberType]
            and self._run_step == other._run_step
        )

    @property
    def id(self) -> str | None:
        return self._id

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        try:
            if self._toolset is not None:
                return await self._toolset.__aexit__(*args)
        finally:
            self._toolset = None
            self._run_step = None

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        if self._toolset is None or (self.per_run_step and ctx.run_step != self._run_step):
            if self._toolset is not None:
                await self._toolset.__aexit__()

            toolset = self.toolset_func(ctx)
            if inspect.isawaitable(toolset):
                toolset = await toolset

            if toolset is not None:
                await toolset.__aenter__()

            self._toolset = toolset
            self._run_step = ctx.run_step

        if self._toolset is None:
            return {}

        return await self._toolset.get_tools(ctx)

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        assert self._toolset is not None
        return await self._toolset.call_tool(name, tool_args, ctx, tool)

    def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
        if self._toolset is None:
            super().apply(visitor)
        else:
            self._toolset.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        if self._toolset is None:
            return super().visit_and_replace(visitor)
        else:
            # Create a new instance with the same config but visited _toolset
            new_instance = DynamicToolset(self.toolset_func, self.per_run_step, self._id)
            new_instance._toolset = self._toolset.visit_and_replace(visitor)
            new_instance._run_step = self._run_step
            return new_instance
