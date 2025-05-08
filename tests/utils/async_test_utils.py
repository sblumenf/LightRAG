"""
Utilities for handling async tests properly.

This module provides utilities to properly handle async tests and prevent
warnings about pending tasks when the event loop is closed.
"""

import asyncio
import functools
import inspect
import pytest
import sys
from typing import Any, Callable, TypeVar, cast

T = TypeVar('T')


def cleanup_pending_tasks() -> None:
    """
    Clean up pending tasks in the current event loop.

    This function should be called at the end of async tests to ensure
    that all pending tasks are properly cancelled and cleaned up before
    the event loop is closed.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop
        return

    # Get all tasks associated with the current event loop
    current_task = asyncio.current_task(loop)
    pending_tasks = [task for task in asyncio.all_tasks(loop)
                    if not task.done() and task is not current_task]

    if not pending_tasks:
        return

    # Cancel all pending tasks
    for task in pending_tasks:
        task.cancel()

    # Create a future that will complete when all tasks are cancelled
    future = asyncio.gather(*pending_tasks, return_exceptions=True)

    # Run the event loop until the future completes
    try:
        loop.run_until_complete(future)
    except RuntimeError:
        # The event loop might be closed or running
        pass


def async_test(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator for async test functions that ensures proper cleanup of pending tasks.

    This decorator wraps an async test function to ensure that all pending tasks
    are properly cancelled and cleaned up before the event loop is closed.

    Args:
        func: The async test function to wrap.

    Returns:
        A wrapped function that handles proper cleanup of async resources.
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            # Clean up any pending tasks
            try:
                cleanup_pending_tasks()
            except RuntimeError:
                # If there's no running event loop, we can't clean up tasks
                pass

    return wrapper


@pytest.fixture
def event_loop():
    """
    Fixture that provides a new event loop for each test.

    This fixture creates a new event loop for each test and ensures
    that all pending tasks are properly cleaned up when the test is done.

    Yields:
        The event loop for the test.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    yield loop

    # Clean up pending tasks
    cleanup_pending_tasks()

    # Close the event loop
    loop.close()
