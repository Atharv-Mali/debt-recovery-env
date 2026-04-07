"""Validator-compatible server entrypoint."""

from __future__ import annotations

import uvicorn

from app import app


def main() -> None:
    """Launch the FastAPI application via uvicorn."""
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
