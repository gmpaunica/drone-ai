"""Shared, lazily constructed DJITelloPy session."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class TelloAdapterError(RuntimeError):
    pass


class TelloSession:
    """Own one SDK client shared by flight and video adapters."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._client = client
        self._client_factory = client_factory
        self._connected = False
        self._streaming = False

    @property
    def client(self) -> Any:
        if self._client is None:
            factory = self._client_factory
            if factory is None:
                try:
                    from djitellopy import Tello
                except ImportError as exc:
                    raise TelloAdapterError(
                        "DJITelloPy is not installed; install requirements-tello.txt"
                    ) from exc
                factory = Tello
            try:
                self._client = factory()
            except Exception as exc:
                raise TelloAdapterError(f"could not create Tello SDK client: {exc}") from exc
        return self._client

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def streaming(self) -> bool:
        return self._streaming

    def connect(self) -> None:
        if self._connected:
            return
        try:
            self.client.connect()
        except Exception as exc:
            raise TelloAdapterError(
                f"could not connect to Tello; verify its Wi-Fi connection: {exc}"
            ) from exc
        self._connected = True

    def require_connected(self) -> Any:
        if not self._connected:
            raise TelloAdapterError("Tello is not connected")
        return self.client

    def start_stream(self) -> Any:
        self.connect()
        started_here = False
        try:
            if not self._streaming:
                self.client.streamon()
                self._streaming = True
                started_here = True
            return self.client.get_frame_read()
        except Exception as exc:
            cleanup_detail = ""
            if started_here:
                try:
                    self.stop_stream()
                except Exception as cleanup_exc:
                    cleanup_detail = f"; stream rollback also failed: {cleanup_exc}"
            raise TelloAdapterError(
                f"could not start Tello video stream: {exc}{cleanup_detail}"
            ) from exc

    def stop_stream(self) -> None:
        if not self._streaming:
            return
        try:
            client = self.client
            send_without_wait = getattr(client, "send_command_without_return", None)
            if callable(send_without_wait):
                # Stream shutdown is cleanup, not a flight decision. Waiting
                # for the UDP acknowledgement can block for 21 seconds, then
                # DJITelloPy.end() may repeat the same delay. Send the official
                # command once and close the local decoder deterministically.
                send_without_wait("streamoff")
                if hasattr(client, "stream_on"):
                    client.stream_on = False
                frame_reader = getattr(client, "background_frame_read", None)
                if frame_reader is not None:
                    frame_reader.stop()
                    client.background_frame_read = None
            else:
                client.streamoff()
        except Exception as exc:
            raise TelloAdapterError(f"could not stop Tello video stream: {exc}") from exc
        finally:
            self._streaming = False

    def close(self) -> None:
        errors: list[str] = []
        if self._streaming:
            try:
                self.stop_stream()
            except Exception as exc:
                errors.append(str(exc))
        if self._client is not None and self._connected:
            try:
                self._client.end()
            except Exception as exc:
                errors.append(f"could not close Tello session: {exc}")
            finally:
                self._connected = False
        if errors:
            raise TelloAdapterError("; ".join(errors))
