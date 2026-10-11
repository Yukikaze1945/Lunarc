"""Presentation backpressure and command/source revision races."""
import asyncio
import json
import threading
import time
from preview_player import PreviewPlayer


class Socket:
    def __init__(self):
        self.messages, self.images = [], []

    async def send_text(self, value):
        self.messages.append(json.loads(value))

    async def send_bytes(self, value):
        self.images.append(value)


class Service:
    video_info = dict(fps=30, frames=30)
    video_path = 'a.mkv'
    _preview_revision = 0

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = []

    def render_preview(self, **options):
        self.calls.append(options)
        self.entered.set()
        assert self.release.wait(3), 'test renderer completion timeout'
        return b'jpeg', 'ok'

    def preview_diagnostics(self):
        return dict(layer_count=2)


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(.005)


async def close(player, task):
    player.closed = True
    player.wake.set()
    await task


def test_latest_seek_discards_inflight_result_and_coalesces_pending_frames():
    async def scenario():
        service, socket = Service(), Socket()
        player = PreviewPlayer(socket, service)
        player.command(dict(action='seek', frame=1, request_id=1, transport='binary', ack=True))
        task = asyncio.create_task(player.run())
        await until(service.entered.is_set)
        for frame in (2, 3, 12):
            player.command(dict(action='seek', frame=frame, request_id=frame))
        assert len(service.calls) == 1
        service.release.set()
        await until(lambda: len(socket.images) == 1)
        frames = [m for m in socket.messages if m['type'] == 'frame']
        assert [m['frame'] for m in frames] == [12]
        assert [m['request_id'] for m in frames] == [12]
        assert [c['frame_idx'] for c in service.calls] == [1, 12]
        assert frames[0]['preview']['stale_results_discarded'] == 1
        await close(player, task)
    asyncio.run(scenario())


def test_playback_waits_for_matching_display_ack_and_pause_stops_work():
    async def scenario():
        service, socket = Service(), Socket()
        service.release.set()
        player = PreviewPlayer(socket, service)
        player.command(dict(action='play', frame=0, transport='binary', ack=True, request_id=4))
        task = asyncio.create_task(player.run())
        await until(lambda: len(socket.images) == 1)
        await asyncio.sleep(.08)
        assert len(service.calls) == 1
        player.command(dict(action='presented', sequence=999))
        assert player.awaiting == 1
        player.command(dict(action='presented', sequence=1, decode_ms=2, paint_wait_ms=3))
        await until(lambda: len(socket.images) == 2)
        player.command(dict(action='pause', request_id=5))
        count = len(service.calls)
        await asyncio.sleep(.05)
        assert len(service.calls) == count and player.awaiting is None
        assert player.acks == dict(decode_ms=2., paint_wait_ms=3.)
        await close(player, task)
    asyncio.run(scenario())


def test_source_revision_changed_during_render_cannot_publish_old_source():
    async def scenario():
        service, socket = Service(), Socket()
        player = PreviewPlayer(socket, service)
        player.command(dict(action='seek', frame=0, request_id=1))
        task = asyncio.create_task(player.run())
        await until(service.entered.is_set)
        service.video_path = 'b.mkv'; service._preview_revision += 1
        service.release.set()
        await until(lambda: any(m['type'] == 'frame' for m in socket.messages))
        frames = [m for m in socket.messages if m['type'] == 'frame']
        assert frames[0]['source'] == 'b.mkv'
        assert frames[0]['preview']['stale_results_discarded'] == 1
        assert len(service.calls) == 2
        await close(player, task)
    asyncio.run(scenario())


def test_clients_share_native_slot_and_cancellation_waits_for_completion():
    async def scenario():
        services = [Service(), Service()]
        players = [PreviewPlayer(Socket(), s) for s in services]
        for p in players:
            p.command(dict(action='seek', frame=0))
        tasks = [asyncio.create_task(p.run()) for p in players]
        await until(services[0].entered.is_set)
        tasks[0].cancel()
        await asyncio.sleep(.03)
        assert not services[1].entered.is_set()
        services[0].release.set()
        try:
            await tasks[0]
        except asyncio.CancelledError:
            pass
        await until(services[1].entered.is_set)
        services[1].release.set()
        await until(lambda: bool(players[1].socket.messages))
        await close(players[1], tasks[1])
    asyncio.run(scenario())


def test_fast_display_ack_keeps_playback_paced_across_short_video_loops():
    async def scenario():
        service = Service()
        service.video_info = dict(fps=10, frames=3)
        service.release.set()
        sent_at = []

        class ImmediateDisplay(Socket):
            async def send_bytes(self, value):
                await super().send_bytes(value)
                sent_at.append(time.perf_counter())
                player.command(dict(action='presented', sequence=player.sequence))
                if len(self.images) == 7:
                    player.command(dict(action='pause'))

        socket = ImmediateDisplay()
        player = PreviewPlayer(socket, service)
        player.command(dict(action='play', frame=0, transport='binary', ack=True))
        task = asyncio.create_task(player.run())
        await until(lambda: len(socket.images) == 7)
        frames = [m['frame'] for m in socket.messages if m['type'] == 'frame']
        assert frames == [0, 1, 2, 0, 1, 2, 0]
        # An immediate ACK may release backpressure, but must not bypass the
        # frame-rate deadline. In particular, looping must not send cached
        # duplicate frames between clock ticks.
        assert all(b-a >= .08 for a, b in zip(sent_at, sent_at[1:]))
        await close(player, task)
    asyncio.run(scenario())
