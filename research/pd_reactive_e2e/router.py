"""Bounded cache-affine admission and authenticated control for the AgentX adapter.

Queues are timed before admission. A closed gate prevents *both* P and D HTTP
submissions; already streaming requests continue. No future arrivals are read.
"""
from __future__ import annotations
import asyncio
from contextvars import ContextVar
import hashlib
import hmac
import json
import time
from aiohttp import web

class Admission:
    def __init__(self, endpoints, capacity=4):
        self.endpoints = {e.name:e for e in endpoints}
        self.enabled = {next(iter(self.endpoints))}
        self.capacity = capacity
        self.active = {name:0 for name in self.endpoints}
        self.streaming = set()
        self.owners = {}
        self.waiting = {}
        self.paused = False
        self.prefilling = 0
        self.context = ContextVar('reactive_admission', default=None)
        self.condition = asyncio.Condition()
        self.moves = 0

    def select(self, affinity):
        choices = [n for n in self.enabled if self.active[n] < self.capacity]
        if not choices:
            return None
        old = self.owners.get(affinity)
        chosen = old if old in choices else min(choices, key=lambda n:(self.active[n], n))
        if old and old != chosen:
            self.moves += 1
        self.owners[affinity] = chosen
        self.active[chosen] += 1
        return self.endpoints[chosen]

    async def acquire(self, affinity):
        ticket = object()
        started = time.monotonic()
        self.waiting[ticket] = started
        try:
            async with self.condition:
                while True:
                    endpoint = None if self.paused else self.select(affinity)
                    if endpoint:
                        return endpoint, time.monotonic()-started
                    await self.condition.wait()
        finally:
            self.waiting.pop(ticket, None)

    async def release(self, name):
        async with self.condition:
            if self.active[name] <= 0:
                raise RuntimeError('double release')
            self.active[name] -= 1
            self.condition.notify_all()

    def state(self):
        now = time.monotonic()
        return dict(observed_at=now, unix=time.time(), enabled=sorted(self.enabled),
                    paused=self.paused, active=dict(self.active), prefilling=self.prefilling,
                    streaming=len(self.streaming), waiting=len(self.waiting),
                    oldest_wait_s=max((now-t for t in self.waiting.values()),default=0),
                    cache_affinity_moves=self.moves, per_instance_admission=self.capacity)

class AuditQueue(asyncio.Queue):
    def __init__(self, admission):
        super().__init__(maxsize=65536)
        self.admission = admission
    def put_nowait(self, value):
        if value is not None:
            row = json.loads(value)
            ctx = self.admission.context.get()
            if ctx:
                wait = ctx['admission_wait_s']
                row['admission_wait_s'] = wait
                row['elapsed_s'] += wait
                if row.get('first_content_s') is not None:
                    row['first_content_s'] += wait
                row['started_unix_ns'] = ctx['started_unix_ns']
                row['decode_route'] = dict(endpoint=ctx['endpoint'].name,
                    enabled_at_admission=ctx['enabled'], cache_affinity=True)
            value = json.dumps(row)+'\n'
        return super().put_nowait(value)

def install(Adapter):
    old_init, old_owner, old_chat = Adapter.__init__, Adapter.owner, Adapter.chat
    old_start, old_app, old_prefill = Adapter.start, Adapter.app, Adapter.prefill
    def init(self, config, **kw):
        old_init(self, config, **kw)
        self.reactive = Admission(self.pools['decode'], config['reactive_admission'])
        self.audit_queue = AuditQueue(self.reactive)
    async def start(self, app):
        # The second replica is deliberately not a live GPU process before demand.
        endpoints = self.pools['decode']
        self.pools['decode'] = [endpoints[0]]
        try:
            await old_start(self, app)
        finally:
            self.pools['decode'] = endpoints
    def owner(self, role, affinity):
        if role == 'decode':
            ctx = self.reactive.context.get()
            if ctx is None:
                raise RuntimeError('unreserved Decode route')
            return ctx['endpoint'], 0
        return old_owner(self, role, affinity)
    async def prefill(self, endpoint, payload):
        self.reactive.prefilling += 1
        try:
            return await old_prefill(self, endpoint, payload)
        finally:
            self.reactive.prefilling -= 1
    async def chat(self, request):
        if not hmac.compare_digest(request.headers.get('Authorization',''), 'Bearer '+self.frontend_key):
            raise web.HTTPUnauthorized()
        affinity = request.headers.get('X-AgentX-Placement-Key')
        if not affinity or len(affinity) != 64 or any(c not in '0123456789abcdef' for c in affinity):
            raise web.HTTPBadRequest(text='stable placement key required')
        started = time.time_ns()
        endpoint, wait = await self.reactive.acquire(affinity)
        context = dict(endpoint=endpoint, admission_wait_s=wait, started_unix_ns=started,
                       enabled=sorted(self.reactive.enabled))
        token = self.reactive.context.set(context)
        try:
            return await old_chat(self, request)
        finally:
            self.reactive.context.reset(token)
            await self.reactive.release(endpoint.name)
    def auth(self, request):
        if not hmac.compare_digest(request.headers.get('Authorization',''), 'Bearer '+self.backend_key):
            raise web.HTTPUnauthorized()
    async def state(self, request):
        auth(self, request)
        return web.json_response(self.reactive.state())
    async def control(self, request):
        auth(self, request)
        body = await request.json()
        enabled = set(body.get('enabled', self.reactive.enabled))
        if not enabled or not enabled.issubset(self.reactive.endpoints):
            raise web.HTTPBadRequest(text='unknown endpoint')
        added = enabled-self.reactive.enabled
        # Enable only after an actual ready/HiCache check, before routing a request.
        for name in added:
            async with self.session.get(self.reactive.endpoints[name].url+'/get_server_info',
                                        headers=self.headers()) as response:
                response.raise_for_status()
                info = await response.json()
                if not info['internal_states'] or not all(s['enable_hierarchical_cache'] for s in info['internal_states']):
                    raise RuntimeError('new Decode replica lacks HiCache')
        async with self.reactive.condition:
            self.reactive.enabled = enabled
            self.reactive.paused = bool(body.get('paused',self.reactive.paused))
            self.reactive.condition.notify_all()
        return web.json_response(self.reactive.state())
    def app(self):
        result = old_app(self)
        result.router.add_get('/reactive/state', lambda r:state(self,r))
        result.router.add_post('/reactive/control', lambda r:control(self,r))
        return result
    Adapter.__init__, Adapter.start, Adapter.owner = init, start, owner
    Adapter.chat, Adapter.prefill, Adapter.app = chat, prefill, app
