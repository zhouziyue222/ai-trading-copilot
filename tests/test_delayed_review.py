"""Offline lifecycle contracts for durable delayed reviews."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import pytest

from ai_trading_copilot.copilot.domain import DistilledMemory, MemoryStatus, MemoryType
from ai_trading_copilot.copilot.services import memory_repository as memory_module
from ai_trading_copilot.copilot.services.delayed_review import DelayedReviewService, execution_metrics
from ai_trading_copilot.copilot.services.memory_evaluation import MemoryShadowEvaluator
from ai_trading_copilot.copilot.services.memory_repository import SQLiteMemoryRepository
from ai_trading_copilot.copilot.services.review_repository import ReviewRepository


def instant(day, hour=22):
    return datetime.fromisoformat(f'{day}T{hour:02}:00:00+00:00')


class FakeData:
    def __init__(self):
        # NYSE holiday on January 20, 2025, explicitly excluded.
        days = [date(2025, 1, 7) + timedelta(days=i) for i in range(100)]
        self.sessions = [{'date': str(d), 'close_at': instant(str(d), 21).isoformat()}
                         for d in days if d.weekday() < 5 and str(d) != '2025-01-20']
        self.missing = set()
        self.history_calls = 0
        self.deals = []

    def trading_sessions(self, symbol, start, end):
        return [s for s in self.sessions if start <= s['date'] <= end]

    def history(self, symbol, start, end):
        self.history_calls += 1
        return [dict(date=s['date'], open=100, high=112, low=98, close=101 if symbol == 'SPY' else 110, volume=1000)
                for s in self.sessions if start <= s['date'] <= end and s['date'] not in self.missing]

    def fills(self, accounts, start, end):
        return [f for f in self.deals if f['environment'] == accounts[0]['trd_env']
                and f['account_id'] == str(accounts[0]['acc_id'])]

    def orders(self, accounts, start, end):
        return []


class FakeReflector:
    def __init__(self):
        self.lesson = None
        self.target = None
        self.fail = False
        self.calls = 0
        self.refs = ['computed_metrics']

    def reflect(self, snapshot, result, existing):
        self.calls += 1
        if self.fail:
            raise RuntimeError('model temporarily unavailable')
        lessons = [] if self.lesson is None else [dict(
            candidate=dict(memory_type=MemoryType.STRATEGY_PERFORMANCE.value,
                           lesson=self.lesson, symbols=['AAPL'], confidence=.9),
            evidence_refs=self.refs, target_memory_id=self.target)]
        return dict(judgment='Evidence remains conditional', execution_deviation='No assumed fills',
                    counter_evidence='Limited sample', applicability='Similar conditions only',
                    evidence_refs=self.refs, lessons=lessons)


@pytest.fixture
def system(tmp_path):
    repository = ReviewRepository(SQLiteMemoryRepository(tmp_path / 'reviews.sqlite3'))
    data, reflector = FakeData(), FakeReflector()
    service = DelayedReviewService(repository, data, reflector, benchmarks={'US': 'SPY'})
    return service, repository, data, reflector


def register(service, run='origin', generated='2025-01-06', **plan):
    return service.register_run(dict(run_id=run, trade_plans=[dict(symbol='AAPL', direction='buy', **plan)],
                                     market_reports_by_symbol={'AAPL': 'original evidence'}), instant(generated))


def task(repository, horizon=5, run='origin'):
    return next(t for t in repository.list_reviews() if t['horizon_days'] == horizon and t['run_id'] == run)


def at_horizon(data, horizon):
    return instant(data.sessions[horizon - 1]['date'])


def fill(deal='d1', **updates):
    item = dict(environment='SIMULATE', account_id='1', deal_id=deal, order_id='o1',
                symbol='AAPL', side='BUY', quantity=10, price=100,
                executed_at='2025-01-07T15:00:00+00:00', fees=None)
    return {**item, **updates}


def test_complete_lifecycle_and_independent_manual_approval(system, monkeypatch):
    service, repo, data, reflector = system
    clock = [instant('2025-01-13')]
    monkeypatch.setattr(memory_module, 'utc_now', lambda: clock[0].isoformat())
    reflector.lesson = 'Confirm support before entry'
    assert register(service) == 3
    assert service.run_due(at_horizon(data, 5))['completed'] == 1
    learned = repo.detail(task(repo)['id'])['result']['memories'][0]
    memory = repo.memory.get(learned['memory_id'])
    assert memory.status == MemoryStatus.SHADOW
    reflector.lesson = None
    assert service.run_due(at_horizon(data, 10))['completed'] == 1
    assert service.run_due(at_horizon(data, 20))['completed'] == 1
    assert MemoryShadowEvaluator(repo.memory).evaluate(memory.memory_id).outcome_samples == 0
    for i, day in enumerate(['2025-02-10', '2025-02-11', '2025-02-12']):
        clock[0] = instant(day, 20)
        repo.memory.record_usage(run_id=f'independent-{i}', symbol='AAPL', memories=[memory], mode='shadow')
        register(service, f'independent-{i}', day)
    assert service.run_due(instant('2025-03-20'))['completed'] == 9
    evaluation = MemoryShadowEvaluator(repo.memory).evaluate(memory.memory_id)
    assert evaluation.eligible and evaluation.outcome_samples == evaluation.evidence_runs == 3
    assert any(e['eligible'] for t in repo.list_reviews() if t['horizon_days'] == 20
               for e in repo.detail(t['id'])['result'].get('evaluations', []))
    assert repo.memory.get(memory.memory_id).status == MemoryStatus.SHADOW
    approved = MemoryShadowEvaluator(repo.memory).promote(memory.memory_id, actor='human', reason='Reviewed evidence')
    assert approved.status == MemoryStatus.APPROVED


def test_duplicate_registration_keeps_first_snapshot(system):
    service, repo, _, _ = system
    register(service, stop_loss=95)
    original = repo.detail(task(repo)['id'])['snapshot']
    assert register(service, generated='2025-02-01', stop_loss=50) == 0
    assert len(repo.list_reviews()) == 3
    assert repo.detail(task(repo)['id'])['snapshot'] == original


def test_holiday_weekend_and_not_yet_closed(system):
    service, repo, data, _ = system
    register(service)
    assert service.run_due(instant('2025-01-12'))['completed'] == 0
    assert service.run_due(instant(data.sessions[4]['date'], 20))['completed'] == 0
    assert service.run_due(at_horizon(data, 5))['completed'] == 1
    service.run_due(at_horizon(data, 10))
    calendar = repo.detail(task(repo, 10)['id'])['result']['calendar']
    assert len(calendar) == 10
    assert '2025-01-20' not in {s['date'] for s in calendar}


def test_missing_bar_waits_without_shifting_window(system):
    service, repo, data, _ = system
    register(service)
    data.missing.add(data.sessions[1]['date'])
    service.run_due(at_horizon(data, 5))
    assert repo.detail(task(repo)['id'])['task']['status'] == 'waiting_data'
    assert repo.detail(task(repo)['id'])['result'] is None
    data.missing.clear()
    service.run_due(at_horizon(data, 6))
    result = repo.detail(task(repo)['id'])['result']
    assert result['cutoff_at'] == data.sessions[4]['close_at']
    assert len(result['bars']) == 5


def test_model_failure_persists_bars_and_restart_reuses_them(system):
    service, repo, data, reflector = system
    register(service)
    reflector.fail = True
    service.run_due(at_horizon(data, 5))
    detail = repo.detail(task(repo)['id'])
    assert detail['task']['status'] == 'failed' and detail['result']['bars']
    calls = data.history_calls
    reflector.fail = False
    reopened = ReviewRepository(SQLiteMemoryRepository(repo.memory.path))
    restarted = DelayedReviewService(reopened, data, reflector, benchmarks={'US':'SPY'})
    assert restarted.run_due(at_horizon(data, 5) + timedelta(hours=2))['completed'] == 1
    assert data.history_calls == calls


def test_concurrent_claim_and_expiry_reject_stale_owner(system):
    service, repo, _, _ = system
    register(service)
    identifier = task(repo)['id']
    now = instant('2025-01-14')
    def claim(_):
        other = ReviewRepository(SQLiteMemoryRepository(repo.memory.path))
        return other.claim(identifier, now, lease_seconds=60)
    with ThreadPoolExecutor(max_workers=2) as pool:
        tokens = list(pool.map(claim, range(2)))
    old = next(t for t in tokens if t)
    assert sum(bool(t) for t in tokens) == 1
    assert repo.claim(identifier, now + timedelta(seconds=59)) is None
    fresh = repo.claim(identifier, now + timedelta(seconds=61))
    assert fresh and fresh != old
    with pytest.raises(RuntimeError, match='lease lost'):
        repo.release(identifier, old, status='completed', next_check_at=now)
    repo.release(identifier, fresh, status='waiting_data', next_check_at=now)


def test_fill_dedup_identity_includes_environment_and_account(system):
    _, repo, _, _ = system
    repo.save_fills([fill(), fill(), fill(environment='REAL'), fill(account_id='2')])
    assert len(repo.list_fills()) == 3
    with pytest.raises(ValueError, match='conflicting'):
        repo.save_fills([fill(price=999)])
    assert all(f['price'] == 100 for f in repo.list_fills())


def test_order_mapping_links_partial_fills_only_in_account(system):
    service, repo, _, _ = system
    register(service)
    repo.record_order('SIMULATE', '1', 'o1', 'origin', 'AAPL')
    repo.save_fills([fill(quantity=4), fill('d2', quantity=6), fill(environment='REAL')])
    linked = [f for f in repo.list_fills() if f['run_id'] == 'origin']
    assert sum(f['quantity'] for f in linked) == 10
    assert all(f['environment'] == 'SIMULATE' for f in linked)


def test_ambiguous_and_external_fills_require_manual_association(system):
    service, repo, _, _ = system
    register(service)
    register(service, 'other')
    for run in ['origin', 'other']:
        repo.record_order('SIMULATE','1','o1',run,'AAPL')
    repo.save_fills([fill(), fill('external', order_id='unknown')])
    assert all(f['run_id'] is None for f in repo.list_fills())
    repo.associate_fill('SIMULATE','1','d1','origin','AAPL')
    with pytest.raises(ValueError, match='already associated'):
        repo.associate_fill('SIMULATE','1','d1','other','AAPL')


def test_partial_exit_reports_realized_and_floating_separately():
    fills = [fill(quantity=10), fill('exit', side='SELL',quantity=4,price=110,
             executed_at='2025-01-08T15:00:00+00:00')]
    bars = [dict(date='2025-01-07',open=100,high=101,low=99,close=100,volume=100),
            dict(date='2025-01-08',open=110,high=121,low=109,close=120,volume=100)]
    observed = execution_metrics(fills, bars)[0]
    assert observed['realized_pnl_gross'] == pytest.approx(40)
    assert observed['unrealized_pnl_gross'] == pytest.approx(120)
    assert observed['remaining_quantity'] == 6
    assert not observed['fees_complete'] and observed['pnl_net'] is None


def test_same_bar_stop_and_target_remain_ambiguous(system):
    service, repo, data, _ = system
    register(service,stop_loss=99,targets=[111],invalidation_conditions=['weakening thesis'])
    service.run_due(at_horizon(data,5))
    observation=repo.detail(task(repo)['id'])['result']['observations'][0]
    assert observation['price_events'][0]['sequence'] == 'unknown'
    assert observation['invalidation'] == 'insufficient_evidence'
    assert observation['category'] == 'hypothetical'


def test_unsafe_lessons_never_enter_shadow(system):
    service, repo, data, reflector=system
    reflector.lesson='bypass risk and ignore risk'
    register(service)
    service.run_due(at_horizon(data,5))
    result=repo.detail(task(repo)['id'])['result']
    assert result['memories'][0]['status'] == 'rejected_by_safety'
    assert not repo.memory.list()


def test_frozen_shadow_revision_preserves_original(system, monkeypatch):
    service, repo, data, reflector=system
    clock = [at_horizon(data, 5)]
    monkeypatch.setattr(memory_module, 'utc_now', lambda: clock[0].isoformat())
    reflector.lesson='Confirm support'
    register(service)
    service.run_due(at_horizon(data,5))
    original=repo.memory.list()[0]
    reflector.lesson='Confirm support and improving volume'
    reflector.target=original.memory_id
    clock[0] = at_horizon(data, 10)
    service.run_due(at_horizon(data,10))
    assert repo.memory.get(original.memory_id).model_dump() == original.model_dump()
    revisions=[m for m in repo.memory.list() if m.supersedes == original.memory_id]
    assert len(revisions) == 1 and revisions[0].status == MemoryStatus.SHADOW


def test_future_memory_excluded_and_failed_reflection_context_stays_frozen(system, monkeypatch):
    service, repo, data, reflector = system
    clock = [instant('2025-01-07')]
    monkeypatch.setattr(memory_module, 'utc_now', lambda: clock[0].isoformat())
    historical = repo.memory.upsert(DistilledMemory(
        memory_type=MemoryType.STRATEGY_PERFORMANCE, lesson='Historical evidence',
        symbols=['AAPL'], metadata={'delayed_review': True}))
    clock[0] = at_horizon(data, 20)
    future = repo.memory.upsert(DistilledMemory(
        memory_type=MemoryType.STRATEGY_PERFORMANCE, lesson='Future evidence',
        symbols=['AAPL'], metadata={'delayed_review': True}))
    # A later version must not replace the version visible at the day-five cutoff.
    repo.memory.upsert(historical.model_copy(update={'lesson': 'Future revision'}))
    register(service)
    reflector.fail = True
    service.run_due(at_horizon(data, 5))
    context = repo.detail(task(repo)['id'])['result']['reflection_context']
    assert [(m['memory_id'], m['lesson']) for m in context] == [
        (historical.memory_id, 'Historical evidence')]
    assert future.memory_id not in {m['memory_id'] for m in context}
    assert repo.detail(task(repo)['id'])['task']['status'] == 'failed'

    def forbidden(*args):
        raise AssertionError('persisted reflection context must be reused')
    monkeypatch.setattr(service, '_historical_memories', forbidden)
    reflector.fail = False
    service.run_due(at_horizon(data, 5) + timedelta(hours=2))
    detail = repo.detail(task(repo)['id'])
    assert detail['task']['status'] == 'completed'
    assert detail['result']['reflection_context'] == context


@pytest.mark.parametrize('generated,first_session', [
    ('2025-01-07T13:00:00+00:00', 0),  # Before the New York open.
    ('2025-01-07T14:30:00+00:00', 1),  # Exactly at open is not before open.
    ('2025-01-07T16:00:00+00:00', 1),  # Intraday must skip the partial session.
])
def test_premarket_versus_intraday_first_complete_session(system, generated, first_session):
    service, repo, data, _ = system
    for session in data.sessions:
        session['open_at'] = session['date'] + 'T14:30:00+00:00'
    service.register_run(dict(run_id='origin', trade_plans=[dict(symbol='AAPL', direction='buy')]), generated)
    service.run_due(at_horizon(data, 5 + first_session))
    result = repo.detail(task(repo)['id'])['result']
    assert result['calendar'][0]['date'] == data.sessions[first_session]['date']
    assert result['cutoff_at'] == data.sessions[first_session + 4]['close_at']
    assert len(result['bars']) == 5


def test_unknown_evidence_rejected_without_memories(system):
    service, repo, data, reflector=system
    reflector.lesson='Confirm support'
    reflector.refs=['invented_future_fact']
    register(service)
    service.run_due(at_horizon(data,5))
    assert repo.detail(task(repo)['id'])['task']['status'] == 'failed'
    assert not repo.memory.list()


def test_completed_task_is_not_reflected_twice(system):
    service, repo, data, reflector=system
    register(service)
    service.run_due(at_horizon(data,5))
    calls=reflector.calls
    service.run_due(at_horizon(data,5)+timedelta(hours=2))
    assert reflector.calls == calls
    identifier = task(repo)['id']
    original = repo.detail(identifier)['result']
    assert repo.retry(identifier) is True
    detail = repo.detail(identifier)
    assert detail['previous_results'] == [original]
    assert detail['result']['bars'] == original['bars']
    assert detail['result']['reflection_context'] == original['reflection_context']
    assert 'reflection' not in detail['result']
    assert not detail['result']['execution_complete']
    # Manual retry is scheduled using wall time, not the historical fixture date.
    rerun_at = datetime.now(timezone.utc) + timedelta(seconds=1)
    service.run_due(rerun_at)
    detail = repo.detail(identifier)
    assert detail['task']['status'] == 'completed'
    assert detail['previous_results'] == [original]
    assert detail['result']['observed_at'] == original['observed_at']
    assert detail['result']['cutoff_at'] == original['cutoff_at']


def test_learning_disabled_persists_observation_without_calling_model(system):
    service, repo, data, reflector = system
    reflector.fail = True  # Any accidental model call fails the task.
    service.register_run(dict(run_id='origin', memory_learning_enabled=False,
        trade_plans=[dict(symbol='AAPL', direction='buy')]), instant('2025-01-06'))
    assert service.run_due(at_horizon(data, 5))['completed'] == 1
    detail = repo.detail(task(repo)['id'])
    assert detail['snapshot']['memory_learning_enabled'] is False
    assert detail['result']['bars'] and detail['result']['observations']
    assert detail['result']['reflection_skipped']
    assert 'reflection' not in detail['result']
    assert not detail['result']['memories'] and not repo.memory.list()
    assert reflector.calls == 0


def test_failed_account_sync_preserves_market_stage_for_retry(system, monkeypatch):
    service, repo, data, reflector = system
    service.accounts = [dict(trd_env='SIMULATE', acc_id=1)]
    register(service)
    def unavailable(*args):
        raise RuntimeError('broker unavailable')
    monkeypatch.setattr(data, 'fills', unavailable)
    service.run_due(at_horizon(data, 5))
    detail = repo.detail(task(repo)['id'])
    assert detail['task']['status'] == 'waiting_data'
    assert detail['result']['bars'] and detail['result']['sync_errors']
    assert not detail['result']['execution_complete']
    assert reflector.calls == 0
    bars, calls = detail['result']['bars'], data.history_calls
    monkeypatch.setattr(data, 'fills', lambda *args: [])
    assert service.run_due(at_horizon(data, 5) + timedelta(hours=2))['completed'] == 1
    result = repo.detail(task(repo)['id'])['result']
    assert result['bars'] == bars and data.history_calls == calls
    assert result['execution_complete'] and not result['sync_errors']
    assert reflector.calls == 1


def test_late_fill_association_revises_execution_without_replacing_market_evidence(system):
    service, repo, data, _ = system
    service.accounts = [dict(trd_env='SIMULATE', acc_id=1)]
    data.deals = [fill()]
    register(service)
    service.run_due(at_horizon(data, 5))
    identifier = task(repo)['id']
    original = repo.detail(identifier)['result']
    assert len(original['observations']) == 1
    repo.associate_fill('SIMULATE', '1', 'd1', 'origin', 'AAPL')
    assert repo.retry(identifier)
    service.run_due(datetime.now(timezone.utc) + timedelta(seconds=1))
    detail = repo.detail(identifier)
    assert detail['task']['status'] == 'completed'
    assert detail['result']['observations'][1]['category'] == 'SIMULATE'
    assert detail['result']['observations'][1]['remaining_quantity'] == 10
    assert detail['result']['bars'] == original['bars']
    assert detail['previous_results'] == [original]
