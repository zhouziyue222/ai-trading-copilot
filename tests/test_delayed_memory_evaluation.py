import json

import pytest

from ai_trading_copilot.copilot.domain import DistilledMemory, MemoryStatus, MemoryType, RunOutcome
from ai_trading_copilot.copilot.services.memory_evaluation import MemoryPromotionPolicy, MemoryShadowEvaluator
from ai_trading_copilot.copilot.services.memory_repository import SQLiteMemoryRepository


@pytest.fixture
def setup(tmp_path):
    repo = SQLiteMemoryRepository(tmp_path / 'memory.db')
    with repo._connection() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS review_snapshots(run_id TEXT,symbol TEXT,generated_at TEXT,payload_json TEXT,PRIMARY KEY(run_id,symbol));
        CREATE TABLE IF NOT EXISTS review_tasks(id TEXT PRIMARY KEY,run_id TEXT,symbol TEXT,horizon_days INTEGER,status TEXT);
        CREATE TABLE IF NOT EXISTS review_results(review_id TEXT PRIMARY KEY,payload_json TEXT);
        ''')
    memory = repo.upsert(DistilledMemory(memory_type=MemoryType.STRATEGY_PERFORMANCE,
        lesson='Wait for confirmation', confidence=.9, status=MemoryStatus.SHADOW,
        source_run_id='origin', evidence_run_ids=['evidence'], metadata={'delayed_review': True}))
    with repo._connection() as c:
        c.execute("UPDATE memory_versions SET created_at='2025-01-01T00:00:00+00:00'")
    return repo, memory


def add(repo, memory, run='one', symbol='AAPL', horizon=20, category='hypothetical', **updates):
    observation = dict(category=category, eligible=True, realized_return=.1, benchmark_return=.02, max_drawdown=-.05)
    observation.update(updates)
    repo.record_usage(run_id=run, symbol=symbol, memories=[memory], mode='shadow')
    payload = dict(run_id=run, symbol=symbol, horizon_days=horizon, cutoff_at='2025-03-01T00:00:00+00:00', observations=[observation])
    identifier = f'{run}-{symbol}-{horizon}'
    with repo._connection() as c:
        c.execute("UPDATE memory_usage SET created_at='2025-01-02T00:00:00+00:00'")
        c.execute('INSERT OR IGNORE INTO review_snapshots VALUES (?,?,?,?)', (run,symbol,'2025-01-03T00:00:00+00:00','{}'))
        c.execute('INSERT INTO review_tasks(id,run_id,symbol,horizon_days,status) VALUES (?,?,?,?,?)', (identifier,run,symbol,horizon,'completed'))
        c.execute('INSERT INTO review_results VALUES (?,?)', (identifier,json.dumps(payload)))


def evaluator(repo):
    return MemoryShadowEvaluator(repo, MemoryPromotionPolicy(min_evidence_runs=1,min_outcome_samples=1,allow_auto_promotion=True))


def test_group_runs_and_manual_promotion(setup):
    repo, memory = setup
    add(repo,memory)
    add(repo,memory,symbol='MSFT',realized_return=.2,max_drawdown=-.1)
    add(repo,memory,horizon=5)
    result = evaluator(repo).evaluate(memory.memory_id)
    assert result.eligible and result.outcome_samples == result.evidence_runs == 1
    assert result.mean_realized_return == pytest.approx(.15)
    assert result.mean_excess_return == pytest.approx(.13)
    assert result.worst_drawdown == -.1
    assert 'hypothetical' in result.reasons[-1]
    with pytest.raises(PermissionError, match='manual'):
        evaluator(repo).promote(memory.memory_id,actor='test',reason='test',automatic=True)
    assert evaluator(repo).promote(memory.memory_id,actor='test',reason='test').status == MemoryStatus.APPROVED


@pytest.mark.parametrize('updates',[{'benchmark_return':None},{'max_drawdown':None},{'eligible':False},{'realized_return':float('nan')}])
def test_partial_metrics_never_pass(setup,updates):
    repo,memory=setup
    add(repo,memory)
    add(repo,memory,symbol='MSFT',**updates)
    result=evaluator(repo).evaluate(memory.memory_id)
    assert not result.eligible and result.outcome_samples == 0


@pytest.mark.parametrize('run',['origin','evidence'])
def test_source_runs_excluded(setup,run):
    repo,memory=setup
    add(repo,memory,run=run)
    assert evaluator(repo).evaluate(memory.memory_id).outcome_samples == 0


@pytest.mark.parametrize('column',['usage','version','cutoff','status'])
def test_temporal_and_completion_guards(setup,column):
    repo,memory=setup
    add(repo,memory)
    with repo._connection() as c:
        if column=='usage': c.execute("UPDATE memory_usage SET created_at='2025-02-01T00:00:00+00:00'")
        if column=='version': c.execute("UPDATE memory_versions SET created_at='2025-02-01T00:00:00+00:00'")
        if column=='status': c.execute("UPDATE review_tasks SET status='pending'")
        if column=='cutoff':
            payload=json.loads(c.execute('SELECT payload_json FROM review_results').fetchone()[0])
            payload['cutoff_at']='2025-01-01T00:00:00+00:00'
            c.execute('UPDATE review_results SET payload_json=?',(json.dumps(payload),))
    assert evaluator(repo).evaluate(memory.memory_id).outcome_samples == 0


def test_account_category_and_legacy_isolation(setup):
    repo,memory=setup
    memory=repo.upsert(memory.model_copy(update={'metadata':{'delayed_review':True,'validation_category':'REAL','validation_account_id':'a'}}))
    with repo._connection() as c:
        c.execute("UPDATE memory_versions SET created_at='2025-01-01T00:00:00+00:00'")
    add(repo,memory,run='sim',category='SIMULATE',account_id='a')
    add(repo,memory,run='other',category='REAL',account_id='b')
    repo.record_outcome(RunOutcome(run_id='sim',symbol='AAPL',realized_return=.9,benchmark_return=0,max_drawdown=0))
    assert evaluator(repo).evaluate(memory.memory_id).outcome_samples == 0
    add(repo,memory,run='valid',category='REAL',account_id='a')
    assert evaluator(repo).evaluate(memory.memory_id).eligible


def test_exact_version_required(setup):
    repo,memory=setup
    add(repo,memory)
    updated=repo.upsert(memory.model_copy(update={'lesson':'Updated lesson'}))
    assert evaluator(repo).evaluate(updated.memory_id).outcome_samples == 0


def test_no_legacy_outcome_read_for_delayed_memory(setup, monkeypatch):
    repo,memory=setup
    add(repo,memory)
    def forbidden(*args, **kwargs):
        raise AssertionError('legacy outcomes must not be queried')
    monkeypatch.setattr(repo,'outcomes_for_memory',forbidden)
    assert evaluator(repo).evaluate(memory.memory_id).eligible


def test_multiple_symbols_do_not_satisfy_run_threshold(setup):
    repo,memory=setup
    for symbol in ['AAPL','MSFT','GOOG']:
        add(repo,memory,symbol=symbol)
    result=MemoryShadowEvaluator(repo).evaluate(memory.memory_id)
    assert result.evidence_runs == result.outcome_samples == 1
    assert not result.eligible


def test_risk_reduction_still_requires_every_drawdown(setup):
    from ai_trading_copilot.copilot.domain import MemoryValidationTarget
    repo,memory=setup
    memory=repo.upsert(memory.model_copy(update={'validation_target':MemoryValidationTarget.RISK_REDUCTION}))
    with repo._connection() as c:
        c.execute("UPDATE memory_versions SET created_at='2025-01-01T00:00:00+00:00'")
    add(repo,memory,benchmark_return=None,realized_return=None)
    assert evaluator(repo).evaluate(memory.memory_id).eligible
    add(repo,memory,symbol='MSFT',max_drawdown=None)
    assert not evaluator(repo).evaluate(memory.memory_id).eligible


def test_missing_review_tables_fails_closed(setup):
    repo,memory=setup
    with repo._connection() as c:
        c.execute('DROP TABLE review_results')
    result=evaluator(repo).evaluate(memory.memory_id)
    assert not result.eligible
    assert any('unavailable' in reason for reason in result.reasons)
