# tiny_book_sim_ml_fixed_v3.py
"""
Sim + ML (v3) — Predict remaining time, use conservative hold rule + quantile regressor.

Run:
    pip install simpy pandas scikit-learn
    python tiny_book_sim_ml_fixed_v3.py
"""
import simpy
import random
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# -----------------------
# Helpers
# -----------------------
def travel_time_km(speed_kmph, length_km):
    return (length_km / speed_kmph) * 60.0  # minutes

# -----------------------
# Stage 1: generate data but capture per-request features (simulate queue_len)
# -----------------------
def generate_training_logs_with_queue(sim_duration=2000, n_trains=400, seed=0):
    """
    Generate event logs and then build training rows for 'remaining time' predictions.
    We'll simulate simple FIFO queue at a single block and record, at each REQUEST,
    the features: speed (of current occupant), block_length, elapsed (of occupant),
    queue_len (number waiting excluding the incoming train), and label=actual_remaining (minutes).
    """
    random.seed(seed)
    env = simpy.Environment()
    block_length_km = 5.0

    # We'll simulate simple single-occupancy block with queue to collect event traces
    events = []  # store events: ('REQUEST'/'ENTER'/'LEAVE', time, train, speed)

    class SimpleBlock:
        def __init__(self, env):
            self.env = env
            self.occupied = False
            self.current = None  # (train, speed, entry_time)
            self.queue = []

        def request_and_wait(self, train, speed):
            # record request
            events.append(('REQUEST', self.env.now, train, speed, len(self.queue)))
            ev = self.env.event()
            self.queue.append((train, speed, ev))
            # if free, admit immediately
            if not self.occupied:
                self._admit_next()
            return ev

        def _admit_next(self):
            if self.queue and not self.occupied:
                train, speed, ev = self.queue.pop(0)
                self.occupied = True
                self.current = (train, speed, self.env.now)
                events.append(('ENTER', self.env.now, train, speed, len(self.queue)))
                if not ev.triggered:
                    ev.succeed()

        def leave(self, train):
            if self.current and self.current[0] == train:
                events.append(('LEAVE', self.env.now, train, self.current[1], len(self.queue)))
                self.occupied = False
                self.current = None
                self._admit_next()

    block = SimpleBlock(env)

    def train_proc(env, tid, spawn_delay):
        yield env.timeout(spawn_delay)
        speed = random.choice([40, 50, 60, 80, 100, 120])
        ev = block.request_and_wait(tid, speed)
        yield ev  # admitted
        # actual travel time (with noise)
        actual = travel_time_km(speed, block_length_km) * random.uniform(0.9, 1.4)
        yield env.timeout(actual)
        block.leave(tid)

    # spawn trains
    t = 0.0
    for i in range(n_trains):
        spawn = t + random.expovariate(1/2.0)
        env.process(train_proc(env, f"T{i}", spawn))
        t = spawn

    env.run(until=sim_duration)

    # Build training rows: for each REQUEST event where at that time block was occupied,
    # compute elapsed for occupant and label = remaining time until occupant's LEAVE.
    # We need to find the occupant at that request and when that occupant left.
    df_events = pd.DataFrame(events, columns=['event', 'time', 'train', 'speed', 'queue_len'])
    # find all REQUEST rows
    reqs = df_events[df_events['event'] == 'REQUEST'].copy().reset_index(drop=True)

    # For each request, find current occupant at that time (ENTER before request and LEAVE after)
    rows = []
    # Build list of ENTER/LEAVE intervals by train
    enters = df_events[df_events['event'] == 'ENTER'][['train','time','speed']].rename(columns={'time':'enter_time'})
    leaves = df_events[df_events['event'] == 'LEAVE'][['train','time']].rename(columns={'time':'leave_time'})
    intervals = pd.merge(enters, leaves, on='train')
    # iterate requests
    for _, r in reqs.iterrows():
        req_time = r['time']
        qlen = r['queue_len']
        # find interval where enter_time <= req_time < leave_time
        occ = intervals[(intervals['enter_time'] <= req_time) & (intervals['leave_time'] > req_time)]
        if occ.empty:
            # block free at request time; skip (no need to predict remaining)
            continue
        occ = occ.iloc[0]
        elapsed = req_time - occ['enter_time']
        actual_remaining = occ['leave_time'] - req_time
        rows.append({
            'speed': occ['speed'],
            'block_length': block_length_km,
            'elapsed': elapsed,
            'queue_len': qlen,
            'remaining_min': actual_remaining
        })
    df = pd.DataFrame(rows)
    # drop any weirds
    df = df[df['remaining_min'] > 0].reset_index(drop=True)
    return df

# -----------------------
# Stage 2: train quantile regressor for lower quantile (conservative)
# -----------------------
def train_quantile_remaining_model(df, quantile=0.25):
    X = df[['speed','block_length','elapsed','queue_len']].copy()
    y = df['remaining_min'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # quantile regressor via GradientBoostingRegressor
    model = GradientBoostingRegressor(loss='quantile', alpha=quantile, n_estimators=200, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"[QuantileModel q={quantile}] MAE = {mae:.3f} min (on test set).")
    return model

# -----------------------
# Model-aware block using remaining-time predictor
# -----------------------
class ConservativeModelBlock:
    """
    Uses a regressor that predicts remaining time (we use lower quantile to avoid over-holding).
    Admission rule:
      - If block free -> admit immediately
      - If occupied and queue non-empty: compute predicted_remaining_lower = model.predict([cur_speed, block_length, elapsed, queue_len])
      - If predicted_remaining_lower >= hold_threshold_min: schedule watcher for min(predicted_remaining_lower + buffer, max_hold)
        else: admit immediately (do not hold)
    """
    def __init__(self, env, model=None, length_km=5.0, name="Block", hold_threshold_min=0.5, safety_buffer_min=0.1, max_hold_min=2.0):
        self.env = env
        self.model = model
        self.length_km = length_km
        self.name = name
        self.occupied = False
        self.current = None  # {'train','entry_time','speed'}
        self.queue = []      # (train, speed, ev, req_time)
        self._watcher = None
        self.hold_threshold_min = hold_threshold_min
        self.safety_buffer_min = safety_buffer_min
        self.max_hold_min = max_hold_min
        self._feature_cols = ['speed','block_length','elapsed','queue_len']

    def _predict_remaining_lower(self, cur_speed, elapsed, queue_len):
        if self.model is None:
            # fallback: simple physics remaining = total - elapsed
            total = (self.length_km / cur_speed) * 60.0
            return max(total - elapsed, 0.0)
        X = pd.DataFrame([[cur_speed, self.length_km, elapsed, queue_len]], columns=self._feature_cols)
        pred = self.model.predict(X)[0]
        return max(float(pred), 0.0)

    def request_enter(self, train, speed):
        ev = self.env.event()
        self.queue.append((train, speed, ev, self.env.now))
        # if free, admit immediately
        self._try_grant()
        # if occupied and waiters exist, evaluate hold policy and maybe schedule watcher
        if self.occupied and self.queue:
            self._evaluate_and_schedule()
        return ev

    def leave(self, train):
        if not self.occupied:
            return
        if self.current and self.current['train'] == train:
            self.occupied = False
            self.current = None
            # cancel watcher if any
            if self._watcher and not getattr(self._watcher, 'triggered', False):
                try:
                    self._watcher.interrupt()
                except Exception:
                    pass
                self._watcher = None
            self._try_grant()

    def _try_grant(self):
        if (not self.occupied) and self.queue:
            train, speed, ev, req_time = self.queue.pop(0)
            self.occupied = True
            self.current = {'train': train, 'entry_time': self.env.now, 'speed': speed}
            if not ev.triggered:
                ev.succeed()
            # if still waiters, maybe schedule watcher
            if self.queue:
                self._evaluate_and_schedule()

    def _evaluate_and_schedule(self):
        """
        Decide if we should hold using model's predicted lower quantile of remaining.
        If predicted_remaining_lower >= hold_threshold_min -> start watcher for min(pred+buffer, max_hold)
        else -> do nothing (we will admit immediately once occupant leaves).
        """
        if not self.occupied or not self.queue:
            return
        cur = self.current
        elapsed = self.env.now - cur['entry_time']
        qlen = len(self.queue)  # current number waiting
        pred_lower = self._predict_remaining_lower(cur['speed'], elapsed, qlen)

        # If prediction says remaining is small, don't hold (admit next when actual leave occurs)
        if pred_lower < self.hold_threshold_min:
            # no watcher; let actual leave() admit next
            return

        wait_time = pred_lower + self.safety_buffer_min
        if wait_time > self.max_hold_min:
            wait_time = self.max_hold_min

        # if a watcher exists, interrupt/reschedule
        if self._watcher and not getattr(self._watcher, 'triggered', False):
            try:
                self._watcher.interrupt()
            except Exception:
                pass
            self._watcher = None

        self._watcher = self.env.process(self._watcher_proc(wait_time))

    def _watcher_proc(self, wait_time):
        try:
            yield self.env.timeout(wait_time)
        except simpy.Interrupt:
            return
        finally:
            self._watcher = None
        # when watcher expires, try to admit next (if occupant already left, leave() handled it)
        self._try_grant()

# -----------------------
# Train process with logs for final evaluation
# -----------------------
def train_model_aware_logged(env, name, speed, block: ConservativeModelBlock, out_logs):
    ev = block.request_enter(name, speed)
    yield ev
    entry = env.now
    # actual travel time uses physics + noise
    actual = travel_time_km(speed, block.length_km) * random.uniform(0.9, 1.4)
    yield env.timeout(actual)
    exit_time = env.now
    out_logs.append({'train': name, 'entry_time': entry, 'exit_time': exit_time, 'speed': speed, 'block_length': block.length_km, 'travel_min': exit_time-entry})
    block.leave(name)

# -----------------------
# Runner utilities
# -----------------------
def build_and_train_quantile_model():
    df_rem = generate_training_logs_with_queue(sim_duration=2000, n_trains=400, seed=1)
    print(f"[TrainData] rows for remaining prediction: {len(df_rem)}, mean remaining = {df_rem['remaining_min'].mean():.2f}")
    model = train_quantile_remaining_model(df_rem, quantile=0.25)
    return model

def model_aware_run_with_conservative(model, n_trains=150, seed=2, hold_threshold_min=0.5, safety_buffer_min=0.1, max_hold_min=2.0):
    random.seed(seed)
    env = simpy.Environment()
    block = ConservativeModelBlock(env, model=model, length_km=5.0, name="Block-Conservative",
                                   hold_threshold_min=hold_threshold_min, safety_buffer_min=safety_buffer_min, max_hold_min=max_hold_min)
    logs = []

    def spawn(env, name, speed, spawn_delay):
        yield env.timeout(spawn_delay)
        yield env.process(train_model_aware_logged(env, name, speed, block, logs))

    t = 0.0
    for i in range(n_trains):
        spawn_delay = t + random.expovariate(1/2.0)
        speed = random.choice([40,50,60,80,100,120])
        env.process(spawn(env, f"M{i}", speed, spawn_delay))
        t = spawn_delay

    env.run(until=2000)
    return pd.DataFrame(logs)

# -----------------------
# Main
# -----------------------
def main():
    # 1) Build & train quantile model for remaining prediction
    model = build_and_train_quantile_model()

    # 2) Baseline stats (we can re-run baseline generator if needed)
    df_train = generate_training_logs_with_queue(sim_duration=2000, n_trains=400, seed=1)
    print(f"[Baseline sample] mean travel = {df_train['remaining_min'].mean():.2f} (note: this is remaining sample)")

    # 3) Run conservative model-aware sim
    df_model = model_aware_run_with_conservative(model, n_trains=150, seed=2, hold_threshold_min=0.5, safety_buffer_min=0.1, max_hold_min=2.0)

    # 4) Compare simple KPIs using physics baseline (re-generate simple baseline for comparable trains)
    df_baseline_simple = generate_training_logs_with_queue(sim_duration=2000, n_trains=150, seed=2)
    # compute mean full travel time for baseline_run (we need travel times: but df_baseline_simple currently holds remaining rows).
    # As quick proxy, we approximate travel_min ≈ remaining + elapsed (we don't have elapsed here). Simpler: run a separate baseline sim that logs full travel times:
    # We'll just run a simple baseline sim (like earlier) for direct comparison
    df_baseline_full = generate_baseline_full(sim_duration=2000, n_trains=150, seed=2)

    print("\n=== Comparison ===")
    print(f"Baseline mean travel (full sim): {df_baseline_full['travel_min'].mean():.2f} minutes")
    if not df_model.empty:
        print(f"Conservative model-aware mean travel: {df_model['travel_min'].mean():.2f} minutes")
    else:
        print("Model-aware produced no records.")

# helper to produce full-travel baseline (same as earlier simple generator)
def generate_baseline_full(sim_duration=2000, n_trains=150, seed=2):
    random.seed(seed)
    env = simpy.Environment()
    block_length_km = 5.0
    records = []

    class SimpleBlockFull:
        def __init__(self, env):
            self.env = env
            self.occupied = False
            self.current = None
            self.queue = []

        def request_and_wait(self, train, speed):
            ev = self.env.event()
            self.queue.append((train,speed,ev))
            if not self.occupied:
                self._admit_next()
            return ev

        def _admit_next(self):
            if self.queue and not self.occupied:
                train, speed, ev = self.queue.pop(0)
                self.occupied = True
                self.current = (train, speed, self.env.now)
                if not ev.triggered:
                    ev.succeed()

        def leave(self, train):
            if self.current and self.current[0]==train:
                self.occupied = False
                self.current = None
                self._admit_next()

    def train_proc(env, name, speed, spawn_delay):
        yield env.timeout(spawn_delay)
        ev = block.request_and_wait(name, speed)
        yield ev
        entry = env.now
        actual = travel_time_km(speed, block_length_km) * random.uniform(0.9, 1.4)
        yield env.timeout(actual)
        exit_time = env.now
        records.append({'train':name,'entry_time':entry,'exit_time':exit_time,'speed':speed,'block_length':block_length_km,'travel_min':exit_time-entry})
        block.leave(name)

    block = SimpleBlockFull(env)
    t=0.0
    for i in range(n_trains):
        spawn = t + random.expovariate(1/2.0)
        speed = random.choice([40,50,60,80,100,120])
        env.process(train_proc(env, f"B{i}", speed, spawn))
        t = spawn

    env.run(until=sim_duration)
    return pd.DataFrame(records)

if __name__ == "__main__":
    main()
