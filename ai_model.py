# tiny_book_sim_ml_fixed.py
"""
Tiny end-to-end sim + ML demo (cleaned).

Stages:
1) Generate synthetic training logs using a simple SimPy sim.
2) Train RandomForestRegressor on (speed, block_length) -> travel_time.
3) Run a model-aware sim where a ModelAwareBlock consults the trained model
   to predict total traversal time and schedules a watcher to admit the next train.

Run:
    pip install simpy pandas scikit-learn
    python tiny_book_sim_ml_fixed.py
"""

import simpy
import random
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# -----------------------
# Helpers
# -----------------------
def travel_time_km(speed_kmph, length_km):
    """Return minutes to traverse length_km at speed_kmph."""
    hours = length_km / speed_kmph
    return hours * 60.0

# -----------------------
# Stage 1: Generate training logs (fixed)
# -----------------------
def generate_training_logs(sim_duration=2000, n_trains=300, seed=0):
    """
    Generate training data. Each train appends a complete record only after exit_time is known.
    """
    random.seed(seed)
    env = simpy.Environment()
    block_length_km = 5.0
    logs = []

    def train_proc(env, tid, spawn_delay):
        yield env.timeout(spawn_delay)
        speed = random.choice([40, 50, 60, 80, 100, 120])
        entry_time = env.now
        # travel time with noise
        tt = travel_time_km(speed, block_length_km) * random.uniform(0.9, 1.4)
        yield env.timeout(tt)
        exit_time = env.now
        logs.append({
            'train': tid,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'speed': speed,
            'block_length': block_length_km,
            'travel_min': exit_time - entry_time
        })

    # spawn trains with random interarrival times
    t = 0.0
    for i in range(n_trains):
        spawn = t + random.expovariate(1/2.0)  # mean interarrival 2 minutes
        env.process(train_proc(env, f"T{i}", spawn))
        t = spawn

    env.run(until=sim_duration)
    df = pd.DataFrame(logs)
    # ensure columns predictable order
    df = df[['train', 'entry_time', 'exit_time', 'speed', 'block_length', 'travel_min']]
    return df

# -----------------------
# Stage 2: Train model
# -----------------------
def train_travel_time_model(df):
    X = df[['speed', 'block_length']].copy()
    y = df['travel_min'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"[Model] Trained RandomForestRegressor MAE = {mae:.3f} minutes")
    return model

# -----------------------
# Discrete-event Model-aware Block (single occupancy)
# -----------------------
class DiscreteEventModelAwareBlock:
    """
    Improved: watcher schedules using *predicted remaining* at request time,
    with safety buffer and max_hold cap. Recomputes if new requests arrive.
    Single-occupancy version.
    """
    def __init__(self, env, model=None, length_km=5.0, name="Block",
                 safety_buffer_min=0.0, max_hold_min=3.0):
        self.env = env
        self.model = model
        self.length_km = length_km
        self.name = name

        self.occupied = False
        self.current = None          # {'train','entry_time','speed'}
        self.queue = []              # FIFO list of (train_name, speed, event, req_time)
        self._watcher_proc = None    # currently scheduled watcher (if any)

        self._feature_cols = ['speed', 'block_length']
        self.safety_buffer_min = safety_buffer_min
        self.max_hold_min = max_hold_min

    def _predict_total_time(self, speed):
        if self.model is None:
            return (self.length_km / speed) * 60.0
        X = pd.DataFrame([[speed, self.length_km]], columns=self._feature_cols)
        pred = self.model.predict(X)
        return float(pred[0])

    def request_enter(self, train_name, speed):
        ev = self.env.event()
        self.queue.append((train_name, speed, ev, self.env.now))
        # if block free, try to admit immediately
        self._try_grant()
        # if block occupied and no watcher scheduled, schedule one that uses predicted remaining
        if self.occupied:
            # always (re)compute watcher since new train affects decision
            self._schedule_or_update_watcher()
        return ev

    def leave(self, train_name):
        # actual leaving: free block and try grant
        if not self.occupied:
            return
        if self.current and self.current['train'] == train_name:
            self.occupied = False
            self.current = None
            # cancel watcher if any (it's safe to ignore; watcher will no-op)
            if self._watcher_proc and not self._watcher_proc.triggered:
                try:
                    self._watcher_proc.interrupt()
                except Exception:
                    pass
                self._watcher_proc = None
            self._try_grant()

    def _try_grant(self):
        # if free and queue non-empty -> admit head
        if (not self.occupied) and self.queue:
            train_name, speed, ev, req_time = self.queue.pop(0)
            self.occupied = True
            self.current = {'train': train_name, 'entry_time': self.env.now, 'speed': speed}
            if not ev.triggered:
                ev.succeed()
            # after admitting, if still waiters, schedule watcher to wake based on predicted remaining
            if self.queue:
                self._schedule_or_update_watcher()

    def _schedule_or_update_watcher(self):
        """
        (Re)create the watcher process that waits for predicted remaining time of current occupant.
        If a watcher exists, interrupt it and create a new one (so new arrivals can shorten wait).
        """
        # only schedule if occupied and there are waiters
        if not self.occupied or not self.queue:
            return

        # compute predicted remaining
        cur = self.current
        elapsed = self.env.now - cur['entry_time']
        predicted_total = self._predict_total_time(cur['speed'])
        predicted_remaining = max(predicted_total - elapsed, 0.0)

        # apply safety buffer & cap
        wait_time = predicted_remaining + self.safety_buffer_min
        if wait_time > self.max_hold_min:
            wait_time = self.max_hold_min

        # if a watcher exists, interrupt it (we will reschedule)
        if self._watcher_proc and not self._watcher_proc.triggered:
            try:
                self._watcher_proc.interrupt()
            except Exception:
                pass
            self._watcher_proc = None

        # start a new watcher
        self._watcher_proc = self.env.process(self._watcher(wait_time))

    def _watcher(self, wait_time):
        """Wait for this computed wait_time, then try granting the next train."""
        try:
            yield self.env.timeout(wait_time)
        except simpy.Interrupt:
            # interrupted because new arrivals changed predicted remaining; just exit silently
            return
        finally:
            # clear reference so future watchers can be scheduled
            self._watcher_proc = None

        # after waiting, try to admit next (if occupant already left, leave() did it)
        self._try_grant()

# -----------------------
# Train process that uses ModelAwareBlock
# -----------------------
def train_model_aware(env, name, speed, block: DiscreteEventModelAwareBlock, out_logs):
    ev = block.request_enter(name, speed)
    yield ev  # wait until admitted
    entry_time = env.now
    # predicted travel time (model's total prediction)
    predicted = block._predict_total_time(speed)
    # actual = predicted * some noise to simulate variability
    actual_tt = predicted * random.uniform(0.9, 1.2)
    yield env.timeout(actual_tt)
    exit_time = env.now
    out_logs.append({
        'train': name,
        'entry_time': entry_time,
        'exit_time': exit_time,
        'speed': speed,
        'block_length': block.length_km,
        'travel_min': exit_time - entry_time
    })
    block.leave(name)

# -----------------------
# Model-aware sim runner
# -----------------------
def model_aware_run(model, n_trains=60, seed=2):
    random.seed(seed)
    env = simpy.Environment()
    block = DiscreteEventModelAwareBlock(env, model=model, length_km=5.0, name="Block-ModelAware")
    logs = []

    def spawn_train(env, name, speed, spawn_delay):
        yield env.timeout(spawn_delay)
        yield env.process(train_model_aware(env, name, speed, block, logs))

    t = 0.0
    for i in range(n_trains):
        spawn = t + random.expovariate(1/2.0)
        speed = random.choice([40, 50, 60, 80, 100, 120])
        env.process(spawn_train(env, f"M{i}", speed, spawn))
        t = spawn

    env.run(until=2000)
    df = pd.DataFrame(logs)
    return df

# -----------------------
# Baseline simulation to generate training data wrapper
# -----------------------
def baseline_simulation_and_stats(n_trains=300):
    print("\n=== Running baseline (no model) sim to collect data ===")
    df = generate_training_logs(sim_duration=2000, n_trains=n_trains, seed=1)
    print(f"[Baseline] Generated {len(df)} records, mean travel (min) = {df['travel_min'].mean():.2f}")
    return df

# -----------------------
# Main
# -----------------------
def main():
    # 1) Baseline run -> training data
    df_train = baseline_simulation_and_stats(n_trains=300)
    print(df_train.head())

    # 2) Train model
    model = train_travel_time_model(df_train)

    # 3) Run model-aware sim
    df_model = model_aware_run(model, n_trains=120)

    # 4) Compare simple KPIs
    if not df_model.empty:
        print("\n=== Comparison ===")
        print(f"Baseline mean travel (train logs used for training): {df_train['travel_min'].mean():.2f} minutes")
        print(f"Model-aware mean travel: {df_model['travel_min'].mean():.2f} minutes")
    else:
        print("Model-aware run produced no records (weird).")

if __name__ == "__main__":
    main()
