# tiny_book_sim_ml_fixed_v2.py
"""
Clean sim + ML demo with improved model-aware admission.

Run:
    pip install simpy pandas scikit-learn
    python tiny_book_sim_ml_fixed_v2.py
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
# Stage 1: Generate training logs (same as earlier)
# -----------------------
def generate_training_logs(sim_duration=2000, n_trains=300, seed=0):
    """
    Generate training data: each train appends a full record after exit (no race).
    This produces features: speed, block_length -> travel_min
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

    t = 0.0
    for i in range(n_trains):
        spawn = t + random.expovariate(1/2.0)
        env.process(train_proc(env, f"T{i}", spawn))
        t = spawn

    env.run(until=sim_duration)
    df = pd.DataFrame(logs)
    df = df[['train', 'entry_time', 'exit_time', 'speed', 'block_length', 'travel_min']]
    return df

# -----------------------
# Stage 2: Train simple model (same features as above)
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
# Improved Discrete-event Block
# -----------------------
class ImprovedModelAwareBlock:
    """
    Single-occupancy block that:
      - admits next train immediately if free
      - when occupied and there are waiters, schedules a watcher to wait predicted_remaining + buffer
      - rescales/cancels watcher if new arrivals change predicted remaining
      - caps holding time with max_hold_min (safety)
    """
    def __init__(self, env: simpy.Environment, model=None, length_km=5.0, name="Block",
                 safety_buffer_min=0.0, max_hold_min=1.0):
        self.env = env
        self.model = model
        self.length_km = length_km
        self.name = name

        self.occupied = False
        self.current = None          # {'train','entry_time','speed'}
        # queue = list of tuples (train_name, speed, event, req_time)
        self.queue = []
        self._watcher_proc = None    # current watcher process
        self._feature_cols = ['speed', 'block_length']
        self.safety_buffer_min = safety_buffer_min
        self.max_hold_min = max_hold_min

    def _predict_total_time(self, speed):
        """Predict total traversal time (minutes) using model or fallback physics."""
        if self.model is None:
            return (self.length_km / speed) * 60.0
        X = pd.DataFrame([[speed, self.length_km]], columns=self._feature_cols)
        pred = self.model.predict(X)
        return float(pred[0])

    def request_enter(self, train_name, speed):
        """
        Called by a train to request entry. Returns an event the train yields on.
        """
        ev = self.env.event()
        self.queue.append((train_name, speed, ev, self.env.now))
        # try immediate admit if free
        self._try_grant()
        # if occupied and there are waiters, (re)schedule watcher based on predicted remaining
        if self.occupied and self.queue:
            self._schedule_or_update_watcher()
        return ev

    def leave(self, train_name):
        """
        Called when an occupant actually leaves. Frees the block and tries to admit next.
        """
        if not self.occupied:
            return
        if self.current and self.current['train'] == train_name:
            self.occupied = False
            self.current = None
            # cancel watcher if present
            if self._watcher_proc and not getattr(self._watcher_proc, 'triggered', False):
                try:
                    self._watcher_proc.interrupt()
                except Exception:
                    pass
                self._watcher_proc = None
            self._try_grant()

    def _try_grant(self):
        """
        If block free and queue non-empty, admit head of queue.
        After admitting, if more waiters remain, schedule a watcher.
        """
        if (not self.occupied) and self.queue:
            train_name, speed, ev, req_time = self.queue.pop(0)
            self.occupied = True
            self.current = {'train': train_name, 'entry_time': self.env.now, 'speed': speed}
            if not ev.triggered:
                ev.succeed()
            # if others waiting, schedule watcher
            if self.queue:
                self._schedule_or_update_watcher()

    def _schedule_or_update_watcher(self):
        """
        Compute predicted remaining for current occupant and schedule a watcher that waits
        predicted_remaining + safety_buffer but no more than max_hold_min.
        If watcher exists, interrupt and reschedule (so new arrivals can shorten the wait).
        """
        if not self.occupied or not self.queue:
            return

        cur = self.current
        elapsed = self.env.now - cur['entry_time']
        predicted_total = self._predict_total_time(cur['speed'])
        predicted_remaining = max(predicted_total - elapsed, 0.0)

        # apply safety buffer and cap
        wait_time = predicted_remaining + self.safety_buffer_min
        if wait_time > self.max_hold_min:
            wait_time = self.max_hold_min

        # interrupt existing watcher if any (reschedule)
        if self._watcher_proc and not getattr(self._watcher_proc, 'triggered', False):
            try:
                self._watcher_proc.interrupt()
            except Exception:
                pass
            self._watcher_proc = None

        # start new watcher
        self._watcher_proc = self.env.process(self._watcher(wait_time))

    def _watcher(self, wait_time):
        """
        Wait for 'wait_time' (sim time). If interrupted (new arrivals) it simply ends
        and the caller will reschedule. If it completes, it calls _try_grant() to admit next.
        """
        try:
            yield self.env.timeout(wait_time)
        except simpy.Interrupt:
            return
        finally:
            self._watcher_proc = None

        # after waiting, attempt to admit next train
        self._try_grant()

# -----------------------
# Train process that uses the ImprovedModelAwareBlock
# -----------------------
def train_model_aware(env, name, speed, block: ImprovedModelAwareBlock, out_logs):
    ev = block.request_enter(name, speed)
    yield ev  # wait until admitted
    entry_time = env.now
    # predicted total (model)
    predicted = block._predict_total_time(speed)
    # actual = predicted * random noise
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
# Model-aware run wrapper
# -----------------------
def model_aware_run(model, n_trains=120, seed=2, safety_buffer_min=0.2, max_hold_min=2.0):
    random.seed(seed)
    env = simpy.Environment()
    block = ImprovedModelAwareBlock(env, model=model, length_km=5.0, name="Block-Improved",
                                    safety_buffer_min=safety_buffer_min, max_hold_min=max_hold_min)
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
# Baseline generator & main
# -----------------------
def baseline_sim_data(n_trains=300):
    print("\n=== Running baseline (no model) sim to collect data ===")
    df = generate_training_logs(sim_duration=2000, n_trains=n_trains, seed=1)
    print(f"[Baseline] Generated {len(df)} records, mean travel (min) = {df['travel_min'].mean():.2f}")
    return df

def main():
    # 1) baseline data
    df_train = baseline_sim_data(n_trains=300)
    print(df_train.head())

    # 2) train model
    model = train_travel_time_model(df_train)
    print("Model trained.")

    # 3) run improved model-aware with conservative params
    df_model = model_aware_run(model, n_trains=120, seed=2, safety_buffer_min=0.2, max_hold_min=2.0)

    # 4) compare metrics
    print("\n=== Comparison ===")
    print(f"Baseline mean travel (train logs used for training): {df_train['travel_min'].mean():.2f} minutes")
    if not df_model.empty:
        print(f"Improved model-aware mean travel: {df_model['travel_min'].mean():.2f} minutes")
    else:
        print("Improved model-aware produced no records (weird).")

if __name__ == "__main__":
    main()
