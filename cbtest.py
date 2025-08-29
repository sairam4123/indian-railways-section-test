import simpy
import simpy.resources.resource

APPROACH_TIME = 2
DEPARTURE_CLEAR_TIME = 2


class Station:
    def __init__(self, env: simpy.Environment, stn_code: str, tracks: list['Track']) -> None:
        self.stn_code = stn_code
        self.tracks: list[Track] = tracks
        self.env = env
        self.train_map: dict['Train', tuple['Track', simpy.resources.resource.PriorityRequest]] = {}
        self.connections: dict['Station', list['BlockEdge']] = {}

    # def accept(self, train: 'Train'):
    #     track = self.get_first_available_track(train)
    #     if track:
    #         if (req := track.accept(train)):
    #             # Simulate time taken to reach platform
    #             yield self.env.timeout(2) # 2 mins
    #             self.train_map[train] = req
    #             return req
    #     return None

    def accept(self, train: 'Train', sp: 'SchedulePoint'):
        """
        Acquire a suitable platform track as soon as the train arrives.
        We do NOT release here; we keep the request in train_map for live view + later dispatch().
        """
        # choose track: prefer expected platform if valid & fits; else first available that fits
        track = None
        if 0 <= sp.expected_platform < len(self.tracks):
            cand = self.tracks[sp.expected_platform]
            if cand.length >= train.length:
                track = cand

        if track is None or track.resource.count > 0: # Track is not valid or occupied
            track = self.get_first_available_track(train)
            if track is None:
                # all suitable tracks busy -> queue on any suitable track; pick the first that fits
                for t in self.tracks:
                    if t.length >= train.length:
                        track = t
                        break
                if track is None:
                    raise RuntimeError(f"No track long enough for train {train} at {self.stn_code}")


        # Request the platform (PriorityResource). This will queue if busy.
        req = track.resource.request(priority=train.priority)
        yield req  # WAIT here until platform is granted

        # (optional) mark live state
        self.train_map[train] = (track, req)

        train.log.log(f"Accepted {self.stn_code} on track {track.id}")
        # print(f"{self.env.now}: Accepted {self.stn_code} - {train.id} on track {track.id}")
        # simulate approach to the platform starter (don’t release request!)
        yield self.env.timeout(APPROACH_TIME)

        # done: the train is now occupying the platform track.
        return req  # keep request handle in case caller wants it too


    def get_first_available_track(self, train: 'Train'):
        for track in self.tracks:
            if not track.resource.count and track.length >= train.length:
                return track
        return None
    
    def get_first_available_block(self, train: 'Train', sp: 'SchedulePoint'):
        next_sp = train.get_next_schedule_point(sp)
        if next_sp is None:
            return None
        for block in self.connections.get(next_sp.station, []):
            if not block.resource.count > 0:
                return block
        return None
    
    def get_block_to(self, next_station: 'Station') -> 'BlockEdge | None':
        lst = self.connections.get(next_station, [])
        # return the first available block
        for block in lst:
            if not block.resource.count > 0:
                return block
        # if no block is available, return the first block
        return lst[0] if lst else None

    def dispatch(self, train: 'Train'):
        """Release the platform after departure, including a small clear time."""
        entry = self.train_map.get(train)
        if not entry:
            return False
        track, req = entry

        # simulate shunt/starting and clearing the platform starter
        yield self.env.timeout(DEPARTURE_CLEAR_TIME)

        # release platform and update live map
        track.resource.release(req)
        self.train_map.pop(train, None)

        return True

class Track:
    def __init__(self, env: simpy.Environment, id: str, has_platform: bool, length: int) -> None:
        self.resource = simpy.PriorityResource(env, capacity=1)
        self.env = env
        self.has_platform = has_platform
        self.length = length
        self.id = id

        # self.train: 'Train | None' = None

    # def accept(self, train: 'Train'):
    #     if train.length > self.length:
    #         return
    #     self.train = train
    #     return self.resource.request(priority=train.priority)
    

class BlockEdge:
    def __init__(self, env: simpy.Environment, 
                 name: str, from_station: Station, to_station: Station, 
                 line_speed: int, length_km: int, headway_min: int, 
                 bidirectional: bool = False, 
                 electric: bool = False) -> None:
        self.name = name
        self.from_station = from_station
        self.to_station = to_station
        self.line_speed = line_speed  # in km/h
        self.length_km = length_km
        self.headway_min = headway_min
        self.bidirectional = bidirectional
        self.env = env
        self.electric = electric

        self.resource = simpy.PriorityResource(env, capacity=1)
        # one-way only
        self.from_station.connections.setdefault(to_station, []).append(self)
        if bidirectional:
            self.to_station.connections.setdefault(from_station, []).append(self)

        self.occupied_train: tuple['Train', simpy.resources.resource.PriorityRequest] | None = None

    def _run_minutes(self, train: 'Train') -> int:
        """Compute run time in minutes based on train max speed, line speed, length, acceleration profile."""
        # Compute time to accelerate to max speed, cruise, then decelerate.
        speed = max(1, min(self.line_speed, train.max_speed))  # km/h
        accel = 0.5 * (1.2 if self.electric else 1)  # m/s^2 (typical for passenger trains)
        vmax = speed * 1000 / 3600  # convert km/h to m/s
        L = self.length_km * 1000   # meters

        t_accel = vmax / accel
        d_accel = 0.5 * accel * t_accel ** 2

        if 2 * d_accel >= L:
            # Not enough distance to reach vmax, use triangular profile
            t = (2 * (L / accel)) ** 0.5
            total_time = t
        else:
            d_cruise = L - 2 * d_accel
            t_cruise = d_cruise / vmax
            total_time = 2 * t_accel + t_cruise

        return int(round(total_time / 60))  # minutes

    def traverse(self, train: 'Train', force_minutes: int | None = None):
        """Acquire -> run (or force_minutes) -> headway -> release, while holding the lock."""
        req = self.resource.request(priority=train.priority)
        yield req
        self.occupied_train = (train, req)
        try:
            run = self._run_minutes(train) if force_minutes is None else force_minutes
            yield self.env.timeout(run)              # inside block
            yield self.env.timeout(self.headway_min) # clearance
        finally:
            self.resource.release(req)
            self.occupied_train = None

    def enter(self, train: 'Train'):
        req = self.resource.request(priority=train.priority)
        yield req
        self.occupied_train = (train, req)

    def exit(self, train: 'Train'):

        occupied_train, req = self.occupied_train if self.occupied_train else (None, None)

        if occupied_train and req and occupied_train == train:
            self.occupied_train = None
            self.resource.release(req)
        yield self.env.timeout(0)  # just to make this a generator
        return 

class Train:
    def __init__(self, env: simpy.Environment, id: str, schedule: list['SchedulePoint'], max_speed: int, priority: int, length: int) -> None:
        self.env = env
        self.id = id
        self.schedule = schedule
        self.max_speed = max_speed
        self.priority = priority
        self.length = length
        self.schedule_pointer = 0
        self.log = TrainLog(self)
        self.current_block: BlockEdge | None = None

        env.process(self.run())

    def run(self):

        current_sp = self.schedule[self.schedule_pointer]
        map_enter_delay = current_sp.arrival_time - self.env.now
        if map_enter_delay > 0:
            yield self.env.timeout(map_enter_delay)

        while self.schedule_pointer < len(self.schedule):
            current_sp = self.schedule[self.schedule_pointer]

            # print(f"{self.env.now}: Approaching {current_sp.station.stn_code} - {self.id}")
            # Exit previous block if any (you held it until arriving home)
            # if self.current_block:
            #     # If you used traverse(), you won't need this; keeping for compatibility:
            #     yield from self.current_block.exit(self)
            #     self.current_block = None

            # print(f"{self.env.now}: Entering into {current_sp.station.stn_code} - {self.id}")
            # Arrive and occupy platform (even if early)
            yield from self.move_to_schedule_point(current_sp)

            # Dwell/layover
            dwell_time = current_sp.departure_time - self.env.now
            if dwell_time > 0:
                yield self.env.timeout(dwell_time)

            # Look up the next station & block
            next_sp = self.get_next_schedule_point(current_sp)
            if next_sp is None:
                # terminal: dispatch to yard or just clear platform
                self.log.log(f"Terminating at {current_sp.station.stn_code}")
                # print(f"{self.env.now}: Terminating at {current_sp.station.stn_code} - {self.id}")
                # clear the platform
                yield from current_sp.station.dispatch(self)
                break

            block = current_sp.station.get_block_to(next_sp.station)
            if block is None:
                raise RuntimeError(f"No block from {current_sp.station.stn_code} to {next_sp.station.stn_code}")

            # IMPORTANT: Reserve the next block while still on the platform.
            # We'll queue here if needed, without fouling the main line.
            # Option 1: one-shot traversal with computed schedule-based run:
            # If you want to hit the schedule arrival exactly:
            # planned_run = max(0, next_sp.arrival_time - (self.env.now + DEPARTURE_CLEAR_TIME))
            # Option 2: use physics-based running time instead (recommended):
            # planned_run = None  # let block compute from speed/length

            # Now we dispatch (release platform) *only when the block is granted*, i.e., right before we start moving.
            # We'll do it like this: pre-acquire by calling traverse(), but we need to dispatch just before moving.
            # So we split traverse() into acquire+run+release by using a small helper:

            # Acquire the block, but don't start timing until we leave platform:
            # We can emulate this by first acquiring, then dispatch, then run+headway while holding.
            req = block.resource.request(priority=self.priority)
            yield req
            block.occupied_train = (self, req)  # mark for live view if you want

            # Leave the platform now (starter clears)
            # print(f"{self.env.now}: Departing {current_sp.station.stn_code} - {self.id} - {block.name}")
            self.log.log(f"Departing {current_sp.station.stn_code} towards {next_sp.station.stn_code} via {block.name}")
            yield from current_sp.station.dispatch(self)

            # Run inside the block while holding the lock
            try:
                run = block._run_minutes(self)
                yield self.env.timeout(run)
                yield self.env.timeout(block.headway_min)
            finally:
                self.log.log(f"Exited {block.name} - approaching {next_sp.station.stn_code}")
                # print(f"{self.env.now}: Approaching {next_sp.station.stn_code} - {self.id}")
                block.resource.release(req)
                block.occupied_train = None
                yield self.env.timeout(2)  # Allow time for the block to clear

            # Next station’s platform will be taken in the next loop iteration by move_to_schedule_point(next_sp)
            self.current_block = None
            self.schedule_pointer += 1
        
    def move_to_schedule_point(self, sp: 'SchedulePoint'):
        # travel_time = sp.arrival_time - self.env.now
        # if travel_time > 0:
        #     yield self.env.timeout(travel_time)
        self.log.log(f"Entering into {sp.station.stn_code}")

        yield from sp.station.accept(self, sp)
    
    def get_next_schedule_point(self, sp: 'SchedulePoint') -> 'SchedulePoint | None':
        return self.schedule[self.schedule_pointer + 1] if self.schedule_pointer + 1 < len(self.schedule) else None

    def schedule_stop(self, station: Station, arrival_time: int, departure_time: int, expected_platform: int) -> 'SchedulePoint':
        sp = SchedulePoint(station, arrival_time, departure_time, expected_platform)
        self.schedule.append(sp)
        return sp


class SchedulePoint:
    def __init__(self, station: Station, arrival_time: int, departure_time: int, expected_platform: int) -> None:
        self.station = station
        self.arrival_time = arrival_time
        self.departure_time = departure_time
        self.expected_platform = expected_platform


class TrainLog:
    def __init__(self, train: Train) -> None:
        self.train = train
        self.entries = []

    def log(self, message: str) -> None:
        self.entries.append((self.train.env.now, self.train.id, message))

env = simpy.Environment()

PDKT_main = Track(env, "PDKT_main", has_platform=True, length=600)
PDKT_loop1 = Track(env, "PDKT_loop1", has_platform=True, length=400)
PDKT_loop2 = Track(env, "PDKT_loop2", has_platform=True, length=400)
PDKT = Station(env, 'pdkt', [PDKT_main, PDKT_loop1, PDKT_loop2])


KKDI_main = Track(env, "KKDI_main", has_platform=True, length=600)
KKDI_loop1 = Track(env, "KKDI_loop1", has_platform=True, length=400)
KKDI_loop2 = Track(env, "KKDI_loop2", has_platform=True, length=400)
KKDI_loop3 = Track(env, "KKDI_loop3", has_platform=True, length=400)
KKDI_loop4 = Track(env, "KKDI_loop4", has_platform=True, length=400)
KKDI = Station(env, 'kkdi', [KKDI_main, KKDI_loop1, KKDI_loop2, KKDI_loop3, KKDI_loop4])

TPJ_main = Track(env, "TPJ_main", has_platform=True, length=600)
TPJ_loop1 = Track(env, "TPJ_loop1", has_platform=True, length=400)
TPJ_loop2 = Track(env, "TPJ_loop2", has_platform=True, length=400)
TPJ = Station(env, 'tpj', [TPJ_main, TPJ_loop1, TPJ_loop2])

# PDKT_KKDI = BlockEdge(env, "PDKT_KKDI", PDKT, KKDI, headway_min=2, length_km=60, line_speed=110, bidirectional=False)
# KKDI_PDKT = BlockEdge(env, "KKDI_PDKT", KKDI, PDKT, headway_min=2, length_km=60, line_speed=110, bidirectional=False)

# PDKT_TPJ = BlockEdge(env, "PDKT_TPJ", PDKT, TPJ, headway_min=3, length_km=80, line_speed=100, bidirectional=False)
# TPJ_PDKT = BlockEdge(env, "TPJ_PDKT", TPJ, PDKT, headway_min=3, length_km=80, line_speed=100, bidirectional=False)

TPJ_PDKT_3rd = BlockEdge(env, "TPJ_PDKT_3", TPJ, PDKT, headway_min=3, length_km=80, line_speed=100, bidirectional=True, electric=True)
PDKT_KKDI_3rd = BlockEdge(env, "PDKT_KKDI_3", PDKT, KKDI, headway_min=2, length_km=60, line_speed=110, bidirectional=True, electric=True)

# Existing trains
train1 = Train(env, "T1", [], max_speed=110, priority=1, length=300)
train1.schedule_stop(TPJ, 0, 10, 1)
train1.schedule_stop(PDKT, 70, 75, 1)
train1.schedule_stop(KKDI, 110, 115, 0)

train2 = Train(env, "T2", [], max_speed=110, priority=2, length=300)
train2.schedule_stop(KKDI, 10, 20, 1)
train2.schedule_stop(PDKT, 55, 65, 0)
train2.schedule_stop(TPJ, 110, 115, 1)

train3 = Train(env, "T3", [], max_speed=100, priority=1, length=280)
train3.schedule_stop(TPJ, 20, 25, 1)
train3.schedule_stop(PDKT, 85, 90, 0)
train3.schedule_stop(KKDI, 125, 130, 1)

train4 = Train(env, "T4", [], max_speed=110, priority=2, length=320)
train4.schedule_stop(KKDI, 40, 45, 1)
train4.schedule_stop(PDKT, 95, 100, 1)
train4.schedule_stop(TPJ, 150, 155, 0)

train5 = Train(env, "T5", [], max_speed=90, priority=3, length=250)
train5.schedule_stop(TPJ, 60, 65, 0)
train5.schedule_stop(PDKT, 120, 125, 1)
train5.schedule_stop(KKDI, 170, 175, 1)

train6 = Train(env, "T6", [], max_speed=110, priority=1, length=300)
train6.schedule_stop(KKDI, 75, 80, 0)
train6.schedule_stop(PDKT, 130, 135, 1)
train6.schedule_stop(TPJ, 185, 190, 1)

train7 = Train(env, "T7", [], max_speed=100, priority=2, length=270)
train7.schedule_stop(TPJ, 95, 100, 1)
train7.schedule_stop(PDKT, 150, 155, 0)
train7.schedule_stop(KKDI, 200, 205, 1)

env.run()

trains = [train1, train2, train3, train4, train5, train6, train7]
train_logs = sum([train.log.entries for train in trains], [])

train_logs.sort(key=lambda x: (x[0], x[1]))
print("Time\tTrain\tEvent")
for train_log in train_logs:
    print(f"{train_log[0]}\t{train_log[1]}\t{train_log[2]}")

import matplotlib.pyplot as plt

# ---------- (1) GANTT CHART ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))  # side by side
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))  # top & bottom (use this instead if you prefer)

ax1.set_title("Train Schedule Gantt Chart")

def split_text(text, max_chars=20):
    words = text.split()
    lines, current_line = [], ""
    for word in words:
        if len(current_line) + len(word) + 1 <= max_chars:
            current_line += (" " + word if current_line else word)
        else:
            lines.append(current_line)
            current_line = word
    if current_line: lines.append(current_line)
    return "\n".join(lines)

for i, train in enumerate(trains):
    color = plt.cm.tab10.colors[i % 20]
    entries = train.log.entries
    for idx in range(len(entries) - 1):
        start = entries[idx][0]
        end = entries[idx + 1][0]
        message = entries[idx][2]

        ax1.barh(i, end - start, left=start, color=color, edgecolor='black', alpha=0.7)
        ax1.text(start + (end - start) / 2, i, split_text(message, (end-start)/2),
                 va='center', ha='center', fontsize=8, color='black')

ax1.set_xlabel("Time")
ax1.set_ylabel("Train")
ax1.set_yticks(range(len(trains)))
ax1.set_yticklabels([train.id for train in trains])
handles = [plt.Rectangle((0, 0), 1, 1, color=plt.cm.tab10.colors[i % 20]) for i in range(len(trains))]
ax1.legend(handles, [train.id for train in trains], title="Trains")
ax1.grid(True, which="both", axis="x")

# ---------- (2) TRAIN GRAPH ----------
stations = ["tpj", "pdkt", "kkdi"]
xpos = {st: i for i, st in enumerate(stations)}

edge_spans = {}
for train in trains:
    spans = []
    entries = train.log.entries
    for i in range(len(entries) - 1):
        t0, tid, msg0 = entries[i]
        t1, _, msg1 = entries[i+1]

        # --- dwell detection ---
        if "Entering" in msg0 and "Accepted" in msg1:
            # Train is dwelling inside platform (between accepted & dispatch)
            station = msg0.split()[2]
            spans.append((station, station, t0, t1, "dwell"))

        if "Accepted" in msg0 and "Departing" in msg1:
            station = msg0.split()[1]
            spans.append((station, station, t0, t1, "dwell"))

        # --- run detection ---
        elif "Departing" in msg0 and "Exited" in msg1:
            u = msg0.split()[1]           # current station
            v = msg1.split()[-1]          # next station
            spans.append((u, v, t0, t1, "run"))

        # --- terminate detection ---
        elif "Accepted" in msg0 and "Terminating" in msg1:
            u = msg0.split()[1]           # current station
            spans.append((u, u, t0, t1, "dwell"))

    edge_spans[train.id] = spans

colors = plt.cm.tab20.colors
color_map = {train.id: colors[i % len(colors)] for i, train in enumerate(trains)}

for train in trains:
    tid = train.id
    col = color_map[tid]
    for j, (u, v, t0, t1, kind) in enumerate(edge_spans[tid]):
        xs = [xpos[u], xpos[v]]
        ys = [t0, t1]
        style = "solid" if kind == "run" else "dashed"
        ax2.plot(xs, ys, marker="o", linewidth=2, color=col, linestyle=style,
                 label=tid if j == 0 else None)
        # annotate arrival & departure
        ax2.text(xs[0], ys[0], f"{ys[0]}", fontsize=7, va="bottom", ha="right")
        ax2.text(xs[1], ys[1], f"{ys[1]}", fontsize=7, va="bottom", ha="left")

ax2.set_xticks(range(len(stations)))
ax2.set_xticklabels(stations)
ax2.set_ylabel("Time (minutes)")
ax2.set_xlabel("Stations (left to right)")
ax2.set_title("Train Graph with Runs + Dwells")
ax2.invert_yaxis()
ax2.grid(True, which="both", axis="both")
ax2.legend()

plt.tight_layout()
plt.show()
