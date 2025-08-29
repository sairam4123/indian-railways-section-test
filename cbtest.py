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

        print(f"{self.env.now}: Accepted {self.stn_code} - {train.id} on track {track.id}")
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
    def __init__(self, env: simpy.Environment, name: str, from_station: Station, to_station: Station, line_speed: int, length_km: int, headway_min: int, bidirectional: bool = False) -> None:
        self.name = name
        self.from_station = from_station
        self.to_station = to_station
        self.line_speed = line_speed  # in km/h
        self.length_km = length_km
        self.headway_min = headway_min
        self.bidirectional = bidirectional
        self.env = env

        self.resource = simpy.PriorityResource(env, capacity=1)
        # one-way only
        self.from_station.connections.setdefault(to_station, []).append(self)
        if bidirectional:
            self.to_station.connections.setdefault(from_station, []).append(self)

        self.occupied_train: tuple['Train', simpy.resources.resource.PriorityRequest] | None = None

    def _run_minutes(self, train: 'Train') -> int:
        speed = max(1, min(self.line_speed, train.max_speed))  # km/h
        return int(round((self.length_km / speed) * 60))       # minutes

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
        self.current_block: BlockEdge | None = None

        env.process(self.run())

    def run(self):
        while self.schedule_pointer < len(self.schedule):
            current_sp = self.schedule[self.schedule_pointer]

            # print(f"{self.env.now}: Approaching {current_sp.station.stn_code} - {self.id}")
            # Exit previous block if any (you held it until arriving home)
            # if self.current_block:
            #     # If you used traverse(), you won't need this; keeping for compatibility:
            #     yield from self.current_block.exit(self)
            #     self.current_block = None

            print(f"{self.env.now}: Entering into {current_sp.station.stn_code} - {self.id}")
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
                print(f"{self.env.now}: Terminating at {current_sp.station.stn_code} - {self.id}")
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
            planned_run = max(0, next_sp.arrival_time - (self.env.now + DEPARTURE_CLEAR_TIME))
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
            print(f"{self.env.now}: Departing {current_sp.station.stn_code} - {self.id} - {block.name}")
            yield from current_sp.station.dispatch(self)

            # Run inside the block while holding the lock
            try:
                run = planned_run if planned_run is not None else block._run_minutes(self)
                yield self.env.timeout(run)
                yield self.env.timeout(block.headway_min)
            finally:
                print(f"{self.env.now}: Approaching {next_sp.station.stn_code} - {self.id}")
                block.resource.release(req)
                block.occupied_train = None

            # Next station’s platform will be taken in the next loop iteration by move_to_schedule_point(next_sp)
            self.current_block = None
            self.schedule_pointer += 1
        
    def move_to_schedule_point(self, sp: 'SchedulePoint'):
        travel_time = sp.arrival_time - self.env.now
        if travel_time > 0:
            yield self.env.timeout(travel_time)
        
        yield from sp.station.accept(self, sp)
    
    def get_next_schedule_point(self, sp: 'SchedulePoint') -> 'SchedulePoint | None':
        return self.schedule[self.schedule_pointer + 1] if self.schedule_pointer + 1 < len(self.schedule) else None


class SchedulePoint:
    def __init__(self, station: Station, arrival_time: int, departure_time: int, expected_platform: int) -> None:
        self.station = station
        self.arrival_time = arrival_time
        self.departure_time = departure_time
        self.expected_platform = expected_platform


env = simpy.Environment()

PDKT_main = Track(env, "PDKT_main", has_platform=True, length=600)
PDKT_loop1 = Track(env, "PDKT_loop1", has_platform=True, length=400)
PDKT_loop2 = Track(env, "PDKT_loop2", has_platform=True, length=400)
Pudukkottai = Station(env, 'pdkt', [PDKT_main, PDKT_loop1, PDKT_loop2])


KKDI_main = Track(env, "KKDI_main", has_platform=True, length=600)
KKDI_loop1 = Track(env, "KKDI_loop1", has_platform=True, length=400)
KKDI_loop2 = Track(env, "KKDI_loop2", has_platform=True, length=400)
KKDI_loop3 = Track(env, "KKDI_loop3", has_platform=True, length=400)
KKDI_loop4 = Track(env, "KKDI_loop4", has_platform=True, length=400)
Karaikudi = Station(env, 'kkdi', [KKDI_main, KKDI_loop1, KKDI_loop2, KKDI_loop3, KKDI_loop4])

# Pudukkottai_Karaikudi = BlockEdge(env, "Pudukkottai_Karaikudi", Pudukkottai, Karaikudi, headway_min=2, length_km=60, line_speed=110, bidirectional=False)
# Karaikudi_Pudukkottai = BlockEdge(env, "Karaikudi_Pudukkottai", Karaikudi, Pudukkottai, headway_min=2, length_km=60, line_speed=110, bidirectional=False)
Pudukkottai_Karaikudi_3rd = BlockEdge(env, "Pudukkottai_Karaikudi_3rd", Pudukkottai, Karaikudi, headway_min=2, length_km=60, line_speed=110, bidirectional=True)

# Existing trains
sp1 = SchedulePoint(Pudukkottai, 10, 20, 1)
sp2 = SchedulePoint(Karaikudi, 20, 30, 0)
train1 = Train(env, "T1", [sp1, sp2], max_speed=110, priority=1, length=300)

sp4 = SchedulePoint(Karaikudi, 10, 20, 1)
sp3 = SchedulePoint(Pudukkottai, 15, 20, 0)
train2 = Train(env, "T2", [sp4, sp3], max_speed=110, priority=2, length=300)

# Additional trains
# T3: Later up train (Pudukkottai -> Karaikudi) with lower priority
sp5 = SchedulePoint(Pudukkottai, 25, 35, 1)
sp6 = SchedulePoint(Karaikudi, 35, 45, 0)
train3 = Train(env, "T3", [sp5, sp6], max_speed=100, priority=3, length=250)

# T4: Express down train (Karaikudi -> Pudukkottai), higher priority
sp7 = SchedulePoint(Karaikudi, 18, 25, 1)
sp8 = SchedulePoint(Pudukkottai, 30, 40, 0)
train4 = Train(env, "T4", [sp7, sp8], max_speed=120, priority=0, length=350)

# T5: Short local shuttle (Pudukkottai -> Karaikudi), overlaps with T3
sp9 = SchedulePoint(Pudukkottai, 28, 38, 1)
sp10 = SchedulePoint(Karaikudi, 38, 50, 0)
train5 = Train(env, "T5", [sp9, sp10], max_speed=90, priority=4, length=200)

# T6: Late night down train (Karaikudi -> Pudukkottai)
sp11 = SchedulePoint(Karaikudi, 40, 50, 1)
sp12 = SchedulePoint(Pudukkottai, 55, 65, 0)
train6 = Train(env, "T6", [sp11, sp12], max_speed=100, priority=2, length=300)

env.run()