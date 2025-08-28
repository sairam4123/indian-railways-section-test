import simpy
import simpy.resources.resource

class Station:
    def __init__(self, env: simpy.Environment, stn_code: str, tracks: list['Track']) -> None:
        self.stn_code = stn_code
        self.tracks: list[Track] = tracks
        self.env = env
        self.train_map: dict['Train', simpy.resources.resource.Request] = {}
        self.connections: dict['Station', list['BlockEdge']] = {}

    def accept(self, train: 'Train'):
        track = self.get_first_available_track(train)
        if track:
            if (req := track.accept(train)):
                # Simulate time taken to reach platform
                yield self.env.timeout(2) # 2 mins
                self.train_map[train] = req
                return req
        return None

    def get_first_available_track(self, train: 'Train'):
        for track in self.tracks:
            if not track.resource.count and track.length >= train.length:
                return track
        return None
    
    def dispatch(self, train: 'Train'):
        if train in self.train_map:
            self.train_map.pop(train)
            # Simulate time taken to leave platform
            yield self.env.timeout(2)
            return True
        
        return False

class Track:
    def __init__(self, env: simpy.Environment, has_platform: bool, length: int) -> None:
        self.resource = simpy.PriorityResource(env, capacity=1)
        self.env = env
        self.has_platform = has_platform
        self.length = length
        self.train: 'Train | None' = None

    def accept(self, train: 'Train'):
        if train.length > self.length:
            return
        self.train = train
        return self.resource.request(priority=train.priority)
    

class BlockEdge:
    def __init__(self, env: simpy.Environment, from_station: Station, to_station: Station, line_speed: int, bidirectional: bool = True) -> None:
        self.from_station = from_station
        self.to_station = to_station
        self.line_speed = line_speed  # in km/h
        self.bidirectional = bidirectional

        self.resource = simpy.PriorityResource(env, capacity=1)

        # one-way only
        self.from_station.connections.setdefault(to_station, []).append(self)
        if bidirectional:
            self.to_station.connections.setdefault(from_station, []).append(self)

    def accept(self, train: 'Train'):
        with self.resource.request(priority=train.priority) as req:
            yield req

class Train:
    def __init__(self, env: simpy.Environment, schedule: list['SchedulePoint'], max_speed: int, priority: int, length: int) -> None:
        self.env = env
        self.schedule = schedule
        self.max_speed = max_speed
        self.priority = priority
        self.length = length
        self.schedule_pointer = 0

    def run(self):
        while self.schedule_pointer < len(self.schedule):
            current_sp = self.schedule[self.schedule_pointer]

            # go to the next schedule point
            self.env.process(self.move_to_schedule_point(current_sp))

            # move to next schedule point
            self.schedule_pointer += 1
    
    def move_to_schedule_point(self, sp: 'SchedulePoint'):
        travel_time = sp.arrival_time - self.env.now
        if travel_time > 0:
            yield self.env.timeout(travel_time)
        
        yield from sp.station.accept(self)


class SchedulePoint:
    def __init__(self, station: Station, arrival_time: int, departure_time: int, expected_platform: int) -> None:
        self.station = station
        self.arrival_time = arrival_time
        self.departure_time = departure_time
        self.expected_platform = expected_platform


env = simpy.Environment()

PDKT_main = Track(env, has_platform=True, length=600)
PDKT_loop1 = Track(env, has_platform=True, length=400)
PDKT_loop2 = Track(env, has_platform=True, length=400)
Pudukkottai = Station(env, 'pdkt', [PDKT_main, PDKT_loop1, PDKT_loop2])


KKDI_main = Track(env, has_platform=True, length=600)
KKDI_loop1 = Track(env, has_platform=True, length=400)
KKDI_loop2 = Track(env, has_platform=True, length=400)
KKDI_loop3 = Track(env, has_platform=True, length=400)
KKDI_loop4 = Track(env, has_platform=True, length=400)
Karaikudi = Station(env, 'kkdi', [KKDI_main, KKDI_loop1, KKDI_loop2, KKDI_loop3, KKDI_loop4])

Pudukkottai_Karaikudi = BlockEdge(env, Pudukkottai, Karaikudi, line_speed=110)
Karaikudi_Pudukkottai = BlockEdge(env, Karaikudi, Pudukkottai, line_speed=110)

sp1 = SchedulePoint(Pudukkottai, 10, 20, 1)
sp2 = SchedulePoint(Karaikudi, 35, 45, 0)

train1 = Train(env, [sp1, sp2], max_speed=110, priority=1, length=300)

train1.run()