# import simpy
# import math
# import matplotlib.pyplot as plt

# # -----------------------------
# # Topology (stations as nodes)
# # -----------------------------
# stations = ["Tiruchy", "Pudukkottai", "Chettinadu", "Karaikudi", "Devakottai"]
# edges = [("Tiruchy","Pudukkottai", 100), ("Pudukkottai","Chettinadu", 80), ("Chettinadu","Karaikudi", 100), ("Karaikudi","Devakottai", 90)]  # single-track segments

# # Single-track travel times per train type (minutes)
# RUN = {"express": 5, "passenger": 6, "freight": 8}
# HEADWAY = 2  # clearance after edge traversal (min)

# # Priority ranking (lower number wins because SimPy PriorityResource is min-heap)
# CLASS_PRIO = {"express": 0, "passenger": 1, "freight": 2}

# # -----------------------------
# # Trains (6 mixed)
# # dir: "up" = A->C, "down" = C->A
# # path is the station list they will traverse
# # -----------------------------
# def path_for(direction):
#     return stations if direction=="up" else list(reversed(stations))

# TRAINS = [
#     {"id": "T1", "dir": "up",   "sched":  0, "type": "express"},
#     {"id": "T2", "dir": "down", "sched":  5, "type": "freight"},
#     {"id": "T3", "dir": "up",   "sched": 10, "type": "passenger"},
#     {"id": "T4", "dir": "up", "sched": 12, "type": "express"},
#     {"id": "T5", "dir": "up",   "sched": 15, "type": "freight"},
#     {"id": "T6", "dir": "down", "sched": 20, "type": "passenger"},
#     {"id": "T7", "dir": "down",   "sched":  3, "type": "express"},
#     {"id": "T8", "dir": "up", "sched":  5, "type": "freight"},
#     {"id": "T9", "dir": "down",   "sched": 18, "type": "passenger"},
#     {"id": "T10", "dir": "down", "sched": 24, "type": "express"},
#     {"id": "T11", "dir": "up",   "sched": 28, "type": "freight"},
#     {"id": "T12", "dir": "down", "sched": 30, "type": "passenger"},


# ]
# for t in TRAINS:
#     t["path"] = path_for(t["dir"])

# # -----------------------------
# # Sim objects: edges, stations
# # -----------------------------
# def edge_name(u,v):
#     return f"{u}-{v}" if stations.index(u) < stations.index(v) else f"{v}-{u}"

# # PriorityResources for edges (single-track, capacity=1)
# def build_edges(env):
#     return {edge_name(u,v): simpy.PriorityResource(env, capacity=1) for (u,v, _) in edges}

# def build_speed_edges():
#     return {edge_name(u,v): mps for (u,v, mps) in edges}

# # Stations/loops can host multiple trains (we model only a simple “platform capacity”) [99 = inf]
# STATION_CAP = {"Tiruchy": 99, "Pudukkottai": 4, "Chettinadu": 2, "Karaikudi": 4, "Devakottai": 99}

# def build_stations(env):
#     return {s: simpy.Resource(env, capacity=STATION_CAP[s]) for s in stations}

# # -----------------------------
# # Helpers
# # -----------------------------
# def travel_time(train_type, u, v):
#     # simple constant, could be per-edge later
#     return RUN[train_type] * 60 / MPS[edge_name(u,v)]

# def precedence_tuple(train, now):
#     """
#     Lower tuple is higher priority:
#     (classPriority, -lateness, scheduled_arrival_rank)
#     """
#     cls = CLASS_PRIO[train["type"]]
#     # lateness = how long past sched the train has already existed in system
#     lateness = max(0, now - train["sched"])
#     # scheduled order as weak tiebreak (lower is earlier sched)
#     sched_rank = train["sched"]
#     return (cls, -lateness, sched_rank)

# # -----------------------------
# # Logging for plotting
# # We’ll log station arrival/departure times to draw Y=time, X=station
# # -----------------------------
# station_times = {t["id"]: [] for t in TRAINS}  # list of (station, t_arr, t_dep)
# edge_spans   = {t["id"]: [] for t in TRAINS}  # list of (u, v, t_start, t_end)

# # -----------------------------
# # Processes
# # -----------------------------
# def train_process(env: simpy.Environment, train, E, S):
#     tid = train["id"]
#     pth = train["path"]
#     ttype = train["type"]

#     # Wait for scheduled departure at origin station
#     origin = pth[0]
#     yield env.timeout(train["sched"])
#     # Enter origin station
#     with S[origin].request() as sreq:
#         yield sreq
#         t_arr = env.now
#         # minimal dwell at origin (0 for proto)
#         t_dep = env.now
#         station_times[tid].append((origin, t_arr, t_dep))

#     # Traverse edges between stations
#     for i in range(len(pth)-1):
#         u, v = pth[i], pth[i+1]
#         ekey = edge_name(u,v)

#         # Request single-line edge with precedence priority
#         prio = precedence_tuple(train, env.now)
#         with E[ekey].request(priority=prio) as ereq:
#             yield ereq
#             t_start = env.now
#             # release station u immediately (we don’t hold station while in block)
#             # traverse edge
#             run = travel_time(ttype, u, v)
#             yield env.timeout(run)
#             # headway (we keep the edge locked for clearance)
#             yield env.timeout(HEADWAY)
#             t_end = env.now
#             edge_spans[tid].append((u, v, t_start, t_end))

#         # Arrive at station v (may queue if both tracks are full at a loop)
#         with S[v].request() as sreq_v:
#             yield sreq_v
#             t_arr_v = env.now
#             # simple dwell at loops: 0 (proto). You can set dwell if you want.
#             t_dep_v = env.now
#             station_times[tid].append((v, t_arr_v, t_dep_v))

# # -----------------------------
# # Run simulation
# # -----------------------------
# env = simpy.Environment()
# E = build_edges(env)
# S = build_stations(env)
# MPS = build_speed_edges()
# for tr in TRAINS:
#     env.process(train_process(env, tr, E, S,))
# env.run() # run until complete

# # -----------------------------
# # Plot: Y=time, X=station
# # -----------------------------
# xpos = {s:i for i,s in enumerate(stations)}
# plt.figure(figsize=(10,7))

# def train_data(id):
#     return {**[train for train in TRAINS if train["id"] == id][0]}

# # assign a unique color per train
# colors = plt.cm.tab20.colors  # pyright: ignore[reportAttributeAccessIssue] # 10 distinct colors
# color_map = {tr["id"]: colors[i % len(colors)] for i,tr in enumerate(TRAINS)}

# for tr in TRAINS:
#     tid = tr["id"]
#     col = color_map[tid]
#     for (u,v,t0,t1) in edge_spans[tid]:
#         xs = [xpos[u], xpos[v]]
#         ys = [t0, t1]
#         plt.plot(xs, ys, marker="o", linewidth=2, color=col,
#                  label=f"{tid} - {train_data(tid)["type"][0].upper()}" if (u,v)==edge_spans[tid][0][:2] else None)


# # Axis + styling
# plt.xticks(range(len(stations)), stations)
# plt.ylabel("Time (minutes)")
# plt.xlabel("Stations (left to right)")
# plt.title("Single-Line with Loop Crossings: Y = Time, X = Stations")
# plt.grid(True, which="both", axis="both")
# plt.legend()
# plt.gca().invert_yaxis()  # optional: make time increase downward
# plt.tight_layout()
# plt.show()
# # -----------------------------
# # Simple KPIs
# # -----------------------------
# def kpis():
#     # total delay proxy = sum( (final time - sched) )
#     delays = []
#     for tr in TRAINS:
#         tid = tr["id"]
#         sched = tr["sched"]
#         last_times = station_times[tid][-1] if station_times[tid] else None
#         if last_times:
#             _, _, tdep = last_times
#             delays.append(tdep - sched)
#     return {
#         "avg_total_time_minus_sched": round(sum(delays)/len(delays), 2),
#         "num_crossings_at_loop1": sum(1 for t in edge_spans for u,v,_,_ in edge_spans[t] if {"Loop1"} & {u,v}),
#         "num_crossings_at_loop2": sum(1 for t in edge_spans for u,v,_,_ in edge_spans[t] if {"Loop2"} & {u,v}),
#     }

# print(kpis())
