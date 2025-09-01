import math
import random
import simpy
import simpy.resources.resource

from networks.kkdi_tpj_network import create_tpj_kkdi_network
from train_lib.models import Train, Station, Track, BlockSection
from train_lib.constants import ARRIVAL, DEPARTURE

env = simpy.Environment()

[[TPJ, KRMG, KRUR, VEL, PDKT, TYM, CTND, KKDI], block_sections, *loop_tracks] = create_tpj_kkdi_network(env)

# Existing trains
# train1 = Train(env, "T1", [], max_speed=80, priority=1, length=300, weight=1173, initial_delay=0, hp=6120, accel_mps2=0.5, decel_mps2=0.5)
# train1.schedule_stop(TPJ, 0, 10, 1)
# train1.schedule_stop(KRMG, 20, 25, 0)
# train1.schedule_stop(KRUR, 40, 45, 1)
# train1.schedule_stop(VEL, 50, 50, 0)
# train1.schedule_stop(PDKT, 70, 75, 0)
# train1.schedule_stop(TYM, 85, 90, 0)
# train1.schedule_stop(CTND, 100, 105, 1)
# train1.schedule_stop(KKDI, 110, 115, 0)

# 0 = 5:30
# 10 = 5:40
# 33 = 6:03
# 35 = 6:05
# 80 = 06:50
# 85 = 06:55

train2 = Train(env, "T2", [], max_speed=110, priority=2, length=564, weight=1000, initial_delay=0, hp=6120, accel_mps2=1.2, decel_mps2=0.5)
train2.schedule_stop(KKDI, 0, 10, 0)
train2.schedule_stop(CTND, 15, 15, 0)
train2.schedule_stop(TYM, 25, 25, 0)
train2.schedule_stop(PDKT, 33, 35, 0)
train2.schedule_stop(VEL, 45, 45, 0)
train2.schedule_stop(KRUR, 60, 60, 0)
train2.schedule_stop(KRMG, 75, 75, 0)
train2.schedule_stop(TPJ, 80, 85, 0)
train2.set_direction("DOWN")

# train3 = Train(env, "T3", [], max_speed=60, priority=1, length=280, weight=1170, initial_delay=0, hp=6120, accel_mps2=0.5, decel_mps2=0.5)
# train3.schedule_stop(TPJ, 20, 25, 1)
# train3.schedule_stop(KRMG, 30, 30, 1)
# train3.schedule_stop(KRUR, 40, 40, 0) # Run through main
# train3.schedule_stop(VEL, 60, 60, 0)
# train3.schedule_stop(PDKT, 85, 90, 0)
# train3.schedule_stop(TYM, 95, 100, 0)
# train3.schedule_stop(CTND, 100, 100, 0) # Run through main
# train3.schedule_stop(KKDI, 125, 130, 1)

# train4 = Train(env, "T4", [], max_speed=120, priority=2, length=320, weight=2580, initial_delay=0, hp=4500, accel_mps2=0.5, decel_mps2=0.5)
# train4.schedule_stop(KKDI, 40, 45, 1)
# train4.schedule_stop(CTND, 60, 65, 1)
# train4.schedule_stop(TYM, 80, 85, 1)
# train4.schedule_stop(PDKT, 95, 100, 1)
# train4.schedule_stop(VEL, 110, 115, 1)
# train4.schedule_stop(KRUR, 130, 135, 1)
# train4.schedule_stop(KRMG, 140, 140, 0)
# train4.schedule_stop(TPJ, 150, 155, 0)

# train5 = Train(env, "T5", [], max_speed=90, priority=3, length=250, weight=1277, initial_delay=0, hp=4500, accel_mps2=0.3, decel_mps2=0.2)
# train5.schedule_stop(TPJ, 60, 65, 2)
# train5.schedule_stop(KRMG, 80, 80, 0)
# train5.schedule_stop(KRUR, 90, 90, 0)
# train5.schedule_stop(VEL, 110, 110, 0)
# train5.schedule_stop(PDKT, 150, 150, 0)
# train5.schedule_stop(TYM, 210, 210, 0)
# train5.schedule_stop(CTND, 220, 220, 0)
# train5.schedule_stop(KKDI, 270, 280, 1)

# train6 = Train(env, "T6", [], max_speed=160, priority=1, length=300, weight=4000, initial_delay=0, hp=12000, accel_mps2=0.5, decel_mps2=0.5)
# train6.schedule_stop(KKDI, 75, 80, 0)
# train6.schedule_stop(CTND, 100, 100, 0) # Run through main
# train6.schedule_stop(TYM, 120, 120, 0)
# train6.schedule_stop(PDKT, 130, 140, 0)
# train6.schedule_stop(VEL, 150, 150, 0)
# train6.schedule_stop(KRUR, 160, 160, 0)
# train6.schedule_stop(KRMG, 170, 170, 0)
# train6.schedule_stop(TPJ, 185, 190, 1)

# train7 = Train(env, "T7", [], max_speed=100, priority=2, length=270, weight=3600, initial_delay=0, hp=3125, accel_mps2=0.5, decel_mps2=0.5)
# train7.schedule_stop(TPJ, 95, 100, 1)
# train7.schedule_stop(KRMG, 110, 115, 0)
# train7.schedule_stop(KRUR, 120, 125, 1)
# train7.schedule_stop(VEL, 130, 135, 0)
# train7.schedule_stop(PDKT, 150, 155, 0)
# train7.schedule_stop(TYM, 170, 175, 0)
# train7.schedule_stop(CTND, 180, 185, 1)
# train7.schedule_stop(KKDI, 200, 205, 1)

# train8 = Train(env, "T8", [], max_speed=160, priority=0, length=384, weight=430, initial_delay=0, hp=9010, accel_mps2=0.7, decel_mps2=0.7) # Non stop vande bharat express
# train8.schedule_stop(KKDI, 10, 10, 0)
# train8.schedule_stop(CTND, 40, 40, 0)
# train8.schedule_stop(TYM, 60, 60, 0)
# train8.schedule_stop(PDKT, 70, 70, 0)
# train8.schedule_stop(VEL, 90, 90, 0)
# train8.schedule_stop(KRUR, 105, 105, 0)
# train8.schedule_stop(KRMG, 115, 115, 0)
# train8.schedule_stop(TPJ, 220, 220, 0)

# # # === EXTRA TRAINS (T9–T16) ===

# train9 = Train(env, "T9", [], max_speed=80, priority=2, length=300, weight=4100, initial_delay=0, hp=6120, accel_mps2=0.5, decel_mps2=0.5)
# train9.schedule_stop(TPJ, 15, 20, 1)
# train9.schedule_stop(KRMG, 30, 35, 1)
# train9.schedule_stop(KRUR, 50, 55, 1)
# train9.schedule_stop(VEL, 70, 75, 1)
# train9.schedule_stop(PDKT, 85, 90, 0)
# train9.schedule_stop(TYM, 100, 105, 1)
# train9.schedule_stop(CTND, 115, 120, 1)
# train9.schedule_stop(KKDI, 130, 135, 0)

# train10 = Train(env, "T10", [], max_speed=110, priority=1, length=310, weight=4050, initial_delay=0, hp=6120, accel_mps2=0.5, decel_mps2=0.5)
# train10.schedule_stop(KKDI, 20, 25, 1)
# train10.schedule_stop(CTND, 45, 50, 0)
# train10.schedule_stop(TYM, 60, 65, 0)
# train10.schedule_stop(PDKT, 70, 75, 0)
# train10.schedule_stop(VEL, 80, 85, 1)
# train10.schedule_stop(KRUR, 95, 100, 1)
# train10.schedule_stop(KRMG, 110, 115, 1)
# train10.schedule_stop(TPJ, 125, 130, 1)

# train11 = Train(env, "T11", [], max_speed=70, priority=3, length=280, weight=3700, initial_delay=0, hp=3125, accel_mps2=0.5, decel_mps2=0.5)
# train11.schedule_stop(TPJ, 30, 35, 1)
# train11.schedule_stop(KRUR, 60, 60, 0)
# train11.schedule_stop(PDKT, 95, 100, 0)
# train11.schedule_stop(CTND, 120, 120, 0)
# train11.schedule_stop(KKDI, 140, 145, 1)

# train12 = Train(env, "T12", [], max_speed=120, priority=1, length=330, weight=4200, initial_delay=0, hp=6000, accel_mps2=0.5, decel_mps2=0.5)
# train12.schedule_stop(KKDI, 50, 55, 1)
# train12.schedule_stop(CTND, 75, 80, 1)
# train12.schedule_stop(PDKT, 105, 110, 0)
# train12.schedule_stop(KRUR, 140, 145, 1)
# train12.schedule_stop(TPJ, 165, 170, 0)

# train13 = Train(env, "T13", [], max_speed=100, priority=2, length=260, weight=3500, initial_delay=0, hp=3125, accel_mps2=0.5, decel_mps2=0.5)
# train13.schedule_stop(TPJ, 70, 75, 0)
# train13.schedule_stop(KRUR, 100, 100, 0)
# train13.schedule_stop(PDKT, 130, 130, 0)
# train13.schedule_stop(CTND, 155, 160, 1)
# train13.schedule_stop(KKDI, 180, 185, 1)

# train14 = Train(env, "T14", [], max_speed=140, priority=1, length=300, weight=3950, initial_delay=0, hp=4500, accel_mps2=0.5, decel_mps2=0.5)
# train14.schedule_stop(KKDI, 85, 90, 0)
# train14.schedule_stop(CTND, 110, 110, 0)
# train14.schedule_stop(PDKT, 140, 140, 0)
# train14.schedule_stop(KRUR, 165, 165, 0)
# train14.schedule_stop(TPJ, 190, 195, 1)

# train15 = Train(env, "T15", [], max_speed=90, priority=3, length=250, weight=4900, initial_delay=0, hp=4500, accel_mps2=0.5, decel_mps2=0.5)
# train15.schedule_stop(TPJ, 105, 110, 1)
# train15.schedule_stop(KRUR, 135, 135, 0)
# train15.schedule_stop(PDKT, 160, 160, 0)
# train15.schedule_stop(CTND, 185, 185, 0)
# train15.schedule_stop(KKDI, 210, 215, 1)

# train16 = Train(env, "T16", [], max_speed=160, priority=0, length=384, weight=430, initial_delay=0, hp=9010, accel_mps2=0.15, decel_mps2=0.18)  # Another VB express
# train16.schedule_stop(KKDI, 40, 40, 0)
# train16.schedule_stop(CTND, 65, 65, 0)
# train16.schedule_stop(PDKT, 80, 80, 0)
# train16.schedule_stop(KRUR, 115, 115, 0)
# train16.schedule_stop(TPJ, 145, 145, 0)


env.run()

trains = Train.TRAINS
train_logs = sum([train.log.entries for train in trains], [])
train_marks = sum([train.log.marks for train in trains], [])

train_logs.sort(key=lambda x: (x[0], x[1]))
train_marks.sort(key=lambda x: (x[0], x[1]))
print("Time\tTrain\tEvent\tStation")
# for train_log in train_logs:
#     print(f"{train_log[0]}\t{train_log[1]}\t{train_log[2]}")

for train_mark in train_marks:
    print(f"{train_mark[0]:.3f}\t{train_mark[1]}\t{train_mark[2]}\t{train_mark[3]}")

import matplotlib.pyplot as plt
import pandas as pd


def plot_timetable(trains: list[Train], stations: list[Station]):
    # Build structured log dataframe
    # logs = []
    # for train in trains:
    #     for t, tid, ev, stn in train.log.marks:
    #         logs.append(dict(time=t, train=tid, event=ev, station=stn))
    # df = pd.DataFrame(logs).sort_values(["time", "train"])

    # Print logs nicely (instead of subplot)
    # print("\n===== Event Logs =====")
    # print(df.to_string(index=False))

    # ---------- create 2x2 layout ----------
    fig, axs = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    ax1, ax2, ax3, ax4 = axs[0,0], axs[0,1], axs[1,0], axs[1,1]

    # ---------- (1) CLEAN GANTT CHART ----------
    ax1.set_title("Train Schedule Gantt Chart (with Station/Block labels)")

    for i, train in enumerate(trains):
        color = plt.cm.tab20.colors[i % 20]
        marks = train.log.marks
        for j in range(len(marks)-1):
            t0, tid, type0, stn0 = marks[j]
            t1, _, type1, stn1 = marks[j+1]

            # Station dwell
            if type0 == "ARRIVE" and type1 == "DEPART":
                bar = ax1.barh(i, t1 - t0, left=t0, color=color, alpha=0.6, label=tid if j == 0 else "")
                label = stn0
                mid = (t0 + t1) / 2

            # Block run
            elif type0 == "ENTRY" and type1 == "EXIT":
                bar = ax1.barh(i, t1 - t0, left=t0, color=color, alpha=0.9, hatch="//", label="")
                label = stn0  # block name
                mid = (t0 + t1) / 2
            else:
                continue

            # --- Only add text if it fits inside bar ---
            bar_width = t1 - t0
            renderer = ax1.figure.canvas.get_renderer()
            txt = ax1.text(mid, i, label, ha="center", va="center", fontsize=8, color="white" if type0=="ENTRY" else "black")
            bb = txt.get_window_extent(renderer=renderer)
            txt_width = bb.width / ax1.figure.dpi * 72  # approx in data coords

            if txt_width > bar_width * 1.2:  # too wide -> remove
                txt.remove()
    ax1.set_yticks(range(len(trains)))
    ax1.set_yticklabels([t.id for t in trains])
    ax1.set_xlabel("Time (min)")
    ax1.legend()
    ax1.grid(True, axis="x")

    # ---------- (2) DELAY HEATMAP ----------
    ax2.set_title("Delay Heatmap (min)")
    station_list = stations
    matrix = []

    for train in trains:
        scheduled = [sp.arrival_time for sp in train.schedule]
        actual = [t for t, tid, ev, stn in train.log.marks if tid==train.id and ev=="ARRIVE"]
        if len(scheduled) == len(actual):
            delay = list(reversed([a-s for a,s in zip(actual, scheduled)])) if train.direction=="DOWN" else [a-s for a,s in zip(actual, scheduled)]
        else:
            delay = [None]*len(scheduled)  # incomplete runs
        matrix.append(delay)

    import numpy as np
    matrix = np.array(matrix, dtype=float)  # force float for NaN handling

    print(matrix)


    im = ax2.imshow(matrix, cmap="coolwarm", aspect="auto", interpolation="nearest")
    ax2.set_xticks(range(len(station_list)))
    ax2.set_xticklabels(station_list)
    ax2.set_yticks(range(len(trains)))
    ax2.set_yticklabels([t.id for t in trains])

    for i in range(matrix.shape[0]):   # rows = trains
        for j in range(matrix.shape[1]):  # cols = stations
            val = matrix[i, j]
            if not np.isnan(val):  # skip missing
                ax2.text(j, i, f"{int(val)}", ha="center", va="center",
                        color="black" if abs(val) < 5 else "white", fontsize=8)

    plt.colorbar(im, ax=ax2, label="Delay (min)")


    # ---------- (3) TRAIN GRAPH ----------
    ax3.set_title("Train Graph (Time vs Stations/Blocks)")

    # map stations+blocks to Y positions
    ypos = {st: i for i, st in enumerate(stations)}  # add blocks if needed
    for i, train in enumerate(trains):
        col = plt.cm.tab20.colors[i % 20]
        marks = list(filter(lambda x: x[2] in [ARRIVAL, DEPARTURE], train.log.marks))
        for j in range(len(marks) - 1):
            t0, tid, type0, stn0 = marks[j]
            t1, _, type1, stn1 = marks[j+1]

            if type0 == ARRIVAL and type1 == DEPARTURE:
                # dwell (horizontal in your old plot, vertical now!)
                ax3.plot([t0, t1], [ypos[stn0], ypos[stn0]], 
                        linestyle="dashed", color=col,
                        label=f"{tid} - {train.priority}" if j == 0 else "")
            elif type0 == DEPARTURE and type1 == ARRIVAL:
                # run (diagonal from one station to next)
                ax3.plot([t0, t1], [ypos[stn0], ypos[stn1]], 
                        linestyle="solid", marker="o", color=col)

    ax3.set_yticks(range(len(stations)))
    ax3.set_yticklabels(stations)
    ax3.set_xlabel("Time (min)")
    ax3.set_ylabel("Stations / Blocks")
    ax3.grid(True)
    ax3.legend()


    ax4.set_title("Schedule vs Actual Timetable (Overlay)")

    for i, train in enumerate(trains):
        col = plt.cm.tab20.colors[i % 20]

        # Plot scheduled timetable
        for sp in train.schedule:
            arr, dep = sp.arrival_time, sp.departure_time
            ax4.barh(i, dep-arr, left=arr, edgecolor=col, color=col, fill=True, hatch='//', linewidth=1.5, alpha=0.9)

        # Plot actual timetable from logs
        marks = train.log.marks
        for j in range(len(marks)-1):
            t0, tid, type0, stn0 = marks[j]
            t1, _, type1, stn1 = marks[j+1]
            if type0=="ARRIVE" and type1=="DEPART":
                ax4.barh(i, t1-t0, left=t0, color=col, alpha=0.6)

    ax4.set_yticks(range(len(trains)))
    ax4.set_yticklabels([t.id for t in trains])
    ax4.set_xlabel("Time (min)")
    ax4.grid(True, axis="x")

    # fig.tight_layout()
    plt.show()

plot_timetable(Train.TRAINS, [station.stn_code for station in Station.STATIONS])

# import matplotlib.pyplot as plt

# # ---------- (1) GANTT CHART ----------
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))  # side by side
# # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))  # top & bottom (use this instead if you prefer)

# ax1.set_title("Train Schedule Gantt Chart")

# def split_text(text, max_chars=20):
#     words = text.split()
#     lines, current_line = [], ""
#     for word in words:
#         if len(current_line) + len(word) + 1 <= max_chars:
#             current_line += (" " + word if current_line else word)
#         else:
#             lines.append(current_line)
#             current_line = word
#     if current_line: lines.append(current_line)
#     return "\n".join(lines)

# for i, train in enumerate(trains):
#     color = plt.cm.tab20.colors[i % 20]
#     entries = train.log.entries
#     for idx in range(len(entries) - 1):
#         start = entries[idx][0]
#         end = entries[idx + 1][0]
#         message = entries[idx][2]

#         ax1.barh(i, end - start, left=start, color=color, edgecolor='black', alpha=0.7)
#         ax1.text(start + (end - start) / 2, i, split_text(message, (end-start)/2),
#                  va='center', ha='center', fontsize=8, color='black')

# ax1.set_xlabel("Time")
# ax1.set_ylabel("Train")
# ax1.set_yticks(range(len(trains)))
# ax1.set_yticklabels([train.id for train in trains])
# handles = [plt.Rectangle((0, 0), 1, 1, color=plt.cm.tab20.colors[i % 20]) for i in range(len(trains))]
# ax1.legend(handles, [f"{train.id} - {train.priority}" for train in trains], title="Trains")
# ax1.grid(True, which="both", axis="x")

# # ---------- (2) TRAIN GRAPH ----------
# stations = ["tpj", "pdkt", "kkdi"]
# xpos = {st: i for i, st in enumerate(stations)}

# edge_spans = {}
# for train in trains:
#     spans = []
#     marks = list(filter(lambda x: x[2] in [ARRIVAL, DEPARTURE], train.log.marks))
#     for i in range(len(marks) - 1):
#         t0, tid, mtype0, stn0 = marks[i]
#         t1, _, mtype1, stn1 = marks[i+1]

#         # --- dwell detection ---
#         # if "Entering" in msg0 and "Accepted" in msg1:
#         if mtype0 == ARRIVAL and mtype1 == DEPARTURE:
#             # Train is dwelling inside platform (between accepted & dispatch)
#             spans.append((stn0, stn0, t0, t1, "dwell"))

#         # # --- run detection ---
#         # elif mtype0 == DEPARTURE and mtype1 == ENTRY:
#         #     spans.append((stn0, stn1, t0, t1, "run"))
        
#         elif mtype0 == DEPARTURE and mtype1 == ARRIVAL:
#             spans.append((stn0, stn1, t0, t1, "run"))

#         # --- terminate detection ---
#         # elif "Accepted" in msg0 and "Terminating" in msg1:
#         #     u = msg0.split()[1]           # current station
#         #     spans.append((u, u, t0, t1, "dwell"))

#     edge_spans[train.id] = spans

# colors = plt.cm.tab20.colors
# color_map = {train.id: colors[i % len(colors)] for i, train in enumerate(trains)}

# for train in trains:
#     tid = train.id
#     col = color_map[tid]
#     for j, (u, v, t0, t1, kind) in enumerate(edge_spans[tid]):
#         xs = [xpos[u], xpos[v]]
#         ys = [t0, t1]
#         style = "solid" if kind == "run" else "dashed"
#         ax2.plot(xs, ys, marker="o", linewidth=2, color=col, linestyle=style,
#                  label=tid if j == 0 else None)
#         # annotate arrival & departure
#         ax2.text(xs[0], ys[0], f"{ys[0]}", fontsize=7, va="bottom", ha="right")
#         ax2.text(xs[1], ys[1], f"{ys[1]}", fontsize=7, va="bottom", ha="left")

# ax2.set_xticks(range(len(stations)))
# ax2.set_xticklabels(stations)
# ax2.set_ylabel("Time (minutes)")
# ax2.set_xlabel("Stations (left to right)")
# ax2.set_title("Train Graph with Runs + Dwells")
# ax2.invert_yaxis()
# ax2.grid(True, which="both", axis="both")
# ax2.legend()

# plt.tight_layout()
# plt.show()
