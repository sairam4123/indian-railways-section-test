from ortools.sat.python import cp_model

# --- Setup ---
model = cp_model.CpModel()

# Desired entry times for trains (in minutes after 8:00)
desired = {"A": 0, "B": 3, "C": 6}
duration = 1  # each train takes 5 minutes in the block

# --- Variables ---
start = {t: model.new_int_var(0, 20, f"start_{t}") for t in desired}

# --- Constraints ---
# 1. No two trains overlap
trains = list(desired.keys())
for i in range(len(trains)):
    for j in range(i + 1, len(trains)):
        t1, t2 = trains[i], trains[j]
        b1 = model.new_bool_var(f"{t1}_before_{t2}")
        b2 = model.new_bool_var(f"{t2}_before_{t1}")
        # Either t1 finishes before t2 starts OR vice versa
        model.add(start[t1] + duration <= start[t2]).only_enforce_if(b1)
        model.add(start[t2] + duration <= start[t1]).only_enforce_if(b2)
        model.add_bool_or([b1, b2])

# --- Objective: minimize total delay ---
delays = []
for t in trains:
    delay = model.new_int_var(0, 20, f"delay_{t}")
    model.add(delay >= start[t] - desired[t])  # delay = max(0, actual - desired)
    model.add(delay >= 0)
    model.add(start[t] >= desired[t])  # no early arrivals
    delays.append(delay)

model.minimize(sum(delays))

# --- Solve ---
solver = cp_model.CpSolver()
solver.solve(model)

# --- Results ---
for t in trains:
    print(f"Train {t}: desired {desired[t]}, scheduled {solver.value(start[t])}, delay {solver.value(start[t]) - desired[t]}")
print("Total delay:", sum(solver.value(d) for d in delays))
