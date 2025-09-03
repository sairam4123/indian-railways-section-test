import math


def total_runtime(length_km: float,
                  line_speed_kmh: float,
                  train_max_speed_kmh: float,
                  train_accel_mps2: float,
                  train_decel_mps2: float,
                  train_headway: float,
                  should_accelerate: bool = True,
                  should_decelerate: bool = True) -> float:
    """
    Returns traversal time in minutes for a segment of length_km.
    Handles all four boolean combinations explicitly and keeps units consistent.
    """
    L = max(0.0, float(length_km)) * 1000.0   # meters
    if L == 0.0:
        return 0.0
    
    print(should_accelerate, should_decelerate)

    v_cap_kmh = max(1.0, min(float(line_speed_kmh), float(train_max_speed_kmh)))
    v_cap = v_cap_kmh * 1000.0 / 3600.0       # m/s

    a = max(train_accel_mps2, 1e-6)
    b = max(train_decel_mps2, 1e-6)

    # helper: times in seconds
    total_s = None

    # Case A: both accel AND decel allowed -> classic trapezoid/triangle
    if should_accelerate and should_decelerate:
        print("Accelerate & Decelerate")
        d_accel = v_cap**2 / (2.0 * a)
        d_decel = v_cap**2 / (2.0 * b)
        if d_accel + d_decel <= L:
            # trapezoidal: accel -> cruise -> decel
            t_accel = v_cap / a
            t_decel = v_cap / b
            d_cruise = L - (d_accel + d_decel)
            t_cruise = d_cruise / v_cap
            total_s = t_accel + t_cruise + t_decel
        else:
            # triangular: accelerate to v_peak then decelerate
            v_peak = math.sqrt((2.0 * a * b * L) / (a + b))
            t_accel = v_peak / a
            t_decel = v_peak / b
            total_s = t_accel + t_decel

    # Case B: accel only (no decel)
    elif should_accelerate and not should_decelerate:
        print("Accelerate only")
        d_accel = v_cap**2 / (2.0 * a)
        if d_accel <= L:
            # accel to v_cap then cruise remainder
            t_accel = v_cap / a
            t_cruise = (L - d_accel) / v_cap
            total_s = t_accel + t_cruise
        else:
            # triangular accel-only: can't reach v_cap
            v_peak = math.sqrt(2.0 * a * L)
            total_s = v_peak / a

    # Case C: decel only (no accel) — assume entering at v_cap then decel to 0 (or lower)
    elif (not should_accelerate) and should_decelerate:
        print("Decelerate only")
        d_decel = v_cap**2 / (2.0 * b)
        if d_decel <= L:
            # cruise at v_cap then decel at end
            t_decel = v_cap / b
            t_cruise = (L - d_decel) / v_cap
            total_s = t_cruise + t_decel
        else:
            # triangular decel-only: decelerate from v_peak to 0 across L
            v_peak = math.sqrt(2.0 * b * L)
            total_s = v_peak / b

    # Case D: neither accel nor decel -> constant speed
    else:
        print("Is this what we are doing?")
        total_s = L / v_cap

    # convert to minutes
    total_min = (total_s / 60.0) - train_headway / 2 if total_s is not None else 0.0
    return total_min


import math

# --------------------
# Helpers / converters
# --------------------
def kmh_to_mps(v_kmh: float) -> float:
    return (v_kmh * 1000.0) / 3600.0

def ensure_pos(x: float, eps: float = 1e-9) -> float:
    return max(float(x), eps)

# --------------------
# Performance clearance time
# --------------------
def performance_clear_time_min(block_m: float,
                               train_length_m: float,
                               line_speed_kmh: float,
                               accel_mps2: float,
                               should_accelerate: bool = True) -> float:
    """
    Clearance time in MINUTES for a train to clear "block_m" meters plus its own length.
    - Inputs: block_m (meters), train_length_m (meters), line_speed_kmh, accel in m/s^2.
    - Returns: minutes (float).
    """
    V = kmh_to_mps(line_speed_kmh)
    V = ensure_pos(V)
    a = ensure_pos(accel_mps2)

    total_dist_m = ensure_pos(block_m) + ensure_pos(train_length_m)

    if not should_accelerate:
        # Traverse at constant line speed
        t_s = total_dist_m / V
        return t_s / 60.0

    # Time & distance to accelerate to V
    t_accel = V / a
    d_accel = 0.5 * a * t_accel * t_accel

    if d_accel >= total_dist_m:
        # Triangular: never reaches V
        # s = 0.5 * a * t^2 => t = sqrt(2*s/a)
        t_s = math.sqrt(2.0 * total_dist_m / a)
    else:
        cruise_dist = total_dist_m - d_accel
        t_cruise = cruise_dist / V
        t_s = t_accel + t_cruise

    return t_s / 60.0

# --------------------
# Total headway (clean)
# --------------------
def total_headway(
        train_speed_kmh: float,
        line_speed_kmh: float,
        train_length_m: float,
        accel_mps2: float,
        decel_mps2: float,
        aspects: int = 2,
        buffer_min: float = 1.0,
        block_km: float = 1.5,
        should_accelerate: bool = True,
        should_decelerate: bool = True,
        max_eff_block_factor: float | None = None
        ) -> float:
    """
    Fixed headway calculation (returns minutes).
    - If should_decelerate is False, braking distance will NOT increase eff_block.
    - Optional max_eff_block_factor: if set, caps eff_block to block_m * factor to avoid runaway values.
    """
    # converters / small helpers
    def kmh_to_mps(v_kmh: float) -> float:
        return (v_kmh * 1000.0) / 3600.0
    def ensure_pos(x: float, eps: float = 1e-9) -> float:
        return max(float(x), eps)

    # base block in meters
    block_m = max(0.0, float(block_km)) * 1000.0
    denom = max(1, int(aspects) - 1)

    # compute braking distance only if decel matters
    if should_decelerate:
        v_mps = kmh_to_mps(train_speed_kmh)
        b = ensure_pos(decel_mps2)
        d_brake_m = (v_mps * v_mps) / (2.0 * b)
        eff_block_m = max(block_m, d_brake_m / denom)
    else:
        eff_block_m = block_m

    # optional cap to prevent astronomically large effective blocks
    if max_eff_block_factor is not None:
        eff_block_m = min(eff_block_m, block_m * float(max_eff_block_factor))

    # performance-limited time (minutes)
    perf_min = performance_clear_time_min(eff_block_m, train_length_m, line_speed_kmh, accel_mps2, should_accelerate)

    # signalling-limited time in minutes (traverse eff_block at line speed)
    v_line_mps = ensure_pos(kmh_to_mps(line_speed_kmh))
    sig_time_min = (eff_block_m / v_line_mps) / 60.0

    headway_min = max(perf_min, sig_time_min) + max(0.0, float(buffer_min))
    return float(headway_min) / 60.0
