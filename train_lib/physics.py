import math


def braking_distance(speed_kmph: float, decel_mps2: float) -> float:
    """Braking distance in km from deceleration."""
    V = speed_kmph * 1000 / 3600  # m/s
    d = V**2 / (2 * decel_mps2) / 1000  # km
    return d

def performance_clear_time(block_km: float,
                           train_length_m: float,
                           line_speed: float,
                           accel_mps2: float) -> float:
    """Clearance time in minutes based on accel + cruising."""
    V = line_speed * 1000 / 3600   # m/s
    total_dist = (block_km + train_length_m/1000) * 1000  # m

    # Time + distance to accelerate to line speed
    t_accel = V / accel_mps2
    d_accel = 0.5 * accel_mps2 * t_accel**2

    if d_accel >= total_dist:
        # Train never reaches line speed
        t_clear = math.sqrt(2 * total_dist / accel_mps2)
    else:
        cruise_dist = total_dist - d_accel
        t_clear = t_accel + cruise_dist / V

    return t_clear / 60  # minutes

def total_headway(
        train_speed: float,
        line_speed: float,
                  train_length_m: float,
                  accel_mps2: float,
                  decel_mps2: float,
                  aspects: int = 2,
                  buffer_min: float = 1.0,
                  block_km: float = 1.5) -> float:
    """
    Combined headway considering performance + braking + signalling.
    """
    # Braking distance at train speed
    d_brake = braking_distance(train_speed, decel_mps2)

    # Effective block length (must cover braking distance / aspects)
    eff_block = max(block_km, d_brake / max(1, aspects - 1))

    # Performance-limited time
    perf_time = performance_clear_time(eff_block,
                                       train_length_m,
                                       line_speed,
                                       accel_mps2)

    # Signalling-limited time (distance/speed in minutes)
    sig_time = (eff_block / line_speed) * 60

    # Headway = whichever dominates + buffer
    return max(perf_time, sig_time) + buffer_min


def total_runtime(length_km, line_speed, train_max_speed, train_accel, train_decel, should_accelerate = True, should_decelerate = True) -> int:
    L = max(0.0, float(length_km)) * 1000.0                  # meters
    if L == 0:
        return 0

    v_cap_kmh = max(1.0, min(float(line_speed), float(train_max_speed)))
    v_cap = v_cap_kmh * (1000.0 / 3600.0)                         # m/s

    # Use train-provided accel/decel if available; otherwise reasonable defaults
    a = max(train_accel, 0.1)  # m/s^2
    b = max(train_decel, 0.1)  # m/s^2

    # --- Distances to accel to v_cap and decel from v_cap ---
    d_accel = 0.5 * v_cap * v_cap / a
    d_decel = 0.5 * v_cap * v_cap / b
    
    # if d_accel + d_decel <= L:
    #     # Trapezoidal: accel → cruise → decel
    #     d_cruise = L - (d_accel + d_decel)
    #     t_accel  = v_cap / a
    #     t_cruise = d_cruise / v_cap if v_cap > 0 else 0.0
    #     t_decel  = v_cap / b
    #     total_s  = t_accel + t_cruise + t_decel
    
    # else:
    #     # Triangular: cannot reach v_cap; peak speed determined by length
    #     # Solve L = v^2/(2a) + v^2/(2b) → v_peak = sqrt(2abL/(a+b))
    #     v_peak = math.sqrt((2.0 * a * b * L) / (a + b))
    #     t_accel = v_peak / a
    #     t_decel = v_peak / b
    #     total_s = t_accel + t_decel
    
    total_s = length_km / v_cap

    # Convert to minutes, ensure at least 1 minute for any positive length
    total_min = max(1, math.ceil(total_s / 60.0))

    return total_min