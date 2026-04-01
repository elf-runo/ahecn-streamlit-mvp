# routing_engine.py
from __future__ import annotations
from typing import Tuple
import math

def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = a; lat2, lon2 = b
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    h = math.sin(dlat/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2
    return 2 * r * math.asin(math.sqrt(h))

def get_eta(origin: Tuple[float, float], dest: Tuple[float, float], speed_kmh: float = 40.0, is_hilly_terrain: bool = True) -> float:
    base_km = haversine_km(origin, dest)
    
    # Topographical Multiplier: Winding mountain roads increase actual transit distance
    actual_km = base_km * 1.8 if is_hilly_terrain else base_km * 1.2
    
    # Degrade speed based on distance (longer mountain routes have worse average speeds)
    effective_speed = speed_kmh * 0.7 if (is_hilly_terrain and actual_km > 20) else speed_kmh
    
    return max(1.0, (actual_km / effective_speed) * 60.0)
