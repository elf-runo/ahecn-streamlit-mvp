# kpis.py
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional

@dataclass
class CaseTimes:
    requested: Optional[int] = None
    accepted: Optional[int] = None
    dispatched: Optional[int] = None
    arrived: Optional[int] = None

def _median(nums: List[float]) -> Optional[float]:
    if not nums:
        return None
    nums = sorted(nums)
    n = len(nums)
    mid = n // 2
    return float(nums[mid]) if n % 2 else (nums[mid - 1] + nums[mid]) / 2.0

def compute_kpis(events: List[dict]) -> Dict[str, Optional[float]]:
    case_times = defaultdict(CaseTimes)
    mismatch_flags: List[int] = []
    reroute_flags: List[int] = []
    last_status: Dict[str, str] = {}

    for ev in events:
        cid = ev.get("case_id")
        if not cid:
            continue

        if ev.get("event_type") == "REFERRAL_STATUS_CHANGE":
            ts = int(ev["ts_epoch"])
            s_to = (ev.get("status_to") or "").upper()
            s_from = (ev.get("status_from") or "").upper()

            if s_to == "REQUESTED":
                case_times[cid].requested = case_times[cid].requested or ts
            elif s_to == "ACCEPTED":
                case_times[cid].accepted = case_times[cid].accepted or ts
            elif s_to == "DISPATCHED":
                case_times[cid].dispatched = case_times[cid].dispatched or ts
            elif s_to == "ARRIVED":
                case_times[cid].arrived = case_times[cid].arrived or ts

            # reroute proxy: odd transition inconsistencies
            if cid in last_status and s_from and s_from != last_status[cid]:
                reroute_flags.append(1)
            last_status[cid] = s_to

            # mismatch proxy (required vs facility caps)
            payload = ev.get("payload") or {}
            req = payload.get("required_capabilities") or {}
            fac = payload.get("facility_capabilities") or {}
            if req and fac:
                ok = True
                for k, v in req.items():
                    if v and not bool(fac.get(k, False)):
                        ok = False
                        break
                mismatch_flags.append(0 if ok else 1)

    tta, d2a, golden = [], [], []
    for _, t in case_times.items():
        if t.requested and t.accepted and t.accepted >= t.requested:
            tta.append(t.accepted - t.requested)
        if t.dispatched and t.arrived and t.arrived >= t.dispatched:
            d2a.append(t.arrived - t.dispatched)
        if t.requested and t.arrived and t.arrived >= t.requested:
            golden.append(1 if (t.arrived - t.requested) <= 3600 else 0)

    mismatch_rate = (sum(mismatch_flags) / len(mismatch_flags)) if mismatch_flags else None
    reroute_rate = (sum(reroute_flags) / max(1, len(case_times))) if case_times else None
    golden_rate = (sum(golden) / len(golden)) if golden else None

    return {
        "cases_count": float(len(case_times)),
        "median_time_to_accept_seconds": _median(tta),
        "median_dispatch_to_arrival_seconds": _median(d2a),
        "mismatch_rate": mismatch_rate,
        "reroute_rate": reroute_rate,
        "golden_hour_proxy_rate": golden_rate,
    }
