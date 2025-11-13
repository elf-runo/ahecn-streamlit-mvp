import random, time, threading
from storage import publish_event

def start_demo_feeder():
    stop_ev = threading.Event()
    def _run():
        while not stop_ev.is_set():
            # emit a random synthetic event every 2–5s
            publish_event({
                "type": random.choice(["comment.added","status.updated","route.update"]),
                "case_id": "DEMO",
                "actor": "feeder",
                "payload": {"note": "synthetic"}
            })
            stop_ev.wait(random.uniform(2,5))
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return stop_ev
