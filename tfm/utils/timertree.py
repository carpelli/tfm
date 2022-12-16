from contextlib import contextmanager
from datetime import datetime
from timeit import default_timer
from pathlib import Path
import json
import logging

data = {}
trail = [data]
path = []

@contextmanager
def timer(name):
    head = trail[-1]
    if name in head:
        # make this better
        logging.warn(f"Timer {name} already exists!")
    head[name] = {}
    trail.append(head[name])
    path.append(name)
    start = default_timer()
    try:
        yield
    finally:
        time = default_timer() - start
        head = trail.pop()
        if head:
            head['_total'] = time
        else:
            trail[-1][name] = time
        logging.info(f"timed {'/'.join(path)}: {time}s")
        path.pop()


def save_timer(dir: Path):
    if not dir.exists():
        dir.mkdir(parents=True)
    filepath = dir / (datetime.today().isoformat() + '.json')
    filepath.write_text(json.dumps(data, indent=4))
