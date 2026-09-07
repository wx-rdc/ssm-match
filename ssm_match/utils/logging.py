"""极简日志：stdout 关键行 + JSONL 指标文件（训练任务日志平台可见，JSONL 供离线画图）。"""

import json
import os
import time


class Logger:
    def __init__(self, log_dir: str, name: str = "train"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"{name}.jsonl")
        self.t0 = time.time()

    def log(self, record: dict, also_print: bool = True) -> None:
        record = dict(record, wall_time=round(time.time() - self.t0, 1))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if also_print:
            msg = " | ".join(f"{k}={v}" for k, v in record.items() if k != "wall_time")
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
