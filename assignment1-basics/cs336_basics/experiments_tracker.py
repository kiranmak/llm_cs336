import os
import json
import time
from dataclasses import asdict, is_dataclass
from typing import Any
from torch.utils.tensorboard import SummaryWriter
from cs336_basics.paths import EXP_PATH

class ExperimentTracker:
    """
    A lightweight experiment tracker that logs scalar metrics locally against:
    - global step (gradient update count)
    - wall-clock time (seconds since run start)

    Logs are written to:
    - logs/<run_name>/metrics.jsonl
    - logs/<run_name>/config.json
    - TensorBoard event files in logs/<run_name>/
    """
    def __init__(self, service_name:str, mode:str) -> None:
        self.log_dir = EXP_PATH
        self.mode = "decode" if mode is None else mode
        log_step_dir = self.get_next_log_filename(service_name, self.mode)

        self.metrics_path = os.path.join(log_step_dir, "metrics.jsonl")
        self.t0 = time.time()
        # Initialize TensorBoard SummaryWriter directly inside the run folder
        self.tb_writer = SummaryWriter(log_dir=self.log_dir)

    def wall_time_s(self) -> float:
        return time.time() - self.t0

    def get_next_log_filename(self, service:str, mode:str) -> str:
        """
        Scan logs/ finds the highest existing index for the
        date, and returns the next log filename.
        """
        import os
        import re
        from datetime import datetime
        # 1. Sanitize inputs exactly like before
        date_str = datetime.now().strftime("%Y%m%d")

        # 3. regex pattern to match existing files and capture the index
        # Matches: tokenfile_YYYYMMDD_(digits).log
        pattern = re.compile(
            rf"^{re.escape(service)}_{re.escape(mode)}_{re.escape(date_str)}_(\d+)\$")

        # 4. Scan directory and find the maximum index
        max_index = 0
        for filename in os.listdir(self.log_dir):
            match = pattern.match(filename)
            if match:
                # Extract the index digits and convert to an integer
                file_index = int(match.group(1))
                if file_index > max_index:
                    max_index = file_index

        # 5. Increment to get the next index
        next_index = max_index + 1

        # 6. Return the fully formatted file path or filename
        directory = os.path.join(self.log_dir,
                              f"{service}_{mode}_{date_str}_{next_index:02d}")

        # 2. Create the directory if it doesn't exist yet
        os.makedirs(directory, exist_ok=True)
        relative = os.path.relpath(directory)
        print(f"Created directory: {relative}")
        return directory


    def log(self, step: int, metrics: dict[str, float | int]) -> None:
        """
        Log a dictionary of scalar metrics at a given step.
        A wall_time_s field will be automatically added.
        """
        record = {"step": int(step), "wall_time_s": float(self.wall_time_s())}
        for k, v in metrics.items():
            record[k] = float(v) if isinstance(v, (int, float)) else v

        # 1. Append to local JSONL log file
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # 2. Log numerical items to TensorBoard
        for tag, val in metrics.items():
            if isinstance(val, (int, float)):
                self.tb_writer.add_scalar(tag, val, global_step=int(step))

        # Log wall time explicitly as a metric on the X-axis
        self.tb_writer.add_scalar("meta/wall_time_s",
                                  record["wall_time_s"], global_step=int(step))

    def close(self) -> None:
        """ Flush pending entries and close TensorBoard writer. """
        if hasattr(self, "tb_writer") and self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()

    @staticmethod
    def _to_jsonable(obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, dict):
            return {k: ExperimentTracker._to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [ExperimentTracker._to_jsonable(x) for x in obj]
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        return str(obj)

# Sample Usage example
"""
tracker = ExperimentTracker(log_dir="./logs/exp_01", config={"lr": 0.001, "batch_size": 32})

# Training Loop Simulation
for step in range(100):
    loss = 0.5 / (step + 1)
    tracker.log(step=step, metrics={"train/loss": loss})

# Remember to close it at the end
tracker.close()

"""
