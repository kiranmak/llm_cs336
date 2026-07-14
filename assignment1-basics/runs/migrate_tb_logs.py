import os
import glob
import json
import shutil
from torch.utils.tensorboard import SummaryWriter

log_dir = './logs'
backup_dir = './old_tfevents_backup'

# 1. Recreate proper event files in subfolders from metrics.jsonl
jsonl_files = glob.glob(os.path.join(log_dir, '*/metrics.jsonl'))

for file_path in jsonl_files:
    dir_path = os.path.dirname(file_path)
    print(f"Recreating TensorBoard event logs for: {dir_path}")
    
    writer = SummaryWriter(log_dir=dir_path)
    
    with open(file_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            step = record['step']
            
            # Log all numerical scalars in the record
            for key, val in record.items():
                if key in ('step', 'wall_time_s'):
                    continue
                if isinstance(val, (int, float)):
                    writer.add_scalar(key, val, global_step=step)
            
            # Log wall time explicitly
            if 'wall_time_s' in record:
                writer.add_scalar("meta/wall_time_s", record["wall_time_s"], global_step=step)
                
    writer.flush()
    writer.close()

# 2. Safely move top-level event files out of logs/ to backup_dir
top_level_events = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
if top_level_events:
    os.makedirs(backup_dir, exist_ok=True)
    print(f"Moving top-level event files to {backup_dir} to clean up TensorBoard...")
    for event_file in top_level_events:
        shutil.move(event_file, os.path.join(backup_dir, os.path.basename(event_file)))

print("Migration completed! Please restart TensorBoard using: tensorboard --logdir=logs")
