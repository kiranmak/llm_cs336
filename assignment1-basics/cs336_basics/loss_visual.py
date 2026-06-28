import re
import matplotlib.pyplot as plt
from cs336_basics.paths import PROJECT_ROOT, DATA_PATH, OUT_PATH

steps = []
losses = []

# Regex to find 'Step' and 'Loss' values
# This pattern looks for 'Step' followed by one or more spaces and digits, 
# then 'Loss:' followed by one or more spaces and a floating-point number.
pattern  = r"Step\s*(\d+)\s*\| Loss:\s*([0-9.]+)"
prefix   = "TinyStoriesV2-GPT4"
datatype = "samples"
filepath = OUT_PATH / f"loss_{prefix}-{datatype}.txt"

with open(filepath, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                if (step % 5000) == 0:
                    steps.append(step)
                    losses.append(loss)

# Plotting the loss
plt.figure(figsize=(10, 6))
plt.plot(steps, losses, marker='o', linestyle='-', color='b')
plt.title('Training Loss Over Steps')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
