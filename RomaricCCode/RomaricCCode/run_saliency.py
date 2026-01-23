import tonic
import saliency
import numpy as np
import matplotlib.pyplot as plt

# venv must be activated
# to run the code: python RomaricCCode/run_saliency.py
# to run the build: cd RomaricCCode && pip install -e . --no-cache-dir


# -------- Dataset --------
dataset = tonic.datasets.DVSGesture(save_to="../Datasets", train=True)
events, target = dataset[0]

# -------- Visualization --------
plt.ion()

def on_close(event):
    global running
    running = False
    print("Window closed — stopping processing")

fig = plt.figure()
fig.canvas.mpl_connect("close_event", on_close)

running = True
current_saliency = None

def on_saliency(frame):
    if not running:
        return
    
    plt.clf()
    plt.imshow(frame, cmap="hot")
    plt.title("Saliency")
    plt.colorbar()
    plt.pause(0.001)
    global current_saliency
    current_saliency = frame
    

# -------- Saliency detector --------
detector = saliency.DetectSaliency(on_saliency)

# -------- Process ONE sample --------
events, label = dataset[0]

print("Label:", label)
print("Events shape:", events.shape)

# -------- Filter events based on saliency --------

filtered_events = []

for e in events:
    if not running:
        break

    x, y, t = int(e["x"]), int(e["y"]), int(e["t"])
    detector.feed(x, y, t)

    if current_saliency is not None:
        if current_saliency[y, x] > 0.3:
            filtered_events.append(e)

print(f"Original events: {len(events)}, Filtered events: {len(filtered_events)}")

# events fields: x, y, t, p

"""
# -------- Load DSEC data --------
import numpy as np
import matplotlib.pyplot as plt
import saliency  # assuming your saliency module

# -------- Load DSEC-style events --------
# Columns: x y t p
data = np.loadtxt("../Dsec.txt")

# Optional: ensure correct dtype
data = data.astype(np.float32)

print("Loaded events:", data.shape)

# -------- Visualization --------
plt.ion()

running = True
current_saliency = None

def on_close(event):
    global running
    running = False
    print("Window closed — stopping processing")

fig = plt.figure()
fig.canvas.mpl_connect("close_event", on_close)

def on_saliency(frame):
    global current_saliency
    if not running:
        return

    plt.clf()
    plt.imshow(frame, cmap="hot")
    plt.title("Saliency")
    plt.colorbar()
    plt.pause(0.001)

    current_saliency = frame


# -------- Saliency detector --------
detector = saliency.DetectSaliency(on_saliency)

# -------- Process events --------
filtered_events = []

for e in data:
    if not running:
        break

    x = int(e[0])
    y = int(e[1])
    t = int(e[2])
    p = int(e[3])  # polarity (unused unless needed)

    detector.feed(x, y, t)

    if current_saliency is not None:
        if current_saliency[x, ] > 0.3:
            filtered_events.append(e)


filtered_events = np.array(filtered_events)

print(f"Original events: {len(data)}")
print(f"Filtered events: {len(filtered_events)}")
"""