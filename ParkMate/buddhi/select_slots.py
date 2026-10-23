"""
BUDDHI - Slot Selector (matplotlib version)
-----------------------------------------------
Uses matplotlib instead of OpenCV's GUI window - this avoids the common
"cv2.imshow window doesn't open" problem caused by opencv-python-headless
being installed alongside ultralytics. Matplotlib's window works out of
the box on Windows without any extra setup.

How it works:
  - A window opens showing your parking photo.
  - For EACH slot: click the TOP-LEFT corner, then click the
    BOTTOM-RIGHT corner (inside the white lines of ONE parking space only).
  - After 2 clicks, that slot is saved automatically and a green box is
    drawn so you can see your progress.
  - Keep going until you've marked every slot you want to track.
  - Close the window (X button) when done - slots.json is saved automatically.

Usage:
    python select_slots.py --image parking.jpg
"""

import argparse
import json
import matplotlib
matplotlib.use("TkAgg")  # reliable default GUI backend on Windows
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

slots = []
click_buffer = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Reference image from your fixed camera")
    parser.add_argument("--out", default="slots.json")
    args = parser.parse_args()

    img_bgr = cv2.imread(args.image)
    if img_bgr is None:
        raise FileNotFoundError(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img_rgb)
    ax.set_title("Click TOP-LEFT then BOTTOM-RIGHT of each slot (one space at a time).\n"
                  "Close this window when finished - slots.json saves automatically.")

    def redraw():
        # remove old rectangles, keep the image
        for p in list(ax.patches):
            p.remove()
        for i, (x1, y1, x2, y2) in enumerate(slots):
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      linewidth=2, edgecolor="lime", facecolor="none")
            ax.add_patch(rect)
            ax.text(x1 + 4, y1 + 18, str(i + 1), color="lime", fontsize=10, weight="bold")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.xdata is None or event.ydata is None:
            return  # click was outside the image
        click_buffer.append((event.xdata, event.ydata))
        if len(click_buffer) == 2:
            (x1, y1), (x2, y2) = click_buffer
            box = (int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2)))
            slots.append(box)
            click_buffer.clear()
            redraw()
            print(f"Slot {len(slots)} saved: {box}")

    def on_close(event):
        data = [{"id": i + 1, "box": list(box)} for i, box in enumerate(slots)]
        with open(args.out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved {len(slots)} slots to {args.out}")

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("close_event", on_close)

    print("Window opening... click TOP-LEFT then BOTTOM-RIGHT of each slot.")
    print("Close the window when you're done - it saves automatically.")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()