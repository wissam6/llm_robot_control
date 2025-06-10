import time
import ast
import os
import json
import math
from datetime import datetime
from zoneinfo import ZoneInfo  
from ollama import chat, ChatResponse
import requests
import pyttsx3
from typing import Dict

# Text-to-speech setup
_tts_engine = pyttsx3.init()
_tts_engine.setProperty("rate", 180)
_tts_engine.setProperty("volume", 0.9)

def say(text: str, wait: bool = True) -> None:
    _tts_engine.say(text)
    if wait:
        _tts_engine.runAndWait()
    else:
        import threading
        threading.Thread(target=_tts_engine.runAndWait, daemon=True).start()

# --- Position tracking & logging ---
class Pose:
    def __init__(self, x=0.0, y=0.0, heading=0.0):
        self.x = x
        self.y = y
        self.heading = heading  # degrees, 0 = initial forward

    def update(self, command):
        dx, theta = command
        # update heading first
        self.heading = (self.heading + theta) % 360
        # move along new heading
        rad = math.radians(self.heading)
        self.x += dx * math.cos(rad)
        self.y += dx * math.sin(rad)

    def pos_key(self, precision: int = 2):
        return (round(self.x, precision), round(self.y, precision), round(self.heading))

# Globals
current_pose = Pose()
log_file = 'exploration_log.json'
# Load existing exploration log with empty-file handling
if os.path.exists(log_file):
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            if content.strip():
                exploration_log = json.loads(content)
                if not isinstance(exploration_log, list):
                    exploration_log = []
            else:
                exploration_log = []
    except (json.JSONDecodeError, ValueError):
        exploration_log = []
    # Rebuild visited from log entries
    visited = {
        (round(entry.get('pose', {}).get('x', 0.0), 2),
         round(entry.get('pose', {}).get('y', 0.0), 2),
         round(entry.get('pose', {}).get('heading', 0.0)))
        for entry in exploration_log
        if isinstance(entry, dict) and 'pose' in entry
    }
    visited.add(current_pose.pos_key())
else:
    exploration_log = []
    visited = {current_pose.pos_key()}

# System prompt
SYSTEM_PROMPT = """
You are **NavPilot**, an embodied-AI assistant that controls a wheeled indoor robot.

### Capabilities you MAY use
1. MOVE_FORWARD(<meters: float 0 to 5.0>)
2. MOVE_BACKWARD(<meters: float 0 to 5.0>)
3. ROTATE_LEFT(<degrees: int 1 to 90>)
4. ROTATE_RIGHT(<degrees: int 1 to 90>)

### Constraints
- in your output dont inlcude ```
- do not go under the tables
- do not do any numeric ordering for the output
- Always avoid collisions: If the forward distance sensor reports any object ≤ 0.50 m straight ahead, do not move forward. Instead, choose a different heading (rotate or reverse) to maintain a safe clearance.
- The safe distance that you can choose to travel is the one from the distance sensor minus 0.5 meters (distance - 0.5).
- Do not output [0,0] unless you intend to stay still.

### Output format  
Return **exactly three separate lines without any ``` on the first line**:  
1. The chosen command with the distance/degrees. It should be an array only containing the x and theta(rotation degrees) as positive values. For example, more forward should be [0.5,0] and rotate right 90 degrees should be [0,90]. You are not allowed to have rotation with movement.  
2. REASONING: followed by extensive explanation.
3. The exact command. For example, MOVE_BACKWARD
"""

# Helper functions

def fetch_image(api_url: str = "http://192.168.4.1:5000/capture", folder: str = "captures") -> str:
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(folder, f"captured_{timestamp}.jpg")
    resp = requests.get(api_url, timeout=5)
    resp.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(resp.content)
    print("📸  Saved image →", save_path)
    return save_path


def fetch_center_distance(api_url: str = "http://192.168.4.1:5000/distance/center", timeout: int = 3) -> float:
    resp = requests.get(api_url, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if "error" in payload:
        raise RuntimeError(f"Sensor error: {payload['error']}")
    if "distance_m" not in payload:
        raise RuntimeError(f"Malformed payload: {payload}")
    distance = float(payload["distance_m"])
    print(f"📏  Center distance → {distance:.3f} m")
    return distance


def send_move_command(command, api_url: str = "http://192.168.4.1:5000/move"):
    x, theta = command
    payload = {"x": x, "y": 0, "theta": theta}
    resp = requests.post(api_url, json=payload)
    resp.raise_for_status()
    print(f"✅ Command '{command}' sent: {resp.json()}")

# Main loop

def call_ollama(image_path: str) -> bool:
    # distance = fetch_center_distance()
    distance = 2

    # Serialize pose and visited context
    pose_descr = f"POSITION: x={current_pose.x:.2f}, y={current_pose.y:.2f}, heading={current_pose.heading:.0f}°."
    recent = list(visited)[-10:]
    visited_descr = "VISITED: " + ", ".join(f"({x:.1f},{y:.1f})" for x,y,_ in recent)

    # Build exploratory prompt
    USER_PROMPT = (
        f"{pose_descr} {visited_descr} "
        f"You need to map the surrounding area and find the exit (glass door). "
        f"Favor moves that take you to unexplored positions, and when safe, rotate to scan new viewpoints. "
        f"If no forward path leads to a novel position, choose a rotation to uncover unseen areas. "
        f"The closest obstacle is {distance:.2f} m ahead. "
        f"Avoid any move that returns you to a previously visited position."
    )

    response: ChatResponse = chat(
        model='gemma3:12b',
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_PROMPT,   "images": [image_path]},
        ],
    )

    lines = response.message.content.splitlines()
    command_arr = ast.literal_eval(lines[0])
    reasoning   = "\n".join(lines[1:]).strip()

    # Adjust rotation sign
    if "ROTATE_RIGHT" in reasoning.upper() and command_arr[1] > 0:
        command_arr[1] = -command_arr[1]

    # Check revisit
    temp_pose = Pose(current_pose.x, current_pose.y, current_pose.heading)
    temp_pose.update(command_arr)
    if temp_pose.pos_key() in visited:
        print(f"⚠️ Already visited {temp_pose.pos_key()}, skipping")
        return False

    # Execute if non-zero
    if any(c != 0 for c in command_arr):
        proceed = input("Proceed with command? y|n ").strip().lower() == "y"
        if proceed:
            # send_move_command(command_arr)
            current_pose.update(command_arr)
            visited.add(current_pose.pos_key())
            # Append new entry
            exploration_log.append({
                "image": image_path,
                "command": command_arr,
                "reasoning": reasoning,
                "pose": {"x": current_pose.x, "y": current_pose.y, "heading": current_pose.heading}
            })
            # Write full log
            with open(log_file, 'w') as f:
                json.dump(exploration_log, f, indent=2)
        return proceed
    return False

if __name__ == "__main__":
    # img_path = fetch_image()
    # cont = call_ollama(img_path)
    # while cont:
    #     time.sleep(2)
    #     img_path = fetch_image()
    #     cont = call_ollama(img_path)
    
    call_ollama("img3.jpg")
