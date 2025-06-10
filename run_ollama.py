import time
import ast
import os
from datetime import datetime
from zoneinfo import ZoneInfo  
from ollama import chat
from ollama import ChatResponse
import requests
# import pyttsx3
from typing import Dict

# _tts_engine = pyttsx3.init()
# _tts_engine.setProperty("rate", 180)       # words-per-minute
# _tts_engine.setProperty("volume", 0.9)

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
- You should incorporate the previously outputed commands (memory) into your decision-making.

### Output format  
Return **exactly three separate lines without any ``` on the first line**:  
1. The chosen command with the distance/degrees. It should be an array only containing the x and theta(rotation degrees) as positive values. For example, more forward should be [0.5,0] and rotate right 90 degrees should be [0,90]. You are not allowed to have rotation with movement.  
2. REASONING: followed by extensive explanation.
3. The exact command. For example, MOVE_BACKWARD
"""

USER_PROMPT1 = "Find your way out of the room. The exit is a glass door"
USER_PROMPT2 = "Find your way out of the room. The distance between you and the bag is 0.5 meters."
USER_PROMPT3 = "Rotate right 90 degrees"

# assistant_history = []

# def say(text: str, wait: bool = True) -> None:
#     """
#     Speak *text* aloud.  If `wait` is False the call returns
#     immediately and audio plays in the background.
#     """
#     _tts_engine.say(text)
#     if wait:
#         _tts_engine.runAndWait()
#     else:
#         # run non-blocking in a worker thread
#         import threading
#         threading.Thread(target=_tts_engine.runAndWait, daemon=True).start()

# def get_center_distance():
#     try:
#         distance = distance_estimator.get_center_distance()
#         return jsonify({"angle": 0, "distance_m": round(distance, 3)})
#     except Exception as e:  # pragma: no cover
#         return jsonify({"error": str(e)}), 500

def fetch_image(api_url: str = "http://192.168.4.1:5000/capture",
                folder: str = "captures") -> str:
    """Capture a frame and save it as captures/captured_YYYYMMDD_HHMMSS.jpg."""
    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(folder, f"captured_{timestamp}.jpg")

    try:
        resp = requests.get(api_url, timeout=5)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(resp.content)
        print("📸  Saved image →", save_path)
        return save_path
    except requests.RequestException as e:
        raise RuntimeError(f"Camera capture failed: {e}") from e
      
def fetch_center_distance(
    api_url: str = "http://192.168.4.1:5000/distance/center",
    timeout: int = 3,
) -> float:
    """
    Call the `/distance/center` endpoint and return the distance (metres).

    Raises `RuntimeError` if the request fails or the payload is malformed.
    """
    try:
        resp = requests.get(api_url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()

        # The Flask endpoint returns either {"angle": 0, "distance_m": 0.742}
        # or {"error": "..."}.
        if "error" in payload:
            raise RuntimeError(f"Sensor error: {payload['error']}")
        if "distance_m" not in payload:
            raise RuntimeError(f"Malformed payload: {payload}")

        distance = float(payload["distance_m"])
        print(f"📏  Center distance → {distance:.3f} m")
        return distance

    except requests.RequestException as e:
        raise RuntimeError(f"Distance query failed: {e}") from e


def fetch_all_distances(
    api_url: str = "http://192.168.4.1:5000/distance/all",
    timeout: int = 3,
) -> Dict[int, float]:
    """
    Call the `/distance/all` endpoint and return `{angle: metres}`.

    Example result: `{ -30: 0.711, 0: 0.742, 30: 0.695 }`
    """
    try:
        resp = requests.get(api_url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()

        if "error" in payload:
            raise RuntimeError(f"Sensor error: {payload['error']}")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Malformed payload: {payload}")

        # Convert keys and values to the exact types we want
        distances = {int(angle): float(dist) for angle, dist in payload.items()}

        pretty = ", ".join(f"{ang:+d}°={dist:.3f} m"
                           for ang, dist in sorted(distances.items()))
        print(f"📊  All distances → {pretty}")
        return distances

    except requests.RequestException as e:
        raise RuntimeError(f"Distance query failed: {e}") from e
        
def send_move_command(command, api_url="http://192.168.4.1:5000/move"):
    x=command[0]
    #y = command[1]
    theta = command[1]
    
    print("commands", command)
    
    payload = {"x": x, "y": 0, "theta": theta}
    try:
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            print(f"Command '{command}' sent successfully:", response.json())
        else:
            print(f"Failed to send command '{command}'. Status code: {response.status_code}")
            print("Response:", response.text)
    except Exception as e:
        print(f"Error sending command '{command}': {e}")


# At module level
assistant_history: list[str] = []

def call_ollama(image_path: str) -> bool:
    distance = fetch_center_distance()
    # distance = 2.0  # For testing, use a fixed distance
    
    # 1) Build your user prompt as before
    USER_PROMPT = (
        f"history of previously executed commands(memory): {assistant_history}. "
        f"Find your way out of the room. The exit is a glass door. "
        f"The distance to the closest object is {distance:.2f} meters. "
        f"If the distance is 0 it means that there is an object 15 cm or less of it"
    )
    
    print(USER_PROMPT)
    
    # 2) Assemble the messages list, injecting the last N assistant outputs
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    # keep only the last 5 for context (tune as needed)
    for prev in assistant_history[-5:]:
        messages.append({"role": "assistant", "content": prev})
    
    messages.append({
        "role": "user",
        "content": USER_PROMPT,
        "images": [image_path]
    })
    
    # 3) Call Ollama with full context
    response: ChatResponse = chat(
        model='gemma3:12b',
        messages=messages,
    )

    
    # 4) Save the raw assistant output into history BEFORE any post-processing
    assistant_history.append("one command I already executed" + response.message.content + " ")
    
    print("assistant_history", assistant_history)
    
    # 5) Your existing parsing logic
    print(response.message.content)
    lines = response.message.content.splitlines()
    print("lines", lines[0])
    command_arr = ast.literal_eval(lines[0])      # [x, theta]
    reasoning   = "\n".join(lines[2:]).upper()    # easier case-insensitive search
    
    # say("response: " + response.message.content, wait=False)
    
    print("reasoning", reasoning)
    print("before", command_arr)
    # adjust right-turn sign
    if "ROTATE_RIGHT" in reasoning and command_arr[1] > 0:
        print('i am rotating right')
        command_arr[1] = -command_arr[1]
    else:
        print("i am rotating left")
    print("after", command_arr)
    
    # 6) Execute or skip as before
    if any(c != 0 for c in command_arr):
        proceed = input("Proceed with command? y|n ").strip().lower() == "y"
        if proceed:
            send_move_command(command_arr)
            # print("move")
        return proceed
    return False



if __name__ == "__main__":
  img_path = fetch_image()
  flag = call_ollama(img_path)

  
  print(flag)
  while flag:
      time.sleep(2)
      img_path = fetch_image()
      flag = call_ollama(img_path)

#   i = 5
#   while i > 0:
#     call_ollama("img1.jpg")
#     i -= 1

  #send_move_command([1,0], api_url="http://192.168.4.1:5000/move")
  #fetch_center_distance()
  #fetch_all_distances()