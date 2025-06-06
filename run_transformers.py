from pathlib import Path
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


model_id = "google/gemma-3-4b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
    #local_files_only=True,
    #load_in_4bit=True,
)
    
processor = AutoProcessor.from_pretrained(model_id, padding_side="left",use_fast=True)

# local image file
door = Path("img4.jpg") #door
local_img = Path("captured_image.jpg")            # position
# local_img = Path("/absolute/path/cat.jpg") # absolute path also works

# messages = [
#     {
#         "role": "system",
#         "content": [{"type": "text", "text": "You are a robotics control assistant"}]
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "path": str(local_img)}, 
#             {"type": "text", "text": "Find your way out of the room, only if one exists and is clearly visible fully. Else, navigate to make it possible to take another image. Use high level commands like 'move forward', 'turn left', 'turn right', 'rotate', 'stop'."}
#         ]
#     },
# ]

messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": """You are **NavPilot**, an embodied-AI assistant that controls a wheeled indoor robot.

### Capabilities you MAY use
1. MOVE_FORWARD(<meters: float>)
1. MOVE_BACKWARD(<meters: float>)
2. ROTATE_LEFT(<degrees: int>)
3. ROTATE_RIGHT(<degrees: int>)                                
6. TAKE_PHOTO()                                        

### Constraints
- Never drive into obstacles or humans — if uncertain, prefer ROTATE_* + TAKE_PHOTO().
- If the exit is not fully visible which is the glass door, choose an action that improves visibility, then TAKE_PHOTO().
- Don't move forward if you are at a dead end or if the exit is not visible. The exit is a glass door that is fully visible.
- Always take an image after executing the move commands. Don't take an image if you reached the final destination.
- If there is not much distance infront of you, go rotate left or right to see more of the room.
- If you don't see an exit, it is better to rotate to get a better view of the room.

### Output format
Return **exactly two lines**:
1. The chosen commands, separated by semicolons.
2. REASONING: followed by extensive explanation.

Any extra text or missing line will be ignored by the controller."""
            }
        ],
    },
   {
        "role": "user",
        "content": [
            # ---- reference photo of the target door ----
            {
                "type": "text",
                "text": (
                    "The **first image** shows what the EXIT DOOR looks like. "
                    "Study it carefully so you can recognise it."
                ),
            },
            {"type": "image", "path": str(door)},

            # ---- live robot camera frame ----
            {
                "type": "text",
                "text": (
                    "The **second image** is the robot’s current point-of-view "
                    "inside the room. This is the image you will use to give commands."
                ),
            },
            {"type": "image", "path": str(local_img)},

            # ---- task instruction ----
            {
                "type": "text",
                "text": (
                    "If the door from the first image is already visible in the "
                    "second image, navigate toward it.  Otherwise choose the best "
                    "action to reveal it. Gnerate the command for the second image only. Remember to respond in the ‘command ; "
                    "REASONING’ format."
                ),
            },
        ],
    },
]


inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to("cuda")

output = model.generate(**inputs, max_new_tokens=150, cache_implementation="dynamic")
#print(processor.decode(output[0], skip_special_tokens=True))

prompt_len = inputs["input_ids"].shape[1]

# drop the first prompt_len tokens
assistant_tokens = output[0][prompt_len:]

assistant_text = processor.decode(
    assistant_tokens, skip_special_tokens=True
)


print(assistant_text.strip())

# check if cuda is being used
# print("Torch sees CUDA:", torch.cuda.is_available())
# print("First model parameter device:", next(model.parameters()).device)

