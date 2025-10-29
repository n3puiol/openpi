from transformers import pipeline
import torch
import math
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from PIL import Image
import numpy as np


def get_episode(dataset_name: str, ep_idx: int) -> LeRobotDataset:
    dataset = LeRobotDataset(dataset_name, episodes=[ep_idx])
    return dataset


def get_dataset_metadata(dataset_name: str) -> LeRobotDatasetMetadata:
    metadata = LeRobotDatasetMetadata(dataset_name)
    return metadata


def get_episode_subtasks(episode: LeRobotDataset) -> list[str]:
    task = episode[0]["task"]
    start_view = (episode[0]["image"].permute(1, 2, 0).cpu().numpy() * 255).astype(
        np.uint8
    )
    print(start_view.shape)
    print("Task:", task)
    # start_view = Image.fromarray(start_view)
    # start_view.show()
    return []


def find_low_velocity_frames(episode: LeRobotDataset) -> list[int]:
    dt = 1.0
    states = []
    for i in range(len(episode)):
        state = episode[i]["actions"]
        states.append(state)
    states = np.array(states)

    velocities = []
    # Iterate through consecutive pairs of states
    for i in range(10, len(states) - 1):
        p1 = states[i]  # Starting point (x1, y1, z1, x2, y2, z2, z3)
        p2 = states[i + 1]  # Ending point (x1, y1, z1, x2, y2, z2, z3)

        # Calculate squared Euclidean distance (more efficient to compare squares)
        squared_dist = (
            (p2[0] - p1[0]) ** 2
            + (p2[1] - p1[1]) ** 2
            + (p2[2] - p1[2]) ** 2
            + (p2[3] - p1[3]) ** 2
            + (p2[4] - p1[4]) ** 2
            + (p2[5] - p1[5]) ** 2
            + (p2[6] - p1[6]) ** 2
        )

        # Calculate velocity magnitude (speed)
        speed = math.sqrt(squared_dist) / dt
        velocities.append(speed)

    # find 20 lowest velocity frames
    sorted_indices = np.argsort(velocities)[:30]
    sorted_indices = sorted(sorted_indices)
    lowest_velocity_indices = []
    for i in range(len(sorted_indices)):
        if i == 0 or sorted_indices[i] - sorted_indices[i - 1] >= 5:
            lowest_velocity_indices.append(
                sorted_indices[i].item() + 10
            )  # add 10 to match original frame numbers
    return lowest_velocity_indices


def is_subtask_complete(subtask: str, image: Image.Image) -> bool:
    # Placeholder implementation
    return False


def generate(image, task_description: str = "put the white mug on the left plate and put the yellow and white mug on the right plate"):
    pipe = pipeline(
        "image-text-to-text",
        model="google/gemma-3-27b-it",
        device="cuda",
        torch_dtype=torch.bfloat16,
    )
    prompt = """
    You are a robotics assistant that plans tasks. Break down the user's instruction into a sequence of subtasks. For each subtask, define the 'action'.

    Respond ONLY with a JSON object.

    ---
    ## EXAMPLE
    ---
    USER INSTRUCTION: "place the crackers in the bin"
    ASSISTANT RESPONSE:
    {
    "subtasks": [
        {
        "action": "pick up the box of crackers",
        },
        {
        "action": "place in the bin",
        }
    ]
    }

    ---
    ## CURRENT TASK
    ---
    USER INSTRUCTION: "{}"
    ASSISTANT RESPONSE:
    """.format(task_description)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG",
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]

    output = pipe(text=messages, max_new_tokens=350)
    print(output[0]["generated_text"][-1]["content"])


if __name__ == "__main__":
    # generate(image)
    pass
