import json
from abc import ABC, abstractmethod
from typing import Tuple

from ..utils.datatypes import State
instr = """You are a helpful assistant to do some scientific experiment in an environment.
In the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway
You should explore the environment and find the items you need to complete the experiment.
You can teleport to any room in one step.
All containers in the environment have already been opened, you can directly get items from the containers.

The available actions are:
open OBJ: open a container
close OBJ: close a container
activate OBJ: activate a device
deactivate OBJ: deactivate a device
connect OBJ to OBJ: connect electrical components
disconnect OBJ: disconnect electrical components
use OBJ [on OBJ]: use a device/item
look around: describe the current room
examine OBJ: describe an object in detail
look at OBJ: describe a container's contents
read OBJ: read a note or book
move OBJ to OBJ: move an object to a container
pick up OBJ: move an object to the inventory
pour OBJ into OBJ: pour a liquid into a container
mix OBJ: chemically mix a container
teleport to LOC: teleport to a specific room
focus on OBJ: signal intent on a task object
wait: task no action for 10 steps
wait1: task no action for a step

You will be given one example and one task below. The example is to showcase how you should do when you are given the task.

In this example, there are several key elements. 
First is the task you need to complete. You need to break this task down into several steps, each corresponding to a thought, action, and observation. 
The thought is your reasoning for this step, and the action is the move you make based on this reasoning. 
The action must be selected from the aforementioned available actions. 
You need to generate the thought and action without any other words, while the observation is based on the environment you see after performing your action. 
You do not need to generate the observation; it will be provided to you after you generate the thought and action.

Here is the example:
"""

raw_icl = [
    [
        {
            "role": "user",
            "content": "Your task is to use chemistry to create green paint. When you are done, focus on the green paint."
        },
        {
            "role": "assistant",
            "content": "Thought: I need to find materials to create green paint, which are likely to be found in an art studio.\nAction: look around"
        },
        {
            "role": "user",
            "content": "Observation: This room is called the bedroom. In it, you see: \n\tthe agent\n\ta substance called air\n\ta bed. On the bed is: a mattress. On the mattress is: a white pillow..\n\ta closet. The closet door is open. In the closet is: nothing.\n\ta picture\n\ta table. On the table is: nothing.\nYou also see:\n\tA door to the hallway (that is open)"
        },
        {
            "role": "assistant",
            "content": "Thought: The bedroom doesn't have the materials I need. I should go to a place where paint and painting supplies are likely to be found, such as an art studio.\nAction: teleport to art studio"
        },
        {
            "role": "user",
            "content": "Observation: You teleport to the art studio."
        },
        {
            "role": "assistant",
            "content": "Thought: Now that I'm in the art studio, I should look around to find the necessary materials to create green paint.\nAction: look around"
        },
        {
            "role": "user",
            "content": "Observation: This room is called the art studio. In it, you see: \n\tthe agent\n\ta substance called air\n\ta large cupboard. The large cupboard door is open. In the large cupboard is: a drawer, a wood cup (containing yellow paint), a wood cup (containing blue paint), a wood cup (containing red paint).\n\ta table. On the table is: a glass cup (containing nothing).\n\ta wood cup (containing yellow paint)\n\ta wood cup (containing blue paint)\n\ta wood cup (containing red paint)\nYou also see:\n\tA door to the hallway (that is open)"
        },
        {
            "role": "assistant",
            "content": "Thought: I see blue paint here, which is one of the primary colors needed to mix green paint. I should start by pouring it into an empty container to mix with another color.\nAction: pour cup containing blue paint in art studio in cup containing nothing"
        },
        {
            "role": "user",
            "content": "Observation: You pour the contents of the wood cup into the glass cup."
        },
        {
            "role": "assistant",
            "content": "Thought: Yellow paint is the other primary color needed to mix with blue to create green. I should add the yellow paint to the container with blue paint.\nAction: pour cup containing yellow paint in art studio in cup containing blue paint in table"
        },
        {
            "role": "user",
            "content": "Observation: You pour the contents of the wood cup into the glass cup."
        },
        {
            "role": "assistant",
            "content": "Thought: Now that I have both blue and yellow paint in the same container, I need to mix them together to create green paint.\nAction: mix cup containing blue paint and yellow paint"
        },
        {
            "role": "user",
            "content": "Observation: Blue and yellow paint mix to produce green paint."
        },
        {
            "role": "assistant",
            "content": "Thought: The task is to create green paint and then focus on it. Now that the green paint is made, I should concentrate on the final product.\nAction: focus on green paint"
        }
    ]
]

class BaseEnv(ABC):
    def __init__(
        self,
        instruction_path: str,
        icl_path: str,
        icl_format: str = "first",
        max_steps: int = 10,
        **kwargs,
    ):
        # with open(instruction_path) as f:
        #     self.instruction = f.read()
        self.instruction = instr
        # self.raw_icl = json.load(open(icl_path))
        self.raw_icl = raw_icl
        self.icl_format = icl_format
        self.max_steps = max_steps


    @abstractmethod
    def step(self, llm_output: str) -> Tuple[str, State]:
        pass

    @abstractmethod
    def reset(self) -> Tuple[str, State]:
        pass
