#!/usr/bin/env python
"""
Iterative Inference script for LLaVA-3D model with BEHAVIOR execution, External Reflection, and Retro-Reflection.
MEMORY OPTIMIZED VERSION - Uses <24GB GPU memory

This script runs inference iteratively:
1. Executes action and captures observation to room_pointclouds_interact
2. Generates external reflection based on execution result and observation
3. Builds next prompt using previous action and external reflection
4. Maintains a buffer of all GO TO and PICK UP actions with their most recent reflections
5. On EXIT: performs retro-reflection on ALL actions in the buffer
6. By default stops after first iteration (use --max-iterations N for more)

Output: room_pointclouds_interact/{scene_name}_scene_dict.json_1_1/{room_name}_{step_idx}/
"""

import argparse
import os
import sys
import json
import random
import copy
import time
import re
import yaml
from typing import Optional, Dict, List

import torch
import torch as th
import numpy as np
import cv2
import transformers
from PIL import Image

from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_BOX_TOKEN
)
from llava.conversation import conv_templates
from llava.model import LlavaLlamaForCausalLM
from llava.mm_utils import tokenizer_image_token, PlainBoxFormatter

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

import omnigibson as og
from omnigibson.macros import gm
import omnigibson.utils.transform_utils as T
from omnigibson.utils.object_state_utils import sample_kinematics
from omnigibson.utils.usd_utils import create_joint, delete_or_deactivate_prim

from collections import defaultdict

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
torch._dynamo.disable()

gm.ENABLE_FLATCACHE = False
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = False
gm.RENDER_VIEWER_CAMERA = False

AG_JOINT_PRIM = None
MAX_NAVIGATION_ATTEMPTS = 25
MAX_EXPLORATION_ATTEMPTS = 5
IMG_WIDTH = 560
IMG_HEIGHT = 560

POINTCLOUD_DIR = "room_pointclouds_compare"


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def load_json_data(data_path: str) -> List[Dict]:
    """Load training data from JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data


def preprocess_multimodal(source: Dict, mm_use_im_start_end: bool = False) -> List[Dict]:
    """Preprocess multimodal conversation data."""
    conversations = copy.deepcopy(source["conversations"])
    
    for sentence in conversations:
        if DEFAULT_IMAGE_TOKEN in sentence['value'] or DEFAULT_VIDEO_TOKEN in sentence['value']:
            sentence['value'] = sentence['value'].replace(DEFAULT_VIDEO_TOKEN, DEFAULT_IMAGE_TOKEN)
            sentence['value'] = sentence['value'].strip()
            
            replace_token = DEFAULT_IMAGE_TOKEN
            if mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    
    return conversations

def get_most_recent_observation_for_rooms(
    explored_rooms: List[str],
    scene_dir_name: str,
    pointcloud_dir: str = POINTCLOUD_DIR
) -> List[str]:
    """
    Get the most recent observation path for each explored room.
    
    Args:
        explored_rooms: List of rooms that have been explored
        scene_dir_name: Scene directory name (e.g., "scene_config_1_1")
        pointcloud_dir: Base pointcloud directory
        
    Returns:
        List of observation paths, one per explored room (most recent)
    """
    scene_path = os.path.join(pointcloud_dir, scene_dir_name)
    
    observation_paths = []
    
    if not os.path.exists(scene_path):
        return observation_paths
    
    for room in explored_rooms:
        room_dirs = []
        for dir_name in os.listdir(scene_path):
            if dir_name.startswith(f"{room}_"):
                try:
                    step_num = int(dir_name.split('_')[-1])
                    room_dirs.append((step_num, dir_name))
                except:
                    pass
        
        if room_dirs:
            room_dirs.sort(key=lambda x: x[0], reverse=True)
            most_recent_dir = room_dirs[0][1]
            obs_path = f"{pointcloud_dir}/{scene_dir_name}/{most_recent_dir}"
            observation_paths.append(obs_path)
    
    return observation_paths

def prepare_video_input(
    video_paths: List[str],
    video_folder: str,
    video_processor,
    device: torch.device,
    dtype: torch.dtype
) -> Dict:
    """Process video/3D data and prepare inputs."""
    video_dicts = []
    
    for video_path in video_paths:
        full_path = os.path.join(video_folder, video_path)
        video_dict = video_processor.preprocess(full_path, return_tensors='pt')
        video_dicts.append(video_dict)
    
    if not video_dicts:
        return {
            'images': None,
            'world_points': None
        }
    
    images = []
    world_points = []
    
    for video_dict in video_dicts:
        images.append(video_dict['images'])
        wp = video_dict['world_points']
        wp = torch.nan_to_num(wp, nan=0.0, posinf=0.0, neginf=0.0)
        world_points.append(wp)
    
    min_len = min(wp.shape[0] for wp in world_points) if len(video_paths) <= 3 else int(45 / len(video_paths))
    world_points = torch.stack([wp[:min_len] for wp in world_points])
    images = [im[:min_len] for im in images]
    
    return {
        'images': images,
        'world_points': world_points.to(device=device, dtype=dtype)
    }


def build_prompt(conversations: List[Dict], conv_mode: str = "v1") -> str:
    """Build the prompt from conversation history (only human turns for inference)."""
    conv = conv_templates[conv_mode].copy()
    
    for turn in conversations:
        role = turn["from"]
        if role == "human":
            conv.append_message(conv.roles[0], turn["value"])
        elif role == "gpt":
            break
    
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def select_best_action_by_internal_reflection(
    candidate_actions: List[str],
    model,
    tokenizer,
    video_processor,
    task: str,
    total_rooms: List[str],
    explored_rooms: List[str],
    current_room: str,
    previous_action: str,
    previous_reflection: str,
    video_paths: List[str],
    video_folder: str,
    conv_mode: str,
    device: torch.device,
    dtype: torch.dtype,
    room_visit_count
) -> tuple:
    """
    Generate internal reflections for multiple candidate actions and select the best one.
    
    Returns:
        (best_action, best_score, all_reflections)
        where all_reflections is a list of dicts with action, reflection, score
    """
    print(f"\n{'='*60}")
    print(f"SELECTING BEST ACTION FROM {len(candidate_actions)} CANDIDATES")
    print(f"{'='*60}")
    
    action_reflections = []
    
    for idx, action in enumerate(candidate_actions):
        print(f"\n--- Candidate {idx+1}/{len(candidate_actions)}: {action} ---")
        
        internal_sample = build_internal_reflection_prompt(
            task=task,
            total_rooms=total_rooms,
            explored_rooms=explored_rooms,
            current_room=current_room,
            previous_action=previous_action,
            previous_reflection=previous_reflection,
            potential_action=action,
            video_paths=video_paths,
            room_visit_count=room_visit_count
        )
        
        conversations = preprocess_multimodal(
            internal_sample,
            mm_use_im_start_end=False
        )
        
        prompt = build_prompt(conversations, conv_mode)
        
        video_data = prepare_video_input(
            video_paths,
            video_folder,
            video_processor,
            device,
            dtype
        )
        
        output = run_inference(
            model,
            tokenizer,
            prompt,
            images=video_data['images'],
            world_points=video_data['world_points'],
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
            device=device,
            use_quantization=False
        )
        
        reflection, score_str = parse_reflection_and_score(output)
        
        try:
            if '/' in score_str:
                numeric_score = float(score_str.split('/')[0].strip())
            else:
                import re
                numbers = re.findall(r'\d+\.?\d*', score_str)
                numeric_score = float(numbers[0]) if numbers else 0.0
        except:
            numeric_score = 0.0
        
        print(f"  Internal Reflection: {reflection}")
        print(f"  Score: {score_str} (numeric: {numeric_score})")
        
        action_reflections.append({
            'action': action,
            'reflection': reflection,
            'score_str': score_str,
            'numeric_score': numeric_score,
            'full_output': output
        })
        
        del internal_sample, conversations, prompt, output
        if video_data['images'] is not None:
            del video_data['images']
        if video_data['world_points'] is not None:
            del video_data['world_points']
        del video_data
        torch.cuda.empty_cache()
    
    best_action_data = max(action_reflections, key=lambda x: x['numeric_score'])
    
    print(f"\n{'='*60}")
    print(f"BEST ACTION SELECTED")
    print(f"{'='*60}")
    print(f"Action: {best_action_data['action']}")
    print(f"Score: {best_action_data['score_str']} ({best_action_data['numeric_score']})")
    print(f"Reflection: {best_action_data['reflection']}")
    print(f"\nAll Candidates:")
    for i, ar in enumerate(sorted(action_reflections, key=lambda x: x['numeric_score'], reverse=True)):
        print(f"  {i+1}. {ar['action']} - Score: {ar['numeric_score']}")
    print(f"{'='*60}\n")
    
    return best_action_data['action'], best_action_data['score_str'], action_reflections


@torch.inference_mode()
def run_inference(
    model,
    tokenizer,
    prompt: str,
    images: Optional[List[torch.Tensor]] = None,
    world_points: Optional[torch.Tensor] = None,
    clicks: Optional[torch.Tensor] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    do_sample: bool = True,
    device: torch.device = torch.device("cuda"),
    use_quantization: bool = False,
) -> str:
    """Run inference with the model."""
    torch.cuda.empty_cache()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, return_tensors='pt'
    ).unsqueeze(0).to(device)
   
    gen_kwargs = {
        "inputs": input_ids,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
    }
    
    if images is not None:
        images = [image.to(device=device).bfloat16() for image in images]
        gen_kwargs["images"] = [images]
    if world_points is not None:
        gen_kwargs["world_points"] = world_points.unsqueeze(0).to(device=device).bfloat16()
    if clicks is not None:
        gen_kwargs["clicks"] = clicks.to(device=device, dtype=torch.bfloat16)
    
    if use_quantization:
        with torch.no_grad(), torch.amp.autocast("cuda", torch.bfloat16):
            output_ids = model.generate(**gen_kwargs)
    else:
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output_ids = model.generate(**gen_kwargs)
    
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    if prompt in output:
        output = output.split(prompt)[-1].strip()
    
    del input_ids, output_ids, world_points, images, gen_kwargs

    torch.cuda.empty_cache()
    
    return output


def extract_scene_from_sample(sample: Dict) -> str:
    """Extract scene name from video paths in sample."""
    if 'video' in sample and sample['video']:
        video_path = sample['video'][0]
        parts = video_path.split('/')
        if len(parts) >= 2:
            scene_part = parts[1]
            scene_name = scene_part.split('_scene_dict')[0]
            return scene_name
    return "Ihlen_1_int"


def extract_rooms_from_sample(sample: Dict) -> List[str]:
    """Extract total rooms from conversation in sample."""
    if 'conversations' in sample:
        for conv in sample['conversations']:
            if conv['from'] == 'human':
                text = conv['value']
                match = re.search(r'Total rooms:\s*([^.]+)\.', text)
                if match:
                    rooms_str = match.group(1)
                    return [r.strip() for r in rooms_str.split(',')]
    return []


def extract_task_from_sample(sample: Dict) -> str:
    """Extract task description from sample."""
    if 'conversations' in sample:
        for conv in sample['conversations']:
            if conv['from'] == 'human':
                text = conv['value']
                match = re.search(r'Your task:\s*([^\n]+)', text)
                if match:
                    return match.group(1).strip()
    return "Complete the task"


def extract_current_room_from_sample(sample: Dict) -> Optional[str]:
    """Extract current room from conversation in sample."""
    if 'conversations' in sample:
        for conv in sample['conversations']:
            if conv['from'] == 'human':
                text = conv['value']
                match = re.search(r'Your current room:\s*(\w+)', text)
                if match:
                    return match.group(1).strip()
    return None


def parse_reflection_and_score(output: str) -> tuple:
    """Parse reflection text and score from model output."""
    reflection = ""
    score = ""
    
    reflection_match = re.search(r'(?:External Reflection|Retro Reflection):\s*(.+?)(?:;\s*Score:|$)', output, re.DOTALL)
    if reflection_match:
        reflection = reflection_match.group(1).strip()
    else:
        reflection = output.strip()
    
    score_match = re.search(r'Score:\s*([^;\n]+)', output)
    if score_match:
        score = score_match.group(1).strip()
    
    return reflection, score


def generate_multiple_actions_from_model(
    model,
    tokenizer,
    video_processor,
    task: str,
    total_rooms: List[str],
    explored_rooms: List[str],
    current_room: str,
    previous_action: str,
    previous_reflection: str,
    video_paths: List[str],
    video_folder: str,
    conv_mode: str,
    device: torch.device,
    dtype: torch.dtype,
    num_actions: int = 3,
    temperature: float = 0.9,
    use_quantization: bool = False,
    max_attempts: int = 30,
    room_visit_count: int=1,
) -> List[str]:
    """
    Generate exactly num_actions diverse candidate actions from the model using sampling.
    
    Returns:
        List of exactly num_actions unique action strings generated by the model
    """
    print(f"\n{'='*60}")
    print(f"GENERATING EXACTLY {num_actions} UNIQUE CANDIDATE ACTIONS FROM MODEL")
    print(f"{'='*60}")
    
    sample = build_iterative_prompt(
        task=task,
        total_rooms=total_rooms,
        explored_rooms=explored_rooms,
        current_room=current_room,
        previous_action=previous_action,
        previous_reflection=previous_reflection,
        video_paths=video_paths,
        room_visit_count=room_visit_count
    )
    
    conversations = preprocess_multimodal(
        sample,
        mm_use_im_start_end=False
    )
    
    prompt = build_prompt(conversations, conv_mode)
    
    video_data = prepare_video_input(
        video_paths,
        video_folder,
        video_processor,
        device,
        dtype
    )
    
    unique_actions = []
    seen = set()
    attempts = 0
    
    while len(unique_actions) < num_actions and attempts < max_attempts:
        attempts += 1
        print(f"\nGeneration attempt {attempts} (have {len(unique_actions)}/{num_actions} unique actions)...")
        
        output = run_inference(
            model,
            tokenizer,
            prompt,
            images=video_data['images'],
            world_points=video_data['world_points'],
            max_new_tokens=512,
            temperature=1.8,
            do_sample=True,
            device=device,
            use_quantization=use_quantization,
        )
        
        action = output.strip()
        
        if ("go to" not in action.lower()) and ("pick up" not in action.lower()): 
            continue
        
        if "go to" in action.lower():
            room = action.lower().replace("go to", "").strip()
            if room not in total_rooms: 
                continue

        if action not in seen:
            unique_actions.append(action)
            seen.add(action)
            print(f"  Added unique action {len(unique_actions)}: {action}")
        else:
            print(f"  Duplicate action, skipping: {action}")
        
        torch.cuda.empty_cache()
    
    if video_data['images'] is not None:
        del video_data['images']
    if video_data['world_points'] is not None:
        del video_data['world_points']
    del video_data, sample, conversations, prompt
    torch.cuda.empty_cache()
    
    if len(unique_actions) < num_actions:
        print(f"\nWARNING: Could only generate {len(unique_actions)} unique actions after {max_attempts} attempts")
        print(f"    Padding with variations of the last action...")
        
        while len(unique_actions) < num_actions:
            fallback_action = unique_actions[-1] if unique_actions else "EXIT current_room"
            unique_actions.append(fallback_action)
    
    unique_actions = unique_actions[:num_actions]
    
    print(f"\n{'='*60}")
    print(f"Generated EXACTLY {len(unique_actions)} unique candidate actions")
    for i, action in enumerate(unique_actions):
        print(f"  {i+1}. {action}")
    print(f"{'='*60}")
    
    assert len(unique_actions) == num_actions, f"Expected {num_actions} actions, got {len(unique_actions)}"
    
    return unique_actions

def build_iterative_prompt(
    task: str,
    total_rooms: List[str],
    explored_rooms: List[str],
    current_room: str,
    previous_action: str,
    previous_reflection: str,
    video_paths: List[str],
    room_visit_count
) -> Dict:
    """
    Build a prompt for iterative inference.
    
    Args:
        task: Task description
        total_rooms: List of all rooms in scene
        explored_rooms: List of rooms explored so far
        current_room: Current room location
        previous_action: Previous action taken
        previous_reflection: Reflection on previous action
        video_paths: List of observation paths (previous + current)
    """
    total_rooms_str = ", ".join(total_rooms)
    explored_rooms_str = ", ".join(explored_rooms) if explored_rooms else "none"
    
    if len(video_paths) == 0:
        video_token_str = f"Previous observations: None \nCurrent observation: None"
    else:
        prev_videos = " ".join([DEFAULT_VIDEO_TOKEN] * (len(video_paths) - 1))
        video_token_str = f"Previous observations: <video> \nCurrent observation: <video>"
    
    visit_text = ""
    if room_visit_count and current_room:
        visit_num = room_visit_count.get(current_room, 0)
        if visit_num == 1:
            visit_text = "This is your first time visiting the room \n"
        elif visit_num == 2:
            visit_text = "This is your second time visiting the room \n"
        else:
            visit_text = f"This is your {visit_num}-th time visiting the room \n"
    
    prompt_text = (
        f"You are a 3D embodied AI agent, you can see all the objects in the rooms and interact with the objects. "
        f"Given a task, you need to output the current action to interact with the environment to complete the task. \n"
        f"Your task: {task} \n"
        f"Total rooms: {total_rooms_str}. \n"
        f"Rooms you have explored: {explored_rooms_str}. \n"
        f"Your current room: {current_room} \n"
        f"{visit_text}"
        f"Previous action: {previous_action}. \n"
        f"Previous reflection: {previous_reflection}. \n"
        f"{video_token_str} \n"
        f"What should be the next action?"
    )
    
    print(prompt_text)
    print("video paths:")
    print(video_paths)

    return {
        "video": video_paths,
        "conversations": [
            {
                "from": "human",
                "value": prompt_text
            }
        ]
    }


def build_external_reflection_prompt(
    task: str,
    total_rooms: List[str],
    explored_rooms: List[str],
    current_room: str,
    previous_action: str,
    previous_reflection: str,
    executed_action: str,
    execution_result: str,
    video_paths: List[str],
    room_visit_count
) -> Dict:
    """
    Build a prompt for external reflection generation (based on template from script 2).
    
    Args:
        task: Task description
        total_rooms: List of all rooms in scene
        explored_rooms: List of rooms explored so far
        current_room: Current room location
        previous_action: Previous action taken (before the just-executed one)
        previous_reflection: Previous reflection
        executed_action: The action that was just executed
        execution_result: Result of the execution (success/failure message)
        video_paths: List of observation paths (previous + current)
    """
    total_rooms_str = ", ".join(total_rooms)
    explored_rooms_str = ", ".join(explored_rooms) if explored_rooms else "none"

    visit_text = ""
    if room_visit_count and current_room:
        visit_num = room_visit_count.get(current_room, 0)
        if visit_num == 1:
            visit_text = "This is your first time visiting the room \n"
        elif visit_num == 2:
            visit_text = "This is your second time visiting the room \n"
        else:
            visit_text = f"This is your {visit_num}-th time visiting the room \n"
       
    prompt_text = (
        f"You are a 3D embodied AI agent, you can see all the objects in the rooms and interact with the objects. "
        f"You are given a task to complete. You just executed an action in the environment. "
        f"Based on the execution result and observation feedback (current observation) from the environment, "
        f"you need to provide a reflection on this action and give a score. \n"
        f"Your task: {task}. \n"
        f"Total rooms: {total_rooms_str}. \n"
        f"Rooms you have explored: {explored_rooms_str}. \n"
        f"Your current room: {current_room} \n"
        f"{visit_text}"
        f"Previous action: {previous_action}. \n"
        f"Previous reflection: {previous_reflection}. \n"
        f"Previous observations: <video> \n"
        f"Current observation: <video> \n"
        f"Executed action: {executed_action} \n"
        f"Execution Result: {execution_result} \n"
        f"Give external reflection of your executed action based on current observation and execution result. "
        f"Also provide a score. "
    )
    
    print("\n=== External Reflection Prompt ===")
    print(prompt_text)
    print("video paths:")
    print(video_paths)

    return {
        "video": video_paths,
        "conversations": [
            {
                "from": "human",
                "value": prompt_text
            }
        ]
    }

def build_internal_reflection_prompt(
    task: str,
    total_rooms: List[str],
    explored_rooms: List[str],
    current_room: str,
    previous_action: str,
    previous_reflection: str,
    potential_action: str,
    video_paths: List[str],
    room_visit_count
) -> Dict:
    """Generate prompt for pre-action reflection."""
    total_rooms_str = ", ".join(total_rooms)
    explored_rooms_str = ", ".join(explored_rooms) if explored_rooms else "none"

    visit_text = ""
    if room_visit_count and current_room:
        visit_num = room_visit_count.get(current_room, 0)
        if visit_num == 1:
            visit_text = "This is your first time visiting the room \n"
        elif visit_num == 2:
            visit_text = "This is your second time visiting the room \n"
        else:
            visit_text = f"This is your {visit_num}-th time visiting the room \n"

    
    if len(video_paths) == 0:
        prompt_text = (
            f"You are a 3D embodied AI agent, you can see all the objects in the rooms and interact with the objects. "
            f"You need to complete the task. You are provided with the previous action and reflection, and a potential current action to be carried out. "
            f"You need to generate an internal pre-reflection for the action, and the score of this potential action. \n"
            f"Your task: {task}. \n"
            f"Total rooms: {total_rooms_str}. \n"
            f"Rooms you have explored: {explored_rooms_str}. \n"
            f"Your current room: {current_room} \n"
            f"{visit_text}"
            f"Previous action: {previous_action}. \n"
            f"Previous reflection: {previous_reflection}. \n"
            f"Previous observations: None \n"
            f"Current observation: None \n"
            f"Potential next action: {potential_action}. \n"
            f"Give an internal reflection and score of this potential next action ({potential_action})."
        )

    else:
        prompt_text = (
            f"You are a 3D embodied AI agent, you can see all the objects in the rooms and interact with the objects. "
            f"You need to complete the task. You are provided with the previous action and reflection, and a potential current action to be carried out. "
            f"You need to generate an internal pre-reflection for the action, and the score of this potential action. \n"
            f"Your task: {task}. \n"
            f"Total rooms: {total_rooms_str}. \n"
            f"Rooms you have explored: {explored_rooms_str}. \n"
            f"Your current room: {current_room} \n"
            f"{visit_text}"
            f"Previous action: {previous_action}. \n"
            f"Previous reflection: {previous_reflection}. \n"
            f"Previous observations: <video> \n"
            f"Current observation: <video> \n"
            f"Potential next action: {potential_action}. \n"
            f"Give an internal reflection and score of this potential next action ({potential_action})."
        )
    
    print("\n=== Internal Reflection Prompt ===")
    print(prompt_text)
    print("video paths:", video_paths)

    return {
        "video": video_paths,
        "conversations": [{"from": "human", "value": prompt_text}]
    }


def build_retro_reflection_prompt(
    task: str,
    total_rooms: List[str],
    explored_rooms: List[str],
    current_room: str,
    room_actions_reflections: Dict[str, Dict],
    action_to_retro_reflect: str,
    last_reflection: str,
    last_score: str,
    video_paths: List[str]
) -> Dict:
    """
    Build a prompt for retro-reflection on a previous action.
    Based on the transfer_data_retro.py template.
    
    Args:
        task: Task description
        total_rooms: All rooms in the scene
        explored_rooms: Rooms explored so far
        current_room: Current room
        room_actions_reflections: Dict mapping action -> {reflection, score} for all actions in current room
        action_to_retro_reflect: The specific action to retro-reflect on
        last_reflection: Previous external reflection for this action
        last_score: Previous score for this action
        video_paths: Observation paths (previous + current)
    """
    total_rooms_str = ", ".join(total_rooms)
    explored_rooms_str = ", ".join(explored_rooms) if explored_rooms else "none"
    
    prompt_text = (
        f"You are a 3D embodied AI agent, you can see all the objects in the rooms and interact with the objects. "
        f"You are given a task to complete. You just executed a sequence of actions in the room. "
        f"Based on all reflections on all actions in this room, you need to retro-reflect on your previous action. "
        f"Given previous reflection on action prior to this working memory window (i.e., this room), "
        f"you need to update your external reflection. \n"
        f"Your task: {task}. \n"
        f"Total rooms: {total_rooms_str}. \n"
        f"Rooms you have explored: {explored_rooms_str}. \n"
        f"Your current room: {current_room} \n"
        f"Previous observations: <video> \n"
        f"Current observation: <video> \n"
    )

    prompt_text += f"The prior action that you need to retro-reflect on: {action_to_retro_reflect} "
    prompt_text += f"The previous external reflection you gave to this action: External Reflection: {last_reflection}; Score: {last_score} \n"
    prompt_text += f"Actions and their external reflections in this room: \n"

    for action, data in reversed(room_actions_reflections.items()):
        reflection = data.get('reflection', '')
        score = data.get('score', '')
        prompt_text += f"Action in this room: {action}; External Reflection of this Action: {reflection}; Score of this Action: {score}\n"
    

    prompt_text += f"Retro revise the external reflection of this action ({action_to_retro_reflect}) based on your reflections in this room. "
    
    print("\n=== Retro Reflection Prompt ===")
    print(prompt_text)
    print("video paths:")
    print(video_paths)
    
    return {
        "video": video_paths,
        "conversations": [
            {
                "from": "human",
                "value": prompt_text
            }
        ]
    }


def create_ttt_training_example(
    action_data: Dict,
    retro_reflection: str,
    retro_score: str,
    task: str,
    total_rooms: List[str],
    explored_rooms: List[str],
    current_room: str,
    previous_action: str,
    previous_reflection: str,
    video_paths: List[str]
) -> Dict:
    """
    Create a training example where:
    - Input: Internal reflection prompt for the action
    - Ground Truth: The retro-reflection (revised reflection with more context)
    """
    internal_sample = build_internal_reflection_prompt(
        task=task,
        total_rooms=total_rooms,
        explored_rooms=explored_rooms,
        current_room=current_room,
        previous_action=previous_action,
        previous_reflection=previous_reflection,
        potential_action=action_data['action'],
        video_paths=video_paths,
        room_visit_count=executor.room_visit_count
    )
    
    prompt_text = internal_sample['conversations'][0]['value']
    
    ground_truth = f"Internal Reflection: {retro_reflection}; Score: {retro_score}"
    
    action_type = 'GO_TO' if action_data['action'].upper().startswith('GO TO') else \
                  'PICK_UP' if action_data['action'].upper().startswith('PICK UP') else 'OTHER'
    
    return {
        'action': action_data['action'],
        'action_type': action_type,
        'prompt': prompt_text,
        'ground_truth': ground_truth,
        'video_paths': video_paths,
        'action_idx': action_data.get('action_idx', -1)
    }

from bitsandbytes.nn import Linear4bit, Linear8bitLt

def check_quantization(model):
    """Recursively check the model for quantized layers."""
    quantized_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, Linear4bit):
            quantized_layers.append((name, "4-bit (Linear4bit)"))
        elif isinstance(module, Linear8bitLt):
            quantized_layers.append((name, "8-bit (Linear8bitLt)"))
            
    if quantized_layers:
        print("\nFound Quantized Layers:")
        for name, q_type in quantized_layers:
            print(f"  - {name}: {q_type}")
    else:
        print("\nNo standard 4-bit or 8-bit layers found.")
        print("  (This might mean layers were skipped or the model is using a different quantization method.)")


def perform_test_time_training(
    model,
    tokenizer,
    video_processor,
    training_examples: List[Dict],
    device: torch.device,
    dtype: torch.dtype,
    video_folder: str,
    conv_mode: str,
    num_epochs: int = 3,
    learning_rate: float = 5e-5
):
    """
    Perform test-time training using LoRA on retro-reflection supervised examples.
    """
    print(f"\n{'='*70}")
    print(f"STARTING TEST-TIME TRAINING")
    print(f"{'='*70}")
    print(f"Training on {len(training_examples)} examples")
    print(f"Epochs: {num_epochs}, LR: {learning_rate}")
    
    torch.cuda.empty_cache()

    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
    print("Model prepared for k-bit training")

    model.train()
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.15,
        target_modules=["q_proj", "v_proj"],
        inference_mode=False,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        epoch_loss = 0
        
        for idx, example in enumerate(training_examples):
            torch.cuda.empty_cache()
            prompt_text = example['prompt']
            ground_truth = example['ground_truth']
            video_paths = example['video_paths']
            
            sample = {
                "video": video_paths,
                "conversations": [
                    {"from": "human", "value": prompt_text},
                    {"from": "gpt", "value": ground_truth}
                ]
            }
            
            conversations = preprocess_multimodal(sample, mm_use_im_start_end=False)
            
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], conversations[0]["value"])
            conv.append_message(conv.roles[1], ground_truth)
            prompt = conv.get_prompt()
            
            input_ids = tokenizer_image_token(
                prompt, tokenizer, return_tensors='pt'
            ).unsqueeze(0).to(device)
            
            labels = input_ids.clone()
            
            prompt_without_gt = conv_templates[conv_mode].copy()
            prompt_without_gt.append_message(conv.roles[0], conversations[0]["value"])
            prompt_without_gt.append_message(conv.roles[1], None)
            prompt_without_gt_text = prompt_without_gt.get_prompt()
            
            prompt_without_gt_ids = tokenizer_image_token(
                prompt_without_gt_text, tokenizer, return_tensors='pt'
            ).unsqueeze(0).to(device)
            
            labels[:, :prompt_without_gt_ids.shape[1]] = -100
            
            del prompt_without_gt, prompt_without_gt_text, prompt_without_gt_ids

            video_data = prepare_video_input(
                video_paths, video_folder, video_processor, device, dtype
            )
            
            forward_kwargs = {
                "input_ids": input_ids,
                "labels": labels,
            }
            
            if video_data['images'] is not None:
                images = [img.to(device=device, dtype=dtype) for img in video_data['images']]
                forward_kwargs["images"] = [images]
            
            if video_data['world_points'] is not None:
                forward_kwargs["world_points"] = video_data['world_points'].unsqueeze(0)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(**forward_kwargs)
                loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
            optimizer.step()
            
            epoch_loss += loss.item()
            print(f"  Example {idx+1}/{len(training_examples)} - Loss: {loss.item():.4f}")
            
            del input_ids, labels, outputs, loss, forward_kwargs
            if video_data['images'] is not None:
                del images, video_data['images']
            if video_data['world_points'] is not None:
                del video_data['world_points']
            del video_data, sample, conversations, conv, prompt

            torch.cuda.empty_cache()
        
        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(training_examples):.4f}")

    model.eval()
    print(f"Training Complete!")
    print(f"{'='*70}\n")

    torch.cuda.empty_cache()

    del optimizer
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    import gc
    gc.collect()
    
    torch.cuda.empty_cache()

    return model

def perform_rl_training(
    model,
    tokenizer,
    video_processor,
    retro_buffer: Dict,
    current_video_paths: List[str],
    task: str,
    total_rooms: List[str],
    explored_rooms: List[str],
    current_room: str,
    previous_action: str,
    previous_reflection: str,
    device: torch.device,
    dtype: torch.dtype,
    video_folder: str,
    conv_mode: str,
    room_visit_count: Dict,
    num_epochs: int = 3,
    learning_rate: float = 5e-5
):
    """
    Perform REINFORCE RL training on action model using retro-reflection scores.
    Uses current observations and state (at EXIT time) like internal reflection training.
    """
    print(f"\n{'='*70}")
    print(f"STARTING RL TRAINING (REINFORCE)")
    print(f"{'='*70}")
    
    goto_actions = []
    print (retro_buffer)
    for action_idx, retro_data in retro_buffer.items():
        action_info = retro_data.get('action_info', {})
        parsed_action = action_info.get('parsed_action', {})

        print (parsed_action.get('action_type'))

        if parsed_action.get('action_type') == 'GO_TO':
            retro_reflection = retro_data.get('reflection', {})
            score = retro_data.get('score', {})
            normalized_reward = (int(score) / 50.0) - 1.0

            goto_actions.append({
                'action_idx': action_idx,
                'action_info': action_info,
                'reward': normalized_reward,
                'score': score
            })
    
    positive_examples = [action for action in goto_actions if int(action["score"]) > 50]
    positive_examples_copy = copy.deepcopy(positive_examples)
    negative_examples = [action for action in goto_actions if int(action["score"]) <= 50]
    if len(negative_examples) - len(positive_examples) > 1:
        for k in range(int((len(negative_examples) - len(positive_examples)) / len(positive_examples))-1):
            positive_examples.extend(positive_examples_copy)
    goto_actions = positive_examples + negative_examples
    random.shuffle(goto_actions)

    if not goto_actions:
        print("No GO TO actions found in retro buffer. Skipping RL training.")
        return model
    
    print(f"Found {len(goto_actions)} GO TO actions for RL training")
    for action_data in goto_actions:
        print(f"  Action {action_data['action_idx']}: Score={action_data['score']}, Reward={action_data['reward']:.3f}")
    
    torch.cuda.empty_cache()
    
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
    print("Model prepared for k-bit training")
    
    model.train()
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        inference_mode=False,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        epoch_loss = 0
        
        for idx, action_data in enumerate(goto_actions):
            torch.cuda.empty_cache()
            
            action_info = action_data['action_info']
            reward = action_data['reward']
            action_text = action_info.get('action_text', '')
            
            sample = build_iterative_prompt(
                task=task,
                total_rooms=total_rooms,
                explored_rooms=explored_rooms,
                current_room=current_room,
                previous_action=previous_action,
                previous_reflection=previous_reflection,
                video_paths=current_video_paths,
                room_visit_count=room_visit_count
            )

            sample['conversations'].append({
                "from": "gpt",
                "value": action_text
            })

            conversations = preprocess_multimodal(sample, mm_use_im_start_end=False)
            
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], conversations[0]["value"])
            conv.append_message(conv.roles[1], action_text)
            prompt = conv.get_prompt()
            
            input_ids = tokenizer_image_token(
                prompt, tokenizer, return_tensors='pt'
            ).unsqueeze(0).to(device)
            
            labels = input_ids.clone()
            
            prompt_without_action = conv_templates[conv_mode].copy()
            prompt_without_action.append_message(conv.roles[0], conversations[0]["value"])
            prompt_without_action.append_message(conv.roles[1], None)
            prompt_without_action_text = prompt_without_action.get_prompt()
            
            prompt_without_action_ids = tokenizer_image_token(
                prompt_without_action_text, tokenizer, return_tensors='pt'
            ).unsqueeze(0).to(device)
            
            labels[:, :prompt_without_action_ids.shape[1]] = -100
            
            del prompt_without_action, prompt_without_action_text, prompt_without_action_ids
            
            video_data = prepare_video_input(
                current_video_paths,
                video_folder, video_processor, device, dtype
            )
            
            forward_kwargs = {
                "input_ids": input_ids,
                "labels": labels,
            }
            
            if video_data['images'] is not None:
                images = [img.to(device=device, dtype=dtype) for img in video_data['images']]
                forward_kwargs["images"] = [images]
            
            if video_data['world_points'] is not None:
                forward_kwargs["world_points"] = video_data['world_points'].unsqueeze(0)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(**forward_kwargs)
                loss = outputs.loss
                
                rl_loss = loss * reward
            
            optimizer.zero_grad()
            rl_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            epoch_loss += rl_loss.item()
            print(f"  Action {action_data['action_idx']} (Reward: {reward:.3f}) - RL Loss: {rl_loss.item():.4f}")
            
            del input_ids, labels, outputs
            del rl_loss, forward_kwargs
            if video_data['images'] is not None:
                del images, video_data['images']
            if video_data['world_points'] is not None:
                del video_data['world_points']
            del video_data, sample, conversations, conv, prompt
            
            torch.cuda.empty_cache()
        
        print(f"Epoch {epoch+1} Avg RL Loss: {epoch_loss/len(goto_actions):.4f}")
    
    model.eval()
    print(f"RL Training Complete!")
    print(f"{'='*70}\n")
    
    torch.cuda.empty_cache()
    
    del optimizer
    
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    import gc
    gc.collect()
    
    torch.cuda.empty_cache()
    
    return model

def test_internal_reflection(
    model,
    tokenizer,
    video_processor,
    test_action: str,
    test_label: str,
    task: str,
    total_rooms: List[str],
    explored_rooms: List[str],
    current_room: str,
    previous_action: str,
    previous_reflection: str,
    video_paths: List[str],
    video_folder: str,
    conv_mode: str,
    device: torch.device,
    dtype: torch.dtype,
    room_visit_count
) -> Dict:
    """
    Test the model's internal reflection generation for a given action.
    Returns output in INTERNAL REFLECTION format.
    """
    torch.cuda.empty_cache()

    internal_sample = build_internal_reflection_prompt(
        task=task,
        total_rooms=total_rooms,
        explored_rooms=explored_rooms,
        current_room=current_room,
        previous_action=previous_action,
        previous_reflection=previous_reflection,
        potential_action=test_action,
        video_paths=video_paths,
        room_visit_count=room_visit_count
    )
    
    conversations = preprocess_multimodal(internal_sample, mm_use_im_start_end=False)
    prompt = build_prompt(conversations, conv_mode)
    
    video_data = prepare_video_input(
        video_paths, video_folder, video_processor, device, dtype
    )
    
    output = run_inference(
        model, tokenizer, prompt,
        images=video_data['images'],
        world_points=video_data['world_points'],
        max_new_tokens=512,
        temperature=0.1,
        do_sample=False,
        device=device,
        use_quantization=False
    )
    
    reflection, score = parse_reflection_and_score(output)
    
    formatted_output = f"Internal Reflection: {reflection}; Score: {score}"

    result =  {
        'action': test_action,
        'label': test_label,
        'prompt': internal_sample['conversations'][0]['value'],
        'generated_reflection': reflection,
        'generated_score': score,
        'full_output': output,
        'formatted_internal_reflection': formatted_output
    }

    del internal_sample, conversations, prompt, output, reflection, score, formatted_output
    
    if video_data['images'] is not None:
        del video_data['images']
    if video_data['world_points'] is not None:
        del video_data['world_points']
    del video_data
    
    import gc
    gc.collect()
    
    torch.cuda.empty_cache()

    return result

class BehaviorExecutor:
    """Executes actions in the BEHAVIOR/OmniGibson environment."""
    
    def __init__(self, scene_name: str, total_rooms: List[str], scene_dir_name: str,
                 robot_type: str = "R1",
                 room_objects_file: str = None, scene_config_file: str = None,
                 object_inventory_file: str = None):
        self.scene_name = scene_name
        self.total_rooms = total_rooms
        self.scene_dir_name = scene_dir_name
        self.robot_type = robot_type
        
        self.env = None
        self.robot = None
        self.scene = None
        self.grasped_object = None
        self.current_room = None
        self.base_pose = None
        
        self.interaction_step = 0
        self.object_registry = {}
        self.room_visit_count = {}
        
        self.action_history = []
        self.room_action_history = {}
        self.current_room_actions = {}
        
        self.retro_buffer = {}
        self.retro_reflected_actions = set()
        
        self.room_objects = {}
        if room_objects_file and os.path.exists(room_objects_file):
            with open(room_objects_file, 'r') as f:
                data = json.load(f)
                self.room_objects = data.get('scenes', {})
        
        self.scene_config = None
        self.new_objects = []
        self.scene_desp = {}

        if scene_config_file and os.path.exists(scene_config_file):
            with open(scene_config_file, 'r') as f:
                self.scene_config = json.load(f)
            if 'scene_name' in self.scene_config:
                config_scene = self.scene_config['scene_name']
                if config_scene.endswith('_scene_dict.json'):
                    config_scene = config_scene.replace('_scene_dict.json', '')
                self.scene_name = config_scene
            self.scene_desp = self.scene_config.get('scene_desp', {})
            parsed_data = self.scene_config.get('parsed_data', {})
            self.new_objects = parsed_data.get('new objects', [])
            print(f"Loaded scene config: {len(self.new_objects)} new objects to place")
        
        self.object_inventory = {}
        if object_inventory_file and os.path.exists(object_inventory_file):
            with open(object_inventory_file, 'r') as f:
                inventory_data = json.load(f)
                self.object_inventory = inventory_data.get('providers', {})
            print(f"Loaded object inventory: {len(self.object_inventory)} objects")
        
        self.external_camera_poses = [
            [[0.6, 0.6, 2.0], [0.2706, -0.2706, -0.6533, 0.6533]],
            [[0.2, -0.6, 2.0], [-0.1930, 0.4163, 0.8062, -0.3734]],
            [[0.2, 0.6, 2.0], [0.4164, -0.1929, -0.3737, 0.8060]],
        ]
        
        self.pointcloud_scene_dir = os.path.join(POINTCLOUD_DIR, self.scene_dir_name)
        os.makedirs(self.pointcloud_scene_dir, exist_ok=True)
        print(f"Observations will be saved to: {self.pointcloud_scene_dir}")
    
    def add_action_to_history(self, action: str, room: str, reflection: str = "", score: str = "",
                              prompt: str = "", video_paths: List[str] = None):
        """Add an action to the history tracking."""

        action_idx = len(self.action_history)

        self.action_history.append({
            'action': action,
            'room': room,
            'reflection': reflection,
            'score': score,
            'action_idx': action_idx,
            'prompt': prompt,
            'video_paths': video_paths or []
        })
        

        # Track actions in current room
        # if not "exit" in action.lower():
        if "pick " in action.lower():
            self.current_room_actions[action] = {
                'reflection': reflection,
                'score': score
            }
        
        # Add to retro buffer if it's a GO TO or PICK UP action
        # Only keep the most recent reflection for each unique action text
        # if action.upper().startswith('GO TO ') or action.upper().startswith('PUT '):
        if action.upper().startswith('GO TO ') or action.upper().startswith('PICK '):
            self.retro_buffer[action] = {
                'action': action,
                'room': room,
                'reflection': reflection,
                'score': score,
                'action_info': {
                    'parsed_action': {'action_type': 'GO_TO' if action.upper().startswith('GO TO') else 'PICK_UP'},
                    'action_text': action,
                    'prompt': prompt,
                    'video_paths': video_paths or []
                }
            }
        
        return action_idx

    def update_action_reflection(self, action_text, reflection: str, score: str):
        """Update the reflection and score for an action."""
        if action_text in self.retro_buffer:
            self.retro_buffer[action_text]['reflection'] = reflection
            self.retro_buffer[action_text]['score'] = score
    
    def clear_current_room_actions(self):
        """Clear the current room actions when exiting a room."""
        self.current_room_actions = {}
    
    def get_retro_buffer_actions(self) -> List[Dict]:
        """Get all actions in the retro buffer (all GO TO and PICK UP actions)."""
        return [self.retro_buffer[idx] for idx in sorted(self.retro_buffer.keys())]
    
    def depth_to_pcd(self, depth: th.Tensor, rel_pose: th.Tensor, K: th.Tensor) -> th.Tensor:
        """Convert depth images to point clouds with batch processing support."""
        original_shape = depth.shape
        depth = depth.view(-1, original_shape[-2], original_shape[-1])
        rel_pose = rel_pose.view(-1, 7)
        B, H, W = depth.shape
        device = depth.device

        rel_pos = rel_pose[:, :3]
        rel_quat = rel_pose[:, 3:]
        rel_rot = T.quat2mat(rel_quat)

        rot_add = T.euler2mat(th.tensor([np.pi, 0, 0], device=device))
        rel_rot_matrix = th.matmul(rel_rot, rot_add)

        camera_to_base_tf = th.eye(4, device=device).unsqueeze(0).expand(B, 4, 4).clone()
        camera_to_base_tf[:, :3, :3] = rel_rot_matrix
        camera_to_base_tf[:, :3, 3] = rel_pos

        y, x = th.meshgrid(th.arange(H, device=device), th.arange(W, device=device), indexing="ij")
        u = x.unsqueeze(0).expand(B, H, W)
        v = y.unsqueeze(0).expand(B, H, W)
        uv = th.stack([u, v, th.ones_like(u)], dim=-1).float()

        Kinv = th.linalg.inv(K).to(device)

        pc_camera = depth.unsqueeze(-1).float() * th.matmul(uv.float(), Kinv.float().transpose(-2, -1))

        pc_camera_homo = th.cat([pc_camera, th.ones_like(pc_camera[..., :1])], dim=-1)

        pc_camera_homo_flat = pc_camera_homo.view(B, -1, 4)
        pc_base = th.matmul(pc_camera_homo_flat, camera_to_base_tf.transpose(-2, -1))
        pc_base = pc_base[..., :3].view(*original_shape, 3)

        return pc_base

    def capture_observation(self, room_name: str, step_idx: int) -> str:
        """
        Capture observation and return the path for use in next iteration.
        Returns: relative path to observation directory
        """
        print(f"\n=== Capturing Observation: {room_name} (step {step_idx}) ===")
        
        obs_dir_name = f"{room_name}_{step_idx}"
        obs_dir = os.path.join(self.pointcloud_scene_dir, obs_dir_name)
        os.makedirs(obs_dir, exist_ok=True)
        
        all_pcds = []
        all_rgbs = []
        
        floor = self.get_floor_for_room(room_name)
        if not floor:
            print(f"  Could not find floor for exploration, using current position only")
            num_attempts = 1
        else:
            num_attempts = MAX_EXPLORATION_ATTEMPTS
        
        for attempt in range(num_attempts):
            try:
                if floor and attempt > 0:
                    if sample_kinematics("onTop", self.robot, floor):
                        og.sim.step()
                        time.sleep(0.1)
                
                for _ in range(5):
                    og.sim.step()
                    og.sim.render()
                
                for sensor_idx in range(3):
                    sensor = list(self.env._external_sensors.values())[sensor_idx]
                    obs = sensor.get_obs()[0]
                    
                    depth = obs['depth_linear']
                    rgb = obs['rgb']
                    all_rgbs.append(rgb)
                    
                    camera_params = sensor.camera_parameters
                    direct_cam_pose = camera_params["cameraViewTransform"]
                    position, orientation = T.mat2pose(
                        th.tensor(np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T), dtype=th.float32)
                    )
                    pose = th.cat([position, orientation])
                    
                    pcd = self.depth_to_pcd(depth, pose, sensor.intrinsic_matrix)
                    all_pcds.append(pcd)
            except:
                pass
        
        if all_pcds:
            stacked_pcds = th.stack(all_pcds, dim=0)
            print(f"  Stacked point cloud shape: {stacked_pcds.shape}")
            
            stacked_pcds_np = stacked_pcds.cpu().numpy() if isinstance(stacked_pcds, th.Tensor) else stacked_pcds
            pcd_filename = os.path.join(obs_dir, "points.npy")
            np.save(pcd_filename, stacked_pcds_np)
            print(f"  Saved point clouds to {pcd_filename}")
            
            for img_idx, rgb in enumerate(all_rgbs):
                if isinstance(rgb, th.Tensor):
                    rgb_np = rgb.cpu().numpy()
                else:
                    rgb_np = rgb
                
                if rgb_np.max() <= 1.0:
                    rgb_uint8 = (rgb_np * 255).astype(np.uint8)
                else:
                    rgb_uint8 = rgb_np.astype(np.uint8)
                
                img = Image.fromarray(rgb_uint8)
                img_filename = os.path.join(obs_dir, f"{img_idx}.png")
                img.save(img_filename)
            
            print(f"  Saved {len(all_rgbs)} RGB images")
        else:
            print(f"  No observation data captured")
        
        scene_dir = f"{self.scene_name}_scene_dict.json_1_1"
        return f"{POINTCLOUD_DIR}/{scene_dir}/{obs_dir_name}"
    
    def get_object_instances_by_category(self, category: str) -> List[str]:
        instances = [obj_id for obj_id in self.object_inventory.keys() 
                     if obj_id.startswith(f"{category}-")]
        return instances
    
    def select_random_object_instance(self, category: str) -> Optional[str]:
        instances = self.get_object_instances_by_category(category)
        if not instances:
            print(f"Warning: No instances found for category '{category}'")
            return None
        return random.choice(instances)
    
    def parse_location(self, location: List[str]) -> tuple:
        room_name = None
        target_object = None
        
        for loc in location:
            if loc.startswith('in-'):
                name = loc[3:]
                if name in self.scene_desp:
                    room_name = name
                else:
                    target_object = name
            elif loc.startswith('on-'):
                target_object = loc[3:]
        
        return room_name, target_object
    
    def prepare_new_objects(self):
        print("\n=== Preparing New Objects ===")
        
        for idx, obj_spec in enumerate(self.new_objects):
            category = obj_spec['category']
            location = obj_spec['location']
            
            instance_id = self.select_random_object_instance(category)
            if not instance_id:
                print(f"Skipping object {idx} ({category}) - no instances found")
                continue
            
            model_id = instance_id.split('-')[1]
            room_name, target_object_name = self.parse_location(location)
            
            print(f"Preparing {category} (model: {model_id}) for {room_name} on {target_object_name}")
            
            obj_name = f"task_{category}_{idx}"
            uniform_scale = 1.0
            
            obj_config = {
                "type": "DatasetObject",
                "name": obj_name,
                "category": category,
                "model": model_id,
                "position": [100.0, 100.0, 100.0],
                "orientation": [0, 0, 0, 1],
                "scale": uniform_scale,
            }
            
            self.object_registry[obj_name] = {
                'config': obj_config,
                'room': room_name,
                'target': target_object_name,
                'category': category
            }
        
        print(f"Prepared {len(self.object_registry)} objects")
    
    def place_new_objects(self):
        print("\n=== Placing New Objects ===")
        
        for obj_name, obj_data in self.object_registry.items():
            try:
                obj = self.scene.object_registry("name", obj_name)
                if obj is None:
                    print(f"Could not find object {obj_name} in scene")
                    continue
                
                for link in obj.links.values():
                    link.mass = 0.1
                
                target_name = obj_data['target']
                room_name = obj_data['room']
                
                target_obj = self.get_room_object(target_name, room_name)
                if target_obj is None:
                    print(f"Could not find target {target_name} in {room_name}")
                    continue
                
                print(f"Placing {obj_name} on {target_name} in {room_name}")
                
                success = False
                for attempt in range(3):
                    try:
                        success = sample_kinematics("onTop", obj, target_obj, 
                                                   use_last_ditch_effort=True, use_trav_map=False)
                        if success:
                            print(f"Successfully placed {obj_name}")
                            break
                    except Exception as e:
                        print(f"  Attempt {attempt + 1} failed: {e}")
                
                if not success:
                    print(f"Failed to place {obj_name}")
                    
            except Exception as e:
                print(f"Error initializing object {obj_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    def initialize(self):
        print(f"\n=== Initializing BEHAVIOR Environment ===")
        print(f"Scene: {self.scene_name}")
        
        if self.new_objects:
            self.prepare_new_objects()
        
        config_filename = os.path.join(og.example_config_path, f"{self.robot_type.lower()}_primitives.yaml")
        config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
        
        config["scene"]["scene_model"] = self.scene_name
        config["scene"]["load_room_instances"] = self.total_rooms
        config["scene"]["not_load_object_categories"] = ["ceilings", "carpet"]
        config["scene"]["use_floor_plane"] = False
        
        if "name" not in config["robots"][0]:
            config["robots"][0]["name"] = "demo"
        robot_name = config["robots"][0]["name"]
        
        config["robots"][0]["exclude_sensor_names"] = ['left_eef_link', 'right_eef_link', 'eyes']
        
        external_sensors_config = []
        for i, (position, orientation) in enumerate(self.external_camera_poses):
            external_sensors_config.append({
                "sensor_type": "VisionSensor",
                "name": f"external_sensor{i}",
                "relative_prim_path": f"/controllable__r1__{robot_name}/base_link/external_sensors{i}",
                "modalities": ["rgb", "depth_linear"],
                "sensor_kwargs": {
                    "image_height": IMG_WIDTH,
                    "image_width": IMG_HEIGHT,
                    "horizontal_aperture": 40.0,
                },
                "position": th.tensor(position, dtype=th.float32),
                "orientation": th.tensor(orientation, dtype=th.float32),
                "pose_frame": "parent"
            })
        
        config["env"]["external_sensors"] = external_sensors_config
        config["env"]["use_external_obs"] = True
        
        if self.object_registry:
            config["objects"] = [obj_data['config'] for obj_data in self.object_registry.values()]
            print(f"Added {len(config['objects'])} new objects to scene config")
        
        self.env = og.Environment(configs=config)
        self.scene = self.env.scene
        self.robot = self.env.robots[0]

        self.base_pose = self.robot.get_position_orientation()
        
        print("Environment initialized")
        
        if self.object_registry:
            self.place_new_objects()
        
        for _ in range(10):
            og.sim.step()
            og.sim.render()
    
    def get_room_object(self, object_name: str, room_name: str):
        if self.scene_name in self.room_objects:
            if room_name in self.room_objects[self.scene_name]:
                room_objs = self.room_objects[self.scene_name][room_name]
                target_objs = [obj for obj in room_objs if obj.startswith(f'{object_name}-')]
                if target_objs:
                    obj_name = target_objs[0].replace('-', '_') + '_0'
                    obj = self.scene.object_registry("name", obj_name)
                    if obj:
                        return obj
        return None
    
    def get_floor_for_room(self, room_name: str):
        floor = self.get_room_object("floors", room_name)
        if floor is not None:
            return floor
        
        floor_name = f"floors_{room_name}_0"
        floor = self.scene.object_registry("name", floor_name)
        if floor is not None:
            return floor
        
        if '_' in room_name:
            room_base = room_name.rsplit('_', 1)[0]
            room_idx = room_name.rsplit('_', 1)[1]
            floor_name = f"floors_{room_base}_{room_idx}_0"
            floor = self.scene.object_registry("name", floor_name)
            if floor is not None:
                return floor
        
        for obj in self.scene.objects:
            obj_name = obj.name.lower()
            if 'floor' in obj_name and room_name.lower().replace('_', '') in obj_name.replace('_', ''):
                return obj
        
        for obj in self.scene.objects:
            if 'floor' in obj.name.lower():
                return obj
        
        return None
    
    def find_object(self, obj_name: str, specified_room: str = None):
        """Find an object in the scene by name with STRICT room verification."""

        candidates = []
        for task_obj_name in self.object_registry:
            obj_data = self.object_registry[task_obj_name]
            
            if obj_name in task_obj_name or obj_data['category'] == obj_name:
                if specified_room:
                    if obj_data['room'] == specified_room:
                        candidates.append(task_obj_name)
        
        if candidates:
            obj_name_to_get = candidates[0]
            obj = self.scene.object_registry("name", obj_name_to_get)
            return obj
        
        return None
    
    def find_object2(self, obj_name: str, specified_room: str = None):
        """Find an object in the scene by name with STRICT room verification."""

        candidates = []
        for task_obj_name in self.object_registry:
            obj_data = self.object_registry[task_obj_name]

            if specified_room:
                if obj_data['room'] == specified_room:
                    candidates.append(task_obj_name)
        
        if candidates:
            obj_name_to_get = candidates[0]
            obj = self.scene.object_registry("name", obj_name_to_get)
            return obj
        
        return None

    def parse_action(self, action_text: str) -> Dict:
        action_text = action_text.strip()
        
        result = {
            "action_type": "UNKNOWN",
            "target": None,
            "room": None,
            "raw": action_text
        }
        
        if action_text.upper().startswith('GO TO '):
            result["action_type"] = "GO_TO"
            result["target"] = action_text[6:].strip()
        
        elif action_text.upper().startswith('PICK UP '):
            result["action_type"] = "PICK_UP"
            parts = action_text[8:].strip()
            if ' in ' in parts:
                obj_part, room = parts.rsplit(' in ', 1)
                result["target"] = obj_part.strip()
                result["room"] = room.strip()
            else:
                result["target"] = parts
        
        elif action_text.upper().startswith('PUT DOWN '):
            result["action_type"] = "PUT_DOWN"
            parts = action_text[9:].strip()
            if ' on ' in parts:
                obj_part, rest = parts.split(' on ', 1)
                result["target"] = obj_part.strip()
                if ' in ' in rest:
                    surface, room = rest.rsplit(' in ', 1)
                    result["room"] = room.strip()
        
        elif action_text.lower().startswith('exit '):
            result["action_type"] = "EXIT"
            result["target"] = action_text[5:].strip()
        
        elif action_text.lower() == 'done':
            result["action_type"] = "DONE"
        
        return result
    
    def execute_go_to(self, room_name: str) -> tuple:
        print(f"  Executing: GO TO {room_name}")
        
        if room_name not in self.room_visit_count:
            self.room_visit_count[room_name] = 0
        self.room_visit_count[room_name] += 1
        
        current_visit_number = self.room_visit_count[room_name]
        print(f"  This is visit #{current_visit_number} to {room_name}")

        floor = self.get_floor_for_room(room_name)
        if not floor:
            print(f"  Could not find floor for {room_name}")
            return False, f"Could not find floor for {room_name}"
        
        print(f"  Found floor: {floor.name}")
        
        for attempt in range(MAX_NAVIGATION_ATTEMPTS):
            if sample_kinematics("onTop", self.robot, floor):
                og.sim.step()
                self.current_room = room_name
                print(f"  Navigated to {room_name}")
                
                obs_path = self.capture_observation(room_name, self.interaction_step)
                self.interaction_step += 1
                
                return True, f"Successfully navigated to {room_name}", obs_path
        
        print(f"  Failed to navigate to {room_name} after {MAX_NAVIGATION_ATTEMPTS} attempts")
        return False, f"Failed to navigate to {room_name}", None
    
    def execute_pick_up(self, object_name: str, room) -> tuple:
        global AG_JOINT_PRIM
        
        print(f"  Executing: PICK UP {object_name}")
        
        if AG_JOINT_PRIM is not None:
            return False, "Robot already holding an object", None
        
        if self.room_visit_count[room] > 1:
            obj = self.find_object2(object_name, room)
        else:
            obj = self.find_object(object_name, room)
            if not obj:
                return False, f"Could not find object: {object_name}", None
        
        grasp_point = self.robot.get_eef_position(self.robot.default_arm)
        obj.visual_only = True
        obj.set_position_orientation(grasp_point, [0.0, 0.0, 0.0, 1.0])
        obj.keep_still()
        og.sim.step()
        
        joint_prim_path = f"{self.robot.eef_links[self.robot.default_arm].prim_path}/ag_constraint"
        AG_JOINT_PRIM = create_joint(
            prim_path=joint_prim_path,
            joint_type="FixedJoint",
            body0=self.robot.eef_links[self.robot.default_arm].prim_path,
            body1=obj.root_link.prim_path,
            enabled=True,
            exclude_from_articulation=True,
        )
        
        self.grasped_object = obj
        
        obs_path = None

        if self.current_room:
            obs_path = self.capture_observation(self.current_room, self.interaction_step)
            self.interaction_step += 1
        
        return True, f"Successfully picked up {obj.name}", obs_path
    
    def execute_put_down(self, object_name: str, room_name: str = None) -> tuple:
        global AG_JOINT_PRIM
        
        print(f"  Executing: PUT DOWN {object_name}")
        
        if AG_JOINT_PRIM is None or self.grasped_object is None:
            return False, "Robot not holding any object", None
        
        obj = self.grasped_object
        
        delete_or_deactivate_prim(str(AG_JOINT_PRIM.GetPrimPath()))
        AG_JOINT_PRIM = None
        
        floor = self.get_floor_for_room(room_name or self.current_room)
        if floor:
            obj.set_position_orientation([100.0, 100.0, 100.0], [0.0, 0.0, 0.0, 1.0])
            obj.visual_only = False
            obj.keep_still()
            
            for attempt in range(3):
                try:
                    success = sample_kinematics("onTop", obj, floor,
                                               use_last_ditch_effort=True, use_trav_map=False)
                    if success:
                        break
                except:
                    pass
        else:
            robot_pos, _ = self.robot.get_position_orientation()
            obj.set_position_orientation(
                [robot_pos[0] + 0.5, robot_pos[1], robot_pos[2]],
                [0.0, 0.0, 0.0, 1.0]
            )
            obj.visual_only = False
        
        self.grasped_object = None
        
        obs_path = None
        if self.current_room:
            obs_path = self.capture_observation(self.current_room, self.interaction_step)
            self.interaction_step += 1
        
        return True, f"Put down {obj.name}", obs_path
    
    def execute_exit(self, room_name: str) -> tuple:
        print(f"  Executing: Exit {room_name}")
        return True, f"Exiting {room_name}", None
    
    def execute_action(self, parsed_action: Dict) -> tuple:
        action_type = parsed_action["action_type"]
        target = parsed_action["target"]
        room = parsed_action.get("room")
        
        if action_type == "GO_TO":
            return self.execute_go_to(target)
        elif action_type == "PICK_UP":
            return self.execute_pick_up(target, room)
        elif action_type == "PUT_DOWN":
            return self.execute_put_down(target, room)
        elif action_type == "EXIT":
            return self.execute_exit(target)
        elif action_type == "DONE":
            return True, "Task completed", None
        else:
            return False, f"Unknown action: {action_type}", None
    
    def render(self, num_frames: int = 3):
        for _ in range(num_frames):
            og.sim.step()
            og.sim.render()
    
    def cleanup(self):
        if self.env:
            try:
                if hasattr(self.env, '_external_sensors') and self.env._external_sensors:
                    try:
                        for sensor in self.env._external_sensors.values():
                            sensor.remove()
                    except:
                        pass
                
                og.clear()
            except Exception as e:
                print(f"Warning: Cleanup encountered expected errors: {type(e).__name__}")
                pass


def main():
    parser = argparse.ArgumentParser(description="Iterative LLaVA-3D Inference with BEHAVIOR Execution, External Reflection, and Retro-Reflection")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--video-folder", type=str, default="./")
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--scene-config", type=str, required=True,
                        help="Path to scene config JSON file")
    parser.add_argument("--max-iterations", type=int, default=1,
                        help="Maximum number of iterations (default: 1 for single step)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--num-sample-tokens", type=int, default=1152)
    parser.add_argument("--seed", type=int, default=None)
    
    parser.add_argument("--robot-type", type=str, default="R1")
    parser.add_argument("--room-objects", type=str, 
                        default="../BEHAVIOR-1K/bddl3/bddl/generated_data/combined_room_object_list_future.json")
    parser.add_argument("--object-inventory", type=str,
                        default="../BEHAVIOR-1K/bddl3/bddl/generated_data/object_inventory.json")
    
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--max-memory-gb", type=int, default=None)
    parser.add_argument("--reduce-precision", action="store_true")
    
    parser.add_argument("--experiment-type", type=str, 
                    choices=['vanilla', 'full', 'wo-roa', 'wo-ria', 'wo-value', 'wo-action'],
                    default='vanilla',
                    help="Experiment type")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    disable_torch_init()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.reduce_precision:
        dtype = torch.float16
        print("Using FP16 precision")
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    print(f"Loading model from {args.model_path}...")
    
    model_kwargs = {"torch_dtype": dtype}
    
    if args.load_4bit:
        print("Using 4-bit quantization")
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["lm_head", "video_tower", "vision_tower", "prompt_encoder"],
        )
        model_kwargs["device_map"] = "auto"

    elif args.load_8bit:
        print("Using 8-bit quantization")
        model_kwargs["load_in_8bit"] = True
        model_kwargs["llm_int8_skip_modules"] = ["lm_head", "video_tower", "vision_tower", "prompt_encoder"]
        model_kwargs["device_map"] = "auto"
    else:
        if args.max_memory_gb:
            max_memory = {0: f"{args.max_memory_gb}GB"}
            if args.cpu_offload:
                max_memory["cpu"] = "32GB"
            model_kwargs["max_memory"] = max_memory
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = "auto"
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=False,
        padding_side="right"
    )
    
    print(f"Loading BASE MODEL (frozen for actions/external/retro) from {args.model_path}...")
    
    base_model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_path,
        **model_kwargs
    )

    print(f"Loading TTT MODEL (for internal reflection - will be updated) from {args.model_path}...")
    
    ttt_model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_path,
        **model_kwargs
    )

    print(f"Loading ACTION MODEL (for RL training - will be updated) from {args.model_path}...")
    
    action_model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_path,
        **model_kwargs
    )

    for model_name, model in [("BASE", base_model), ("TTT", ttt_model), ("ACTION", action_model)]:
        print(f"Configuring {model_name} model...")
        
        for name, module in model.named_modules():
            if any(k in name for k in [
                "vision_tower",
                "video_tower",
                "prompt_encoder",
                "mm_projector",
            ]):
                module.to(dtype=torch.float16)
        
        for name, param in model.named_parameters():
            if any(k in name for k in [
                "vision_tower",
                "video_tower",
                "prompt_encoder",
                "mm_projector",
            ]):
                param.requires_grad = False
        
        model.eval()
        
        if not (args.load_4bit or args.load_8bit):
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
        
        vision_tower = model.get_vision_tower()
        if vision_tower is not None:
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            vision_tower.to(device=device, dtype=dtype)
        
        video_tower = model.get_video_tower()
        if video_tower is not None:
            if hasattr(video_tower, 'is_loaded') and not video_tower.is_loaded:
                video_tower.load_model()
            video_tower.to(device=device, dtype=dtype)
        
        prompt_encoder = model.get_prompt_encoder() if hasattr(model, 'get_prompt_encoder') else None
        if prompt_encoder is not None:
            prompt_encoder.to(device=device, dtype=dtype)
        
        model.config.num_frames = args.num_frames
        model.config.num_sample_tokens = args.num_sample_tokens
    
    print("\nMoving BASE model to GPU, TTT and ACTION models to CPU...")
    ttt_model = ttt_model.cpu()
    action_model = action_model.cpu()
    torch.cuda.empty_cache()
    print("Model setup complete: BASE on GPU, TTT and ACTION on CPU")
    
    video_processor = None
    video_tower = base_model.get_video_tower()
    if video_tower is not None:
        video_processor = video_tower.video_processor
    
    def move_base_to_gpu():
        """Move base_model to GPU, move ttt_model and action_model to CPU"""
        nonlocal base_model, ttt_model, action_model
        print("Swapping: BASE -> GPU, TTT -> CPU, ACTION -> CPU")
        ttt_model = ttt_model.cpu()
        action_model = action_model.cpu()
        torch.cuda.empty_cache()
        base_model = base_model.to(device)
        torch.cuda.empty_cache()
    
    def move_ttt_to_gpu():
        """Move ttt_model to GPU, move base_model and action_model to CPU"""
        nonlocal base_model, ttt_model, action_model
        print("Swapping: TTT -> GPU, BASE -> CPU, ACTION -> CPU")
        base_model = base_model.cpu()
        action_model = action_model.cpu()
        torch.cuda.empty_cache()
        ttt_model = ttt_model.to(device)
        torch.cuda.empty_cache()
    
    def move_action_to_gpu():
        """Move action_model to GPU, move base_model and ttt_model to CPU"""
        nonlocal base_model, ttt_model, action_model
        print("Swapping: ACTION -> GPU, BASE -> CPU, TTT -> CPU")
        base_model = base_model.cpu()
        ttt_model = ttt_model.cpu()
        torch.cuda.empty_cache()
        action_model = action_model.to(device)
        torch.cuda.empty_cache()
    
    
    print(f"Loading scene config from {args.scene_config}...")
    with open(args.scene_config, 'r') as f:
        scene_config = json.load(f)

    parsed_data = scene_config.get('parsed_data', {})
    task = parsed_data.get('Task', 'Complete the storage task')
    total_rooms = parsed_data.get('Rooms', [])

    new_objects = parsed_data.get('new objects', [])
    target_object = None
    target_room = None

    import re
    preference_pattern = r'(\d+)\)\s*([a-z_ ]+)\s*\((\d+)\)'
    matches = re.findall(preference_pattern, task.lower())

    preferences = {}
    for rank, obj_name, score in matches:
        preferences[obj_name.strip().lower().replace(" ", "_")] = int(rank)

    min_rank = float('inf')
    for obj in new_objects:
        obj_category = obj.get('category', '').replace(' (fillable)', '').strip().lower().replace(" ", "_")
        obj_rank = preferences.get(obj_category, float('inf'))
        if obj_rank < min_rank:
            min_rank = obj_rank
            target_object = obj_category
            target_room = obj['location'][0].replace("in-", "")

    print (target_object, target_room)

    if not target_object:
        os._exit(404)

    print(f"Target object for task completion: {target_object}")

    scene_name = scene_config.get('scene_name', '').replace('_scene_dict.json', '')
    if not scene_name:
        config_basename = os.path.basename(args.scene_config)
        scene_name = config_basename.split('_scene_dict')[0]

    config_basename = os.path.basename(args.scene_config)
    if config_basename.endswith('.json'):
        scene_dir_name = config_basename[:-5] + '_1_1'
    else:
        scene_dir_name = config_basename + '_1_1'

    current_room = total_rooms[0] if total_rooms else "None"

    print(f"\n=== Iterative Inference Setup ===")
    print(f"Task: {task}")
    print(f"Scene: {scene_name}")
    print(f"Total rooms: {total_rooms}")
    print(f"Starting room: {current_room}")

    executor = BehaviorExecutor(
        scene_name=scene_name,
        total_rooms=total_rooms,
        scene_dir_name=scene_dir_name,
        robot_type=args.robot_type,
        room_objects_file=args.room_objects,
        scene_config_file=args.scene_config,
        object_inventory_file=args.object_inventory
    )
    executor.initialize()
    executor.current_room = current_room

    video_paths = []
    explored_rooms = []
    previous_action = "None"
    previous_reflection = ""

    iteration_results = []

    exit_nums = 0
    ttt_comparisons = []

    all_training_examples = []
    retroed_actions = []
    task_completed_now = False

    exit_rooms = []
    tts = True

    for iteration in range(args.max_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{args.max_iterations}")
        print(f"{'='*60}")
        
        video_paths = get_most_recent_observation_for_rooms(
            explored_rooms=explored_rooms,
            scene_dir_name=scene_dir_name,
            pointcloud_dir=POINTCLOUD_DIR
        )

        sample = build_iterative_prompt(
            task=task,
            total_rooms=total_rooms,
            explored_rooms=explored_rooms,
            current_room=current_room,
            previous_action=previous_action,
            previous_reflection=previous_reflection,
            video_paths=video_paths,
            room_visit_count=executor.room_visit_count
        )
        
        print(f"\nCurrent state:")
        print(f"  Room: {current_room}")
        print(f"  Explored: {explored_rooms}")
        print(f"  Video paths: {len(video_paths)}")
        print(f"  Previous action: {previous_action}")
        print(f"  Previous reflection: {previous_reflection}")

        all_rooms_explored = set(exit_rooms) == set(total_rooms)

        if args.experiment_type in ['vanilla', 'wo-ria']:
            num_actions, temp = 1, 0.1
        else:
            num_actions, temp = 4, 2.0

        if all_rooms_explored and tts:
            print(f"\n{'='*60}")
            print(f"Using Action Selection with Internal Reflection")
            print(f"{'='*60}")
            
            move_action_to_gpu()
            
            candidate_actions = generate_multiple_actions_from_model(
                model=action_model,
                tokenizer=tokenizer,
                video_processor=video_processor,
                task=task,
                total_rooms=total_rooms,
                explored_rooms=explored_rooms,
                current_room=current_room,
                previous_action=previous_action,
                previous_reflection=previous_reflection,
                video_paths=video_paths,
                video_folder=args.video_folder,
                conv_mode=args.conv_mode,
                device=device,
                dtype=dtype,
                num_actions=num_actions,
                temperature=temp,
                use_quantization=(args.load_4bit or args.load_8bit),
                room_visit_count=executor.room_visit_count
            )
            
            move_ttt_to_gpu()
            
            output, best_score, all_action_reflections = select_best_action_by_internal_reflection(
                candidate_actions=candidate_actions,
                model=ttt_model,
                tokenizer=tokenizer,
                video_processor=video_processor,
                task=task,
                total_rooms=total_rooms,
                explored_rooms=explored_rooms,
                current_room=current_room,
                previous_action=previous_action,
                previous_reflection=previous_reflection,
                video_paths=video_paths,
                video_folder=args.video_folder,
                conv_mode=args.conv_mode,
                device=device,
                dtype=dtype,
                room_visit_count=executor.room_visit_count
            )
            
            move_base_to_gpu()
            
            internal_reflection_text = next(
                (ar['reflection'] for ar in all_action_reflections if ar['action'] == output),
                ""
            )
            
            print(f"\nSelected Action: {output}")
            print(f"Internal Reflection: {internal_reflection_text}")

            tts = False

        else: 
            print(f"\n--- Generating Action  ---")
            
            move_action_to_gpu()
            
            sample = build_iterative_prompt(
                task=task,
                total_rooms=total_rooms,
                explored_rooms=explored_rooms,
                current_room=current_room,
                previous_action=previous_action,
                previous_reflection=previous_reflection,
                video_paths=video_paths,
                room_visit_count=executor.room_visit_count
            )
            

            conversations = preprocess_multimodal(
                sample,
                mm_use_im_start_end=getattr(action_model.config, 'mm_use_im_start_end', False)
            )
            
            prompt = build_prompt(conversations, args.conv_mode)
            
            images = None
            world_points = None
            
            if video_processor is not None:
                video_data = prepare_video_input(
                    video_paths,
                    args.video_folder,
                    video_processor,
                    device,
                    dtype
                )
                
                images = video_data['images']
                world_points = video_data['world_points']
            
            print("\n--- Generating Action ---")
            max_action_attempts = 5
            current_temperature = args.temperature
            
            for attempt in range(max_action_attempts):
                output = run_inference(
                    action_model,
                    tokenizer,
                    prompt,
                    images=images,
                    world_points=world_points,
                    max_new_tokens=args.max_new_tokens,
                    temperature=current_temperature,
                    device=device,
                    use_quantization=(args.load_4bit or args.load_8bit),
                )
                
                if output.strip().upper() == "DONE":
                    current_temperature += 0.2
                    print(f"  Generated DONE (attempt {attempt+1}/{max_action_attempts})")
                    print(f"  Retrying with temperature={current_temperature:.2f}...")
                else:
                    if attempt > 0:
                        print(f"  Valid action generated after {attempt+1} attempts")
                    break
            else:
                print(f"  Still generated DONE after {max_action_attempts} attempts")
                
                content_object = None
                if previous_action and "PICK" in previous_action.upper():
                    parts = previous_action.split(" in ")[0].strip()
                    parts = parts.replace("PICK UP", "").replace("PICK UP", "").strip()
                    content_object = parts
                
                if content_object:
                    fallback_action = f"PUT DOWN {content_object} on floors in {current_room}"
                    print(f"  Using fallback: {fallback_action}")
                else:
                    fallback_action = f"EXIT {current_room}"
                    print(f"  Using fallback: {fallback_action} (no content object found)")
                
                output = fallback_action
            
            print(f"\n--- Model Output ---")
            print(output)
            
            current_video_paths_for_internal = get_most_recent_observation_for_rooms(
                explored_rooms=explored_rooms,
                scene_dir_name=scene_dir_name,
                pointcloud_dir=POINTCLOUD_DIR
            )

            internal_reflection_text = ""

            print("\n--- Generating Internal Reflection (Pre-Action) ---")
            
            internal_reflection_sample = build_internal_reflection_prompt(
                task=task,
                total_rooms=total_rooms,
                explored_rooms=explored_rooms,
                current_room=current_room,
                previous_action=previous_action,
                previous_reflection=previous_reflection,
                potential_action=output,
                video_paths=current_video_paths_for_internal,
                room_visit_count=executor.room_visit_count
            )
            
            int_conversations = preprocess_multimodal(
                internal_reflection_sample,
                mm_use_im_start_end=getattr(base_model.config, 'mm_use_im_start_end', False)
            )
            
            int_prompt = build_prompt(int_conversations, args.conv_mode)
            
            int_video_data = prepare_video_input(
                current_video_paths_for_internal,
                args.video_folder,
                video_processor,
                device,
                dtype
            )
            
            move_ttt_to_gpu()
            
            internal_reflection_output = run_inference(
                ttt_model,
                tokenizer,
                int_prompt,
                images=int_video_data['images'],
                world_points=int_video_data['world_points'],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device=device,
                use_quantization=(args.load_4bit or args.load_8bit),
            )
            
            move_base_to_gpu()
            
            print(f"\nInternal Reflection: {internal_reflection_output}")
            internal_reflection_text = internal_reflection_output

        parsed_action = executor.parse_action(output)
        print(f"\nParsed action: {parsed_action}")
        
        success, message, obs_path = executor.execute_action(parsed_action)
        print(f"Execution: {'success' if success else 'fail'} {message}")
        
        execution_result = message if not success else "success"
        
        if parsed_action["action_type"] == "GO_TO" and success:
            new_room = parsed_action["target"]
            if new_room in explored_rooms:
                explored_rooms.remove(new_room)
            explored_rooms.append(new_room)
            
            if new_room != current_room:
                executor.clear_current_room_actions()

            current_room = new_room

        current_video_paths = get_most_recent_observation_for_rooms(
            explored_rooms=explored_rooms,
            scene_dir_name=scene_dir_name,
            pointcloud_dir=POINTCLOUD_DIR
        )

        if current_video_paths and (not "exit" in output.lower()):
            print("\n--- Generating External Reflection ---")
            
            external_reflection_sample = build_external_reflection_prompt(
                task=task,
                total_rooms=total_rooms,
                explored_rooms=explored_rooms,
                current_room=current_room,
                previous_action=previous_action,
                previous_reflection=previous_reflection,
                executed_action=output,
                execution_result="success" if success else "fail",
                video_paths=current_video_paths,
                room_visit_count=executor.room_visit_count
            )
            
            ext_conversations = preprocess_multimodal(
                external_reflection_sample,
                mm_use_im_start_end=getattr(base_model.config, 'mm_use_im_start_end', False)
            )
            
            ext_prompt = build_prompt(ext_conversations, args.conv_mode)
            
            ext_video_data = prepare_video_input(
                current_video_paths,
                args.video_folder,
                video_processor,
                device,
                dtype
            )
            
            external_reflection_output = run_inference(
                base_model,
                tokenizer,
                ext_prompt,
                images=ext_video_data['images'],
                world_points=ext_video_data['world_points'],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device=device,
                use_quantization=(args.load_4bit or args.load_8bit),
            )
            
            reflection, score = parse_reflection_and_score(external_reflection_output)
            
            print(f"External Reflection: {reflection}")
            print(f"Score: {score}")
            
            action_idx = executor.add_action_to_history(
                action=output,
                room=current_room,
                reflection=reflection,
                score=score
            )
            
            previous_reflection = reflection
        else:
            print("\nNo observations available for external reflection (or exiting room)")
            previous_reflection = ""
            
            action_idx = executor.add_action_to_history(
                action=output,
                room=current_room,
                reflection="",
                score=""
            )
        
        previous_action2 = output

        if parsed_action["action_type"] == "EXIT" and success:
            exit_rooms.append(current_room)

        if parsed_action["action_type"] == "EXIT" and success and (args.experiment_type not in ['vanilla', 'wo-roa']):            
            retro_reflections = []
            retro_reflection_data = []
            print(f"\n{'='*60}")
            print(f"PERFORMING RETRO-REFLECTION (Exiting {current_room})")
            print(f"{'='*60}")

            buffer_actions = executor.get_retro_buffer_actions()
            
            print(f"\nRetro buffer contains {len(buffer_actions)} GO TO/PICK UP actions")
            print(f"Performing retro-reflection on ALL actions in buffer")
            
            room_actions_reflections = copy.deepcopy(executor.current_room_actions)
            
            print(f"Current room ({current_room}) has {len(room_actions_reflections)} actions for context")
            
            for action_data in buffer_actions:
                action_text = action_data['action']

                if action_text in retroed_actions:
                    continue

                retroed_actions.append(action_text)


                action_room = action_data['room']
                last_reflection = action_data['reflection']
                last_score = action_data['score']
                
                print(f"\n--- Retro-Reflecting on Buffer Action: {action_text} (from {action_room}) ---")
                
                retro_sample = build_retro_reflection_prompt(
                    task=task,
                    total_rooms=total_rooms,
                    explored_rooms=explored_rooms,
                    current_room=current_room,
                    room_actions_reflections=room_actions_reflections,
                    action_to_retro_reflect=action_text,
                    last_reflection=last_reflection,
                    last_score=last_score,
                    video_paths=current_video_paths
                )
                
                retro_conversations = preprocess_multimodal(
                    retro_sample,
                    mm_use_im_start_end=getattr(base_model.config, 'mm_use_im_start_end', False)
                )
                
                retro_prompt = build_prompt(retro_conversations, args.conv_mode)
                
                retro_prompt_text = retro_sample['conversations'][0]['value']
                
                retro_video_data = prepare_video_input(
                    current_video_paths,
                    args.video_folder,
                    video_processor,
                    device,
                    dtype
                )
                
                retro_output = run_inference(
                    base_model,
                    tokenizer,
                    retro_prompt,
                    images=retro_video_data['images'],
                    world_points=retro_video_data['world_points'],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    device=device,
                    use_quantization=(args.load_4bit or args.load_8bit),
                )
                
                retro_reflection, retro_score = parse_reflection_and_score(retro_output)
                
                print(f"Retro Reflection: {retro_reflection}")
                print(f"Retro Score: {retro_score}")
                
                executor.update_action_reflection(action_idx, retro_reflection, retro_score)
                
                retro_reflections.append({
                    'iteration': iteration + 1,
                    'action_idx': action_idx,
                    'action': action_text,
                    'action_room': action_room,
                    'original_reflection': last_reflection,
                    'original_score': last_score,
                    'retro_reflection': retro_reflection,
                    'retro_score': retro_score,
                    'context_room': current_room
                })
                
                retro_reflection_data.append({
                    'iteration': iteration + 1,
                    'action_idx': action_idx,
                    'action': action_text,
                    'action_room': action_room,
                    'context_room': current_room,
                    'room_actions_context': copy.deepcopy(room_actions_reflections),
                    'video_paths': current_video_paths,
                    'prompt': retro_prompt_text,
                    'full_prompt': retro_prompt,
                    'original_reflection': last_reflection,
                    'original_score': last_score,
                    'model_output': retro_output,
                    'parsed_retro_reflection': retro_reflection,
                    'parsed_retro_score': retro_score
                })

                del retro_sample, retro_conversations, retro_prompt, retro_prompt_text
                del retro_output, retro_reflection, retro_score
                
                if retro_video_data['images'] is not None:
                    del retro_video_data['images']
                if retro_video_data['world_points'] is not None:
                    del retro_video_data['world_points']
                del retro_video_data
                
                import gc
                gc.collect()
                
                torch.cuda.empty_cache()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            executor.clear_current_room_actions()
            print(f"\nRetro-reflection complete for {len(buffer_actions)} actions from buffer")

            all_rooms_explored_check = set(exit_rooms) == set(total_rooms)
            
            if args.experiment_type != 'wo-action' and all_rooms_explored_check:
                print(f"\n{'='*70}")
                print(f"RL TRAINING")
                print(f"{'='*70}")
                print(f"Explored rooms: {exit_rooms}")
                print(f"Total rooms: {total_rooms}")
                
                move_action_to_gpu()
                
                action_model = perform_rl_training(
                    model=action_model,
                    tokenizer=tokenizer,
                    video_processor=video_processor,
                    retro_buffer=executor.retro_buffer,
                    current_video_paths=current_video_paths,
                    task=task,
                    total_rooms=total_rooms,
                    explored_rooms=explored_rooms,
                    current_room=current_room,
                    previous_action=previous_action2,
                    previous_reflection=previous_reflection,
                    device=device,
                    dtype=dtype,
                    video_folder=args.video_folder,
                    conv_mode=args.conv_mode,
                    room_visit_count=executor.room_visit_count,
                    num_epochs=3,
                    learning_rate=1e-3
                )
                
                move_base_to_gpu()
                
                print(f"RL training complete - action model updated")

            if len(retro_reflections) > 0:
                print(f"\n{'='*70}")
                print(f"PERFORMING TEST-TIME TRAINING ON RETRO-REFLECTIONS")
                print(f"{'='*70}")
                
                training_examples = []

                for retro_data in retro_reflections:
                    action_idx = retro_data['action_idx']
                    action_text = retro_data['action']
                    action_room = retro_data['action_room']
                    retro_reflection = retro_data['retro_reflection']
                    retro_score = retro_data['retro_score']
                    
                    prev_action = "None"
                    prev_reflection = ""
                    if action_idx > 0:
                        for hist in executor.action_history[:action_idx]:
                            if hist['action_idx'] < action_idx:
                                prev_action = hist['action']
                                prev_reflection = hist.get('reflection', '')
                    
                    if "go to" in action_text.lower():
                        internal_sample = build_internal_reflection_prompt(
                            task=task,
                            total_rooms=total_rooms,
                            explored_rooms=explored_rooms,
                            current_room=current_room,
                            previous_action=previous_action2,
                            previous_reflection=previous_reflection,
                            potential_action=action_text,
                            video_paths=current_video_paths,
                            room_visit_count=executor.room_visit_count
                        )
                    else:
                        internal_sample = build_internal_reflection_prompt(
                            task=task,
                            total_rooms=total_rooms,
                            explored_rooms=explored_rooms,
                            current_room=current_room,
                            previous_action=prev_action,
                            previous_reflection=prev_reflection,
                            potential_action=action_text,
                            video_paths=current_video_paths,
                            room_visit_count=executor.room_visit_count
                        )
                    
                    prompt_text = internal_sample['conversations'][0]['value']
                    
                    ground_truth = f"Internal Reflection: {retro_reflection}; Score: {retro_score}"
                    
                    action_type = 'GO_TO' if action_text.upper().startswith('GO TO') else \
                                'PICK_UP' if action_text.upper().startswith('PICK UP') else 'OTHER'
                    
                    example = {
                        'action': action_text,
                        'action_type': action_type,
                        'prompt': prompt_text,
                        'ground_truth': ground_truth,
                        'video_paths': current_video_paths,
                        'action_idx': action_idx,
                        'training_type': 'retro_reflection'
                    }
                    
                    training_examples.append(example)

                    print(f"Created retro-reflection training example for: {action_text}")
                    print(f"  GT: {ground_truth}")


                training_examples2 = copy.deepcopy(training_examples)
                goto_reflection = dict()

                for (i, example) in enumerate(all_training_examples):
                    if "go to" in all_training_examples[i]['action'].lower():
                        goto_reflection[all_training_examples[i]['action']] = all_training_examples[i]['ground_truth']

                for (i, example) in enumerate(all_training_examples):
                    if "go to" in all_training_examples[i]['action'].lower():
                        internal_sample = build_internal_reflection_prompt(
                            task=task,
                            total_rooms=total_rooms,
                            explored_rooms=explored_rooms,
                            current_room=current_room,
                            previous_action=previous_action2,
                            previous_reflection=previous_reflection,
                            potential_action=all_training_examples[i]['action'].lower(),
                            video_paths=current_video_paths,
                            room_visit_count=executor.room_visit_count
                        )
                    else:
                        internal_sample = build_internal_reflection_prompt(
                            task=task,
                            total_rooms=total_rooms,
                            explored_rooms=explored_rooms,
                            current_room=all_training_examples[i]['action'].split()[-1],
                            previous_action=f"GO TO {all_training_examples[i]['action'].split()[-1]}",
                            previous_reflection=goto_reflection[f"GO TO {all_training_examples[i]['action'].split()[-1]}"],
                            potential_action=all_training_examples[i]['action'].lower(),
                            video_paths=current_video_paths,
                            room_visit_count=executor.room_visit_count
                        )

                    prompt_text = internal_sample['conversations'][0]['value']
                    ground_truth = all_training_examples[i]['ground_truth']

                    example = {
                    'action': all_training_examples[i]['action'],
                    'action_type': all_training_examples[i]['action_type'],
                    'prompt': prompt_text,
                    'ground_truth': all_training_examples[i]['ground_truth'],
                    'video_paths': current_video_paths,
                    'action_idx': action_idx,
                    'training_type': 'retro_reflection'
                    }

                    training_examples.append(example)

                all_training_examples.extend(training_examples2)

                print(f"\nCreated {len(training_examples)} training examples from retro-reflections")

                goto_examples = [ex for ex in training_examples if ex['action_type'] == 'GO_TO']
                pickup_examples = [ex for ex in training_examples if ex['action_type'] == 'PICK_UP']

                available_rooms = [r for r in total_rooms if r not in explored_rooms]

                if len(available_rooms) > 0:
                    random.shuffle(available_rooms)
                    untrained_goto = f"GO TO {available_rooms[0]}" if available_rooms else "GO TO living_room_0"
                    if exit_nums > 1:
                        untrained_pickup = pickup_examples[-1]['action'].replace(pickup_examples[-1]['action'].split(" ")[-1], available_rooms[0])


                print(f"\n{'='*70}")
                print(f"TESTING BEFORE TRAINING (6 actions)")
                print(f"{'='*70}")

                before_results = []

                test_actions = []


                if len(available_rooms):
                    test_actions.append((untrained_goto, "Untrained GO TO (original GT)"))

                    if exit_nums > 1:
                        test_actions.append((untrained_pickup, "Untrained PICK UP (original GT)"))

                move_ttt_to_gpu()

                for test_action, label in test_actions:
                    if test_action:
                        torch.cuda.empty_cache()

                        if "go to" in test_action.lower():
                            result = test_internal_reflection(
                                ttt_model, tokenizer, video_processor,
                                test_action=test_action,
                                test_label=label,
                                task=task,
                                total_rooms=total_rooms,
                                explored_rooms=explored_rooms,
                                current_room=current_room,
                                previous_action=previous_action2,
                                previous_reflection=previous_reflection,
                                video_paths=current_video_paths,
                                video_folder=args.video_folder,
                                conv_mode=args.conv_mode,
                                device=device,
                                dtype=dtype,
                                room_visit_count = executor.room_visit_count
                            )
                        else:
                            result = test_internal_reflection(
                                ttt_model, tokenizer, video_processor,
                                test_action=test_action,
                                test_label=label,
                                task=task,
                                total_rooms=total_rooms,
                                explored_rooms=explored_rooms,
                                current_room=current_room,
                                previous_action=prev_action,
                                previous_reflection=prev_reflection,
                                video_paths=current_video_paths,
                                video_folder=args.video_folder,
                                conv_mode=args.conv_mode,
                                device=device,
                                dtype=dtype,
                                room_visit_count = executor.room_visit_count
                            )
                        before_results.append(result)

                        torch.cuda.empty_cache()

                torch.cuda.empty_cache()

                print(f"\nAdding untrained actions with original outputs as GT (regularization)...")

                for result in before_results:
                    if "original GT" in result['label']:
                        action = result['action']
                        
                        action_type = 'GO_TO' if action.upper().startswith('GO TO') else \
                                    'PICK_UP' if action.upper().startswith('PICK UP') else 'OTHER'
                        
                        internal_sample = build_internal_reflection_prompt(
                            task=task,
                            total_rooms=total_rooms,
                            explored_rooms=explored_rooms,
                            current_room=current_room,
                            previous_action=previous_action2,
                            previous_reflection=previous_reflection,
                            potential_action=action,
                            video_paths=current_video_paths,
                            room_visit_count=executor.room_visit_count
                        )
                        
                        prompt_text = internal_sample['conversations'][0]['value']
                        
                        ground_truth = result['full_output'].strip()
                        
                        untrained_example = {
                            'action': action,
                            'action_type': action_type,
                            'prompt': prompt_text,
                            'ground_truth': ground_truth,
                            'video_paths': current_video_paths,
                            'action_idx': -1,
                            'training_type': 'original_output'
                        }
                        
                        training_examples.append(untrained_example)
                        print(f"  Added untrained action: {action}")
                        print(f"    GT (raw output): {ground_truth}")

                print(f"\nShuffling training examples...")
                random.shuffle(training_examples)

                if args.experiment_type != 'wo-value' and all_rooms_explored_check:
                    ttt_model = perform_test_time_training(
                        model=ttt_model,
                        tokenizer=tokenizer,
                        video_processor=video_processor,
                        training_examples=training_examples,
                        device=device,
                        dtype=dtype,
                        video_folder=args.video_folder,
                        conv_mode=args.conv_mode,
                        num_epochs=5,
                        learning_rate=1e-3
                    )
                
                move_base_to_gpu()

                retro_reflections = []
        
        previous_action = output
        
        iteration_results.append({
            "iteration": iteration + 1,
            "action": output,
            "parsed_action": parsed_action,
            "execution_success": success,
            "execution_message": message,
            "observation_path": obs_path,
            "external_reflection": previous_reflection if not "exit" in output.lower() else "",
            "internal_reflection": internal_reflection_text,
            "current_room": current_room,
            "explored_rooms": explored_rooms.copy()
        })
        
        executor.render(num_frames=5)
        
        all_rooms_explored_check = set(exit_rooms) == set(total_rooms)
        correct_object_picked_check = False
        
        if parsed_action["action_type"] == "PICK_UP":
            action_object = parsed_action.get("target", "").lower()
            if target_object and target_object.replace("_", " ").lower() in action_object.replace("_", " "):
                correct_object_picked_check = True
        
        task_completed_now = all_rooms_explored_check and correct_object_picked_check and (current_room == target_room)
        
        if task_completed_now:
            print(f"\n{'='*60}")
            print(f"TASK COMPLETED AFTER ITERATION {iteration + 1}!")
            print(f"{'='*60}")
            print(f"  All rooms explored: {explored_rooms}")
            print(f"  Correct object picked up: {target_object}")
            print(f"  Stopping early to preserve success!")
            print(f"{'='*60}")
            
            if args.output_file:
                results = {
                    "task": task,
                    "scene": scene_name,
                    "scene_config": args.scene_config,
                    "total_rooms": total_rooms,
                    "iterations": iteration_results,
                    "retro_reflections": retro_reflections if 'retro_reflections' in locals() else [],
                    "action_history": executor.action_history,
                    "retro_buffer": [executor.retro_buffer[key] for key in executor.retro_buffer],
                    "final_explored_rooms": explored_rooms,
                    "total_iterations": len(iteration_results),
                    "all_rooms_explored": True,
                    "correct_object_picked": True,
                    "target_object": target_object,
                    "task_completed": True,
                    "status": "completed_successfully"
                }
                
                with open(args.output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nResults saved to {args.output_file}")
                
                print(f"\nTASK COMPLETED SUCCESSFULLY!")
                print(f"   - All rooms explored: {explored_rooms}")
                print(f"   - Correct object picked: {target_object}")
                print(f"\nUsing os._exit() to avoid OmniGibson segfault during cleanup")
                os._exit(42)
            
            break
        
        if parsed_action["action_type"] == "DONE":
            print("\nAgent generated DONE action")
            break
        
        if iteration == 0 and args.max_iterations == 1:
            print(f"\nFirst iteration complete. Use --max-iterations N for more iterations.")
            break
        
        if all_rooms_explored_check and parsed_action["action_type"] == "PICK_UP":
            print ("early exiting")
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== Cleaning Up ===")
    try:
        executor.cleanup()
        print("Cleanup completed")
    except Exception as e:
        print(f"Cleanup warnings (expected): {type(e).__name__}")

    if args.output_file:
        all_rooms_explored = set(exit_rooms) == set(total_rooms)
        correct_object_picked = False
        
        task_completed = task_completed_now
        
        results = {
            "task": task,
            "scene": scene_name,
            "scene_config": args.scene_config,
            "total_rooms": total_rooms,
            "iterations": iteration_results,
            "action_history": executor.action_history,
            "retro_buffer": [executor.retro_buffer[idx] for idx in sorted(executor.retro_buffer.keys())],
            "final_explored_rooms": explored_rooms,
            "total_iterations": len(iteration_results),
            "all_rooms_explored": all_rooms_explored,
            "correct_object_picked": correct_object_picked,
            "target_object": target_object,
            "task_completed": task_completed,
            "status": "completed_successfully" if task_completed else "completed_unsuccessfully"
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")
        
        if task_completed:
            print(f"\nTASK COMPLETED SUCCESSFULLY!")
            print(f"   - All rooms explored: {all_rooms_explored}")
            print(f"   - Correct object picked: {target_object}")
            os._exit(42)
        else:
            print(f"\nTASK INCOMPLETE")
            print(f"   - All rooms explored: {all_rooms_explored}")
            print(f"   - Correct object picked: {correct_object_picked}")
            os._exit(43)

    print(f"\n{'='*60}")
    print(f"Iterative inference complete: {len(iteration_results)} iteration(s)")
    print(f"Retro-reflections performed: {len(retro_reflections)}")
    print(f"Total actions in history: {len(executor.action_history)}")
    print(f"Actions in retro buffer: {len(executor.retro_buffer)}")
    print(f"Explored rooms: {explored_rooms}")
    if len(iteration_results) == 1:
        print(f"Next observation ready at: {video_paths[-1] if len(video_paths) > 1 else 'N/A'}")
        print(f"Use --max-iterations N to continue for more iterations")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()