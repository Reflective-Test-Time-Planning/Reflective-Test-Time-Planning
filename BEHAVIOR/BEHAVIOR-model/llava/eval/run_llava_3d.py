import argparse
import torch
import json 

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    process_videos,
    tokenizer_special_token,
    get_model_name_from_path,
)

from transformers import StoppingCriteria, StoppingCriteriaList
from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):
    # Model
    disable_torch_init()

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    mode = None

    if args.video_path:
        print(f"Video path provided: {args.video_path}")
        mode = 'video'
    if args.image_file:
        print(f"Image file provided: {args.image_file}")
        mode = 'image'

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, torch_dtype=torch_dtype
    )

    qs = args.query

    matches = re.search(r"\[([^\]]+)\]", qs)
    if matches:
        print('find click matches: \n', matches)
        coord_list = [float(x) for x in matches.group(1).split(',')]
        coord_list = [round(coord, 3) for coord in coord_list[:3]]
        qs = re.sub(r"\[([^\]]+)\]", "<boxes>", qs)
        clicks = torch.tensor([coord_list])
    else:
        clicks = torch.zeros((0,3))

    # image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    # if IMAGE_PLACEHOLDER in qs:
    #     if model.config.mm_use_im_start_end:
    #         qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
    #     else:
    #         qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    # else:
    #     if model.config.mm_use_im_start_end:
    #         qs = image_token_se + "\n" + qs
    #     else:
    #         qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    with open('/home/whu/vl_research/MultiPLY/training_data_test_jan6_test1/00031-Wo6kuutE9i7_031_150.json') as f: #137, 157
        d = json.load(f)
    new_rooms_can_be_explored_str = "Room 1"
    Rooms = ['Room 0<image>', 'Room 3<image>'] #'Room 1<image>']        
    room_str = ', '.join(Rooms)
    
    d['system_message'] = 'Task:' +  d['answer']["Task"] + '\nGoal:' + d['answer']["Goal"] + '\nRequirement:' + d['answer']["Constraint"]
    d['system_message'] +=  "Do not re-explore the rooms you already seen, you can directly Navigate to the object in the room you have seen."
    d['system_message'] += 'Only interact with the object you can see in each room. The id followed after each object indicates starting from nearest from you, if there are multiple same class objects. The new room you can explore to find more objects are' 
    d['system_message'] += new_rooms_can_be_explored_str + "Exsiting Rooms you have explored are: " + room_str

    # task = "Collect and Arrange Candles and Candlesticks"
    # goal = "The task is to collect all candles and candlesticks from the rooms and arrange them on a table in Room 6. The agent needs to explore multiple rooms to gather all items, as they are scattered across different rooms."
    # requirement = "The agent can only hold two items at a time, so it needs to remember which items it has collected and where to find the remaining items."

    # qs = f"Task: {task}\nGoal: {goal}\nRequirement: {requirement} The new room you can explore to find more objects are Room 2, Room 3' \nExsiting Rooms: {Rooms}\n"
    qs = d['system_message']

    # qs += '<NAVIGATE candlestick (0) in room(6)>, <PICK UP candlestick (0)> in room(6), <NAVIGATE candlestick (1) in room(6)>, <PICK UP candlestick (1)> in room(6), <NAVIGATE table (0) in room(6)>, <PUT DOWN candlestick (0) on table (0) in room(6)>, <PUT DOWN candlestick (1) on table (0) in room(6)>, <NAVIGATE candle (0) in room(6)>, <PICK UP candle (0)> in room(6), <NAVIGATE candle (1) in room(6)>, <PICK UP candle (1)> in room(6), <PUT DOWN candle (0) on table (0) in room(6)>, <PUT DOWN candle (1) on table (0) in room(6)>, <EXPLORE room(2)>'
    # qs += 'Room 2<image>'
    # qs += ' <NAVIGATE candlestick (0) in room(2)>, <PICK UP candlestick (0)> in room(2), <NAVIGATE candle (0) in room(6)>, <PICK UP candle (0)> in room(6), <NAVIGATE table (0) in room(6)>, <PUT DOWN candlestick (0) on table (0) in room(6)>, <PUT DOWN candle (0) on table (0) in room(6)>, <EXPLORE room(3)>'
    # qs += 'Room 3<image>'
    
    # task = "Collect Vases"
    # goal = "The goal is to collect a total of 5 vases and place them on a table in Room 7. You start with 3 rooms already explored (Room 5, Room 7, Room 6), and you will need to explore new rooms to complete the task."
    # requirement = 'You can only hold 2 objects at a time.'
    # # Rooms = ['Room 5<image>', 'Room 7<image>', 'Room 6<image>']
    # Rooms = ['Room 6<image>', 'Room 0<image>', 'Room 5<image>']   
    # Rooms = ', '.join(Rooms)
    # qs = f"Task: {task}\nGoal: {goal}\nRequirement: {requirement} Only interact with the object you can see in each room. The id followed after each object indicates starting from nearest from you, if there are multiple same class objects. The new room you can explore to find more objects are Room 2, Room 3' \nExsiting Rooms: {Rooms}\n"

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "3D" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)        
    # conv.append_message(conv.roles[1], "<NAVIGATE soap (0) in room(0)>, <PICK UP soap (0)> in room(0), <NAVIGATE cabinet (0) in room(0)>, <PUT DOWN soap (0) from room(0) on cabinet (0) in room(0)>, <EXPLORE room(1)>") ####################### 
    # conv.append_message(conv.roles[0], "Room 1: <image>")
    # conv.append_message(conv.roles[1], None)  
    prompt = conv.get_prompt()

    # images_tensor = None
    # depths_tensor = None
    # poses_tensor = None
    # intrinsics_tensor = None
    # clicks_tensor = None

    if mode == 'image':
        image_files = image_parser(args)
        images = load_images(image_files)
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            processor['image'],
            model.config
        ).to(model.device, dtype=torch_dtype)
        depths_tensor = None
        poses_tensor = None
        intrinsics_tensor = None
        clicks_tensor = None

    video_paths = json.loads(args.video_path)
    # video_paths = [args.video_path] 
    if mode == 'video':
        print('using video mode')
        images_tensors = []
        depths_tensors = []
        poses_tensors = []
        intrinsics_tensors = []
        clicks_tensors = []
        world_points_tensors = []

        for video_path in video_paths:
            videos_dict = process_videos(
                video_path,
                processor['video'],
                mode='random',
                device=model.device,
                text=args.query
            )
            images_tensor = videos_dict['images'].to(model.device, dtype=torch_dtype)
            world_points = videos_dict['world_points'].to(model.device, dtype=torch_dtype)  
            # depths_tensor = videos_dict['depths'].to(model.device, dtype=torch_dtype)
            # poses_tensor = videos_dict['poses'].to(model.device, dtype=torch_dtype)
            # intrinsics_tensor = videos_dict['intrinsics'].to(model.device, dtype=torch_dtype)
            clicks_tensor = clicks.to(model.device, dtype=torch.bfloat16)
            images_tensors.append(images_tensor)
            # depths_tensors.append(depths_tensor)
            # poses_tensors.append(poses_tensor)
            # intrinsics_tensors.append(intrinsics_tensor)
            clicks_tensors.append(clicks_tensor)
            world_points_tensors.append(world_points)
        
        images_tensor = torch.stack(images_tensors)
        # depths_tensor = torch.stack(depths_tensors)
        # poses_tensor = torch.stack(poses_tensors)
        # intrinsics_tensor = torch.stack(intrinsics_tensors)
        world_points_tensor = torch.stack(world_points_tensors)
        clicks_tensor = torch.stack(clicks_tensors)


    input_ids = (
        tokenizer_special_token(prompt, tokenizer, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )


    class KeywordsStoppingCriteria(StoppingCriteria):
        def __init__(self, keywords, tokenizer, input_ids):
            self.keywords = keywords
            self.keyword_ids = []
            for keyword in keywords:
                cur_keyword_ids = tokenizer(keyword).input_ids
                if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                    cur_keyword_ids = cur_keyword_ids[1:]
                self.keyword_ids.append(torch.tensor(cur_keyword_ids))
            self.tokenizer = tokenizer
            self.start_len = input_ids.shape[1]

        def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
            offset = min(output_ids.shape[1] - self.start_len, 3)
            self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
            for keyword_id in self.keyword_ids:
                # print('keyword_id:', keyword_id)
                # print('output_ids:', output_ids.shape)
                # print('output_ids slice:', output_ids[0, -keyword_id.shape[0]:].shape)
                if output_ids.size(1) >= keyword_id.size(0):
                    if output_ids[0, -keyword_id.shape[0]:] == keyword_id:
                        return True
            outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
            return False

    # class StopOnExplore(StoppingCriteria):
    #     def __init__(self, tokenizer, stop_word="<EXPLORE room"):
    #         self.stop_word = stop_word
    #         self.tokenizer = tokenizer
    #         self.stop_token_id = tokenizer.encode(stop_word, add_special_tokens=False)[0]

    #     def __call__(self, input_ids, scores, **kwargs):
    #         return self.stop_token_id in input_ids[0][-1:].tolist()

    # # Define the stopping criterion
    # stop_on_explore = StopOnExplore(tokenizer)

    # stop_str = conv.sep if conv.sep_stylse != SeparatorStyle.TWO else conv.sep2
    keywords = ['<EXPLORE room']
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)


    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            # depths=depths_tensor,
            # poses=poses_tensor,
            # intrinsics=intrinsics_tensor,
            world_points=world_points_tensor,
            clicks=clicks_tensor,
            image_sizes=None,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            # stopping_criteria=[stopping_criteria],  #StoppingCriteriaList([stop_on_explore]),  # Pass the stopping criteria here
        )

    # input_token_len = input_ids.shape[1]
    # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    # if n_diff_input_output > 0:
    #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    # outputs = outputs.strip()
    # if outputs.endswith(stop_str):
    #     outputs = outputs[:-len(stop_str)]
    # outputs = outputs.strip()
    # print(outputs)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print('outputs:', outputs)
    
    outputpath = '/home/whu/vl_research/MultiPLY/evaluation/evaluation_output'
    outputfile = '00031-Wo6kuutE9i7_031_150.json'
    d['prediction'] = outputs
    d['model_name'] = args.model_path
    with open(f'{outputpath}/{outputfile}', 'w') as f:
        json.dump(d, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video-path", type=str, help="Path to the video file")
    group.add_argument("--image-file", type=str, help="Path to the image file")
    # parser.add_argument("--video-path", type=str, help="Path to the video file")
    # parser.add_argument("--image-file", type=str, help="Path to the image file")

    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
