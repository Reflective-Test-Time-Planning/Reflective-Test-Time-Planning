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
from tqdm.auto import tqdm
import sys
sys.path.append('/home/whu/vl_research/LLaVA-3D')
# from validate_traj_v2 import validate_trajectory
from validate_traj_states_wenbo_loose_nodrop import validate_trajectory


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

    mode = 'video'

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
    # with open('/local1/whu/data/hm3d/train_data/llava-3d_data_0415_simple_180_scenes_18kx10_gemini20flash_v7_memv4_cot_wentrooms_loose_15k_evaluation_samples.json', 'r') as f:
    # with open('/local1/whu/data/hm3d/train_data/llava-3d_data_0415_simple_180_scenes_18kx10_gemini20flash_v7_memv4_cot_wentrooms_loose_15k_unseen_objects_samples.json', 'r') as f:
   
    # with open('/local1/whu/data/hm3d/train_data/llava-3d_data_0415_hard_180_scenes_18kx15_gemini20flash_v7_loose_multipickup_droplast_5k_evaluation_samples.json', 'r') as f:
    # with open('/local1/whu/data/hm3d/train_data/llava-3d_data_0415_simple_180_scenes_18kx10_gemini20flash_v7_memv4_cot_wentrooms_loose_15k_unseen_objects_samples.json', 'r') as f:
   
    with open('/local1/whu/data/hm3d/train_data/llava-3d_data_0415_medium_180_scenes_18kx10_gemini20flash_v7_memv4_cot_wentrooms_loose_8k_evaluation_samples.json', 'r') as f:
    # with open('/local1/whu/data/hm3d/train_data/llava-3d_data_0415_medium_180_scenes_18kx10_gemini20flash_v7_memv4_cot_wentrooms_loose_8k_unseen_objects_samples.json', 'r') as f:

    # with open('/local1/whu/data/hm3d/train_data/llava-3d_data_0415_simple_180_scenes_18kx10_gemini20flash_v7_memv4_cot_wentrooms_loose_15k_unseen_objects_samples_everything_in_context.json', 'r') as f:
    # with open('/local1/whu/data/hm3d/train_data/llava-3d_data_0415_simple_180_scenes_18kx10_gemini20flash_v7_memv4_cot_wentrooms_loose_15k_evaluation_samples_everything_in_context.json', 'r') as f:
    

        
                #llava-3d_data_Feb28_v6_medium_180_scenes_18k_geminiv7_annotation_memv3_3k_cot_wentrooms.json', 'r') as f:   
    # with open('/local1/whu/data/hm3d/train_data/llava-3d_data_Feb28_v6_medium_180_scenes_18k_geminiv7_annotation_memv3_3k_cot_wentrooms_human_hint.json', 'r') as f:   
              #llava-3d_data_Feb28_v6_medium_180_scenes_18k_geminiv7_annotation_memv3_3k_cot.json', 'r') as f:
        #'/local1/whu/llava_3d_files/llava-3d_data_Feb26_v6_medium_train136_lastfix_annotation_0316_longtermmem.json', 'r') as f:
        #llava-3d_data_Feb26_v6_medium_train136_lastfix_annotation_0312_mem.json', 'r') as f:
              #llava-3d_data_Feb26_v6_medium_train136_lastfix_annotation.json', 'r') as f: 
    #('/local1/whu/llava_3d_files/llava-3d_data_Feb11_train1500_multirooms_433_annotation.json', 'r') as f:
    #('/home/whu/vl_research/MultiPLY/training_data_test_jan6_test1/00031-Wo6kuutE9i7_031_150.json') as f: #137, 157
        annotations = json.load(f)

    # annotations = annotations[:10]
    all_predictions = []
    # counter = 0
    for ann in tqdm(annotations):

        all_room_objects = ann['room_objects']
        room_order = ann['answer']['Room Order']
        gt_trajectory = ann['answer']['Trajectory']
        answer = ann['answer']
        if 'new objects' in answer:
                new_objs = answer['new objects']
        elif 'new_objects' in answer:
            new_objs = answer['new_objects']
        obj_result, evaluation_log, valid, states = validate_trajectory(room_order, gt_trajectory,all_room_objects, new_objs,break_on_error=True)
            
        if valid is False:
            pass
            print(f"invalid gt trajectory")
            print(f"evaluation_log: {evaluation_log}")
            # bp()
            continue
        
        # from pdb import set_trace; set_trace()
        system_message = ann['system_message'].replace('<video>', '<image>')
        video_paths = ann['training_data']['video']
        video_paths = ['/local1/whu/data/hm3d/semantic_room_data/original_2d_gt_seg_multiple_x_0226_v6_0415_medium_180_scenes_18kx10_gemini20flash_v7/'+v for v in video_paths]
        training_conv = ann['training_data']['conversations']

        # new_rooms_can_be_explored_str = "Room 1"
        # Rooms = ['Room 0<image>', 'Room 3<image>'] #'Room 1<image>']        
        # room_str = ', '.join(Rooms)

        # qs = system_message

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

        # video_paths = #json.loads(args.video_path)
        # video_paths = [args.video_path] 
    # ["/local1/whu/data/hm3d/semantic_room_data/original_2d_gt_seg_multiple_x_0106_addobjects/hm3d/00031-Wo6kuutE9i7_0", "/local1/whu/data/hm3d/semantic_room_data/original_2d_gt_seg_multiple_x_0106_addobjects/hm3d/00031-Wo6kuutE9i7_3"]' \
        # video_paths = None #

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

        last_out = []
    
        # count = 8
        # # print('training_conv:', training_conv)
        # for j in range(len(training_conv)//2):
        conv = conv_templates[args.conv_mode].copy()
        k = 0
        print("len(training_conv)", len(training_conv))
        for train_conv in training_conv:
            if train_conv['from'] == 'human':
                conv.append_message(conv.roles[0], train_conv['value'].replace('<video>', '<image>'))
            elif train_conv['from'] == 'gpt':
                # print('last_out:', last_out)
                # if len(last_out) != 0 and k < len(last_out):  
                #     # print('k:', k)   
                #     conv.append_message(conv.roles[1], last_out[k])
                #     k+=1
                # else:
                conv.append_message(conv.roles[1], None)     
            else: 
                print('unknown role \n\n\n')  

        prompt = conv.get_prompt()
        print('prompt:', prompt)

        # import sys; sys.exit(1)

        input_ids = (
            tokenizer_special_token(prompt, tokenizer, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )

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
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        print('outputs:', outputs)
        last_out.append(outputs)

        print('last_out:', last_out)
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
        ann['prediction'] = last_out
        all_predictions.append(ann)
        # counter += 1    
        # if counter >= 100: 
        #     break
    outputpath = '/home/whu/vl_research/MultiPLY/evaluation/evaluation_output'
    outputfile = '3dmem_ablation_medium_v6_EVAlSET' + args.model_path.split('/')[-1] + '.json'
    # outputfile = '00031-Wo6kuutE9i7_031_150.json'
    # d['prediction'] = outputs
    # d['model_name'] = args.model_path
    ## TODO save output each iteration TODOTODOTODOTODOTODOTODOTODOTODO
    with open(f'{outputpath}/{outputfile}', 'w') as f:
        json.dump(all_predictions, f, indent=4)


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
