import warnings
import numpy as np
import os
import mujoco

from roboworld.envs.asset_path_utils import full_path_for
from roboworld.envs.mujoco_env.franka.franka_env import FrankaEnv
from roboworld.envs.mujoco_env.utils.rotation import euler2quat, quat2mat, mat2quat
from roboworld.envs.generator_cupboard_table import generate_xml, get_color_name


class FrankaCupboardEnv(FrankaEnv):

    def __init__(self, cupboard_name, shape_names, shape_descriptions, model_name="main.xml",
                 render_mode='offscreen', frame_skip=5, max_episode_length=5000,
                 control_mode="script", **kwargs):
        hand_low = (-0.5, -0.7, 0)
        hand_high = (1, 0.5, 1.5)

        self._model_name = model_name

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            frame_skip=frame_skip,
            **kwargs
        )
        self.max_path_length = max_episode_length
        assert control_mode in {"script"}, f"Unknown control mode: {control_mode}."
        self.control_mode = control_mode

        self.init_config = {
            'obj_init_pos': np.array((0, 0.6, 0.02), dtype=np.float32),
            'hand_init_pos': np.array((0., 0., 0.9), dtype=np.float32),
            'hand_init_quat': np.array((1., 0., 0., 0.), dtype=np.float32)
        }
        self.goal = np.array([0.1, 0.8, 0.1], dtype=np.float32)
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']
        self.hand_init_quat = self.init_config['hand_init_quat']
        self.robot_base_pos = self.get_body_pos("link0")

        self.cupboard_name = cupboard_name
        self.cupboard_id = self.model.body(cupboard_name).id
        self.shape_names = shape_names
        self.shape_desc = {k: v for k, v in zip(shape_names, shape_descriptions)}
        self.shape_colors = []
        for name in shape_names:
            geom_adr = self.model.body(name).geomadr
            assert geom_adr != -1
            color = get_color_name(self.model.geom(geom_adr).rgba[:3])
            self.shape_colors.append(color)
        self.shape_ids = [self.model.body(name).id for name in shape_names]

        # Define workspace bounds
        self.shape_pos_low = (-0.30, -0.70, 0.50)
        self.shape_pos_high = (0.20, 0.70, 0.50)
        self.cupboard_pos_low = (0.20, 0.10, 0.50)
        self.cupboard_pos_high = (0.40, 0.30, 0.50)

        # Store initial poses of shapes (they're already positioned by generator)
        self.object_init_poses = {shape_name: {
            "pos": self.get_body_pos(shape_name), "quat": self.get_body_quat(shape_name)
        } for shape_name in shape_names}

        self.goal_images = None

    @property
    def model_name(self):
        return full_path_for(self._model_name)

    def _get_obs_dict(self):
        obs_dict = super()._get_obs_dict()
        return obs_dict

    def reset_model(self, max_trials=50):
        self._reset_hand()
        # Objects are already positioned by generator, no randomization needed
        return self._get_obs()

    def compute_reward(self, actions, obs):
        raise NotImplementedError

    def act_pick_up(self, obj_name):
        err = 0
        obj_in_hand = self.get_object_in_hand()
        if obj_in_hand is not None:
            if obj_in_hand != obj_name:
                print(f"Cannot pick up `{obj_name}` since another object is grasped in hand. "
                      f"(`{obj_in_hand}` is in hand)")
                err = -1
            else:
                print(f"Should not pick up `{obj_name}` since it is already in hand.")
                err = -2
            return err
        
        # Use standard pickup for all shapes
        self.pickup(obj_name)
        return err

    def done_pick_up(self, obj_name):
        self.goto(quat=[1, 0, 0, 0], pos_err_th=0.05, quat_err_th=0.01)

        return self.object_is_in_hand(obj_name)

    def act_put_down(self, obj_name, target_site=None):
        err = 0
        if self.object_is_in_hand(obj_name):
            if target_site is not None:
                # Put in specific compartment
                self.place_at_site(obj_name, target_site)
            else:
                # Put on table
                self.put_on_table(obj_name)
        else:
            print(f"Cannot put down `{obj_name}` since it is not grasped in hand.")
            err = -1
        return err

    def done_put_down(self, obj_name, target_site=None):
        if target_site is not None:
            # Check if object is properly placed in compartment
            obj_pos = self.get_body_pos(obj_name)
            target_pos = self._get_site_pos(target_site)
            return not self.object_is_in_hand(obj_name) and \
                self.pos_err(obj_pos, target_pos) < 0.02
        else:
            # Check if object is on table and hand is reset
            return not self.object_is_in_hand(obj_name) and \
                self.pos_err(self.eef_pos, self.hand_init_pos) < 0.002 and \
                self.quat_err(self.eef_quat, self.hand_init_quat) < 0.002

    def act_txt(self, text, target_site = None):
        text_split = text.split()
        
        if len(text_split) < 2:
            print(f"Invalid command: {text}")
            return -1
            
        # Parse commands like "pick up the red" or "put down the blue"
        if "pick up" in text:
            color = text_split[-1]  # Last word should be color
            try:
                shape_ind = self.shape_colors.index(color)
                obj_name = self.shape_names[shape_ind]
                return self.act_pick_up(obj_name)
            except ValueError:
                print(f"Unknown color: {color}")
                return -1
                
        elif "put down" in text:
            color = text_split[-1]  # Last word should be color
            try:
                shape_ind = self.shape_colors.index(color)
                obj_name = self.shape_names[shape_ind]
                # # Check if there's a target compartment specified
                # target_site = None
                # if "in compartment" in text:
                #     # Extract compartment number if specified
                #     # e.g., "put down the red in compartment 1"
                #     comp_idx = None
                #     for i, word in enumerate(text_split):
                #         if word == "compartment" and i + 1 < len(text_split):
                #             try:
                #                 comp_idx = int(text_split[i + 1]) - 1  # Convert to 0-based
                #                 target_site = f"shape_{comp_idx + 1}_target"
                #                 break
                #             except ValueError:
                #                 pass
                
                return self.act_put_down(obj_name, target_site)
            except ValueError:
                print(f"Unknown color: {color}")
                return -1
        else:
            print(f"Unknown action: {text}")
            return -1

    def is_success(self, pos_err_th=0.02, debug=False):
        """Check if all shapes are in their target compartments"""
        for i, shape_name in enumerate(self.shape_names):
            target_site = f"shape_{i+1}_target"
            if not self.object_is_in_compartment(shape_name, target_site, pos_err_th, debug=debug):
                if debug:
                    print(f"{shape_name} ({self.shape_colors[i]}) is not in target compartment.")
                return False
        return True

    def object_is_in_compartment(self, shape_name, target_site, pos_err_th=0.02, debug=False):
        """Check if object is positioned in the target compartment"""
        try:
            obj_pos = self.get_body_pos(shape_name)
            target_pos = self._get_site_pos(target_site)
            pos_err = self.pos_err(obj_pos, target_pos)
            if debug:
                print(f"{shape_name} pos err to {target_site}: {pos_err:.4f}")
            return pos_err <= pos_err_th
        except:
            if debug:
                print(f"Could not find target site {target_site}")
            return False

    def release_object(self, object_name):
        weld_name = f"{object_name}_grasp_hand"
        weld_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, weld_name)
        self.data.eq_active[weld_id] = 0
        self.open_gripper()

    def place_at_site(self, obj_name, target_site):
        """Place object at a specific site (e.g., compartment target)"""
        # try:
            # target_pos = self._get_site_pos(site_name)
        target_pos = target_site
        #target_quat = self._get_site_quat(site_name) if hasattr(self, '_get_site_quat') else [1, 0, 0, 0]
        target_quat = [1, 0, 0, 0]

        # Move to above target first
        above_target = target_pos.copy()
        above_target[2] += 0.2  # 10cm above
        
        self.goto(pos=above_target, quat=[1, 0, 0, 0])

        above_target[2] -= 0.1  # 10cm above
        self.goto(pos=above_target, quat=[1, 0, 0, 0])
        
        # # Lower to target position
        # self.goto(pos=target_pos, quat=target_quat)
        
        # Release object
        self.release_object(obj_name)
        
        # Move hand away
        self.goto(pos=self.hand_init_pos, quat=self.hand_init_quat)
            
        # except Exception as e:
        #     print(f"Error placing {obj_name} at {site_name}: {e}")
        #     # Fallback to table placement
        #     self.put_on_table(obj_name)


def main():
    """Main function to test the cupboard fitting environment"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3012, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--output_dir", type=str, default="fitting_tasks_table", help="Directory containing task files")
    args = parser.parse_args()
    
    # Generate XML file using the cupboard generator
    xml, info = generate_xml(args.seed)
    
    # Save XML file
    xml_filename = f"cupboard_task_{args.seed}.xml"
    xml_path = os.path.join("assets", args.output_dir, xml_filename)
    os.makedirs(args.output_dir, exist_ok=True)
    xml.write_to_file(xml_path)
    
    print(f"Generated XML: {xml_path}")
    print(f"Task info: {info['n_shapes']} shapes, {info['n_compartments']} compartments")
    
    # Set up environment parameters
    cupboard_name = "brick_1"  # Cupboard structure name
    shape_ids = [j for j in range(1, info['n_shapes']+1)]
    shape_names = [f"shape_{j}" for j in shape_ids]
    shape_descriptions = [info['shape_descriptions'][shape_name] for shape_name in shape_names]
    shape_labels = [" ".join(pd.split()[:1]) for pd in shape_descriptions]
    shape_compartment_placements = [shape["target_compartment"] for shape in info["shapes_info"]]
    compartment_ids = [j for j in range(1, info['n_compartments']+1)]
    compartment_names = [f"back_panel_{j}" for j in shape_ids]
    compartment_poses = [comp["pos"] for comp in info["compartments_info"]]
    
    print(f"Cupboard: {cupboard_name}")
    print(f"Shapes: {shape_names}")
    print(f"Descriptions: {shape_descriptions}")
    print(f"Labels: {shape_labels}")

    render_mode = "rgb_array"
    
    # Create environment
    xml_path = os.path.join(args.output_dir, xml_filename)
    env = FrankaCupboardEnv(
        cupboard_name=cupboard_name,
        shape_names=shape_names,
        shape_descriptions=shape_descriptions,
        render_mode=render_mode,
        frame_skip=20,
        model_name=xml_path,
        magic_attaching=True
    )
    
    print("Environment created successfully!")
    
    # Test basic functionality
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    indices = list(range(len(shape_names)))
    import random
    random.shuffle(indices)

    # Test picking up first shape
    for i in indices:
        first_shape = shape_names[i]
        first_color = shape_labels[i]
        
        print(f"\nTesting pickup of {first_shape} ({first_color})...")
        
        # Test text command
        err = env.act_txt(f"pick up the {first_color}")
        if err == 0:
            print("✅ Pickup command executed successfully")
            
            # Check if done
            if env.done_pick_up(first_shape):
                print("✅ Pickup completed successfully")
                
                compartment = shape_compartment_placements[i]
                compartment_pose = compartment_poses[compartment]

                # Test put down
                print(f"Testing put down of {first_shape}...")
                err = env.act_txt(f"put down the {first_color}", compartment_pose)
                if err == 0:
                    print("✅ Put down command executed successfully")
                    
                    # Check if done
                    if env.done_put_down(first_shape):
                        print("✅ Put down completed successfully")
                    else:
                        print("⚠️ Put down not completed")
                else:
                    print(f"❌ Put down failed with error: {err}")
            else:
                print("⚠️ Pickup not completed")
        else:
            print(f"❌ Pickup failed with error: {err}")
    
    # # Test success check
    # success = env.is_success(debug=True)
    # print(f"\nTask success: {success}")
    
    # Print shape positions
    print("\nShape positions:")
    for i, shape_name in enumerate(shape_names):
        pos = env.get_body_pos(shape_name)
        color = shape_labels[i]
        print(f"  {shape_name} ({color}): ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    
    print("\nTest completed!")


if __name__ == "__main__":
    main()