from .rotation import *


def get_rel_rotmat(curr_quat, target_quat, global_frame=False):
    # quat (w, x, y, z)
    # get the rotmat of target w.r.t. curr quat
    cur_rot_matrix = quat2mat(curr_quat)
    target_rot_matrix = quat2mat(target_quat)
    if global_frame:
        relative_rot_matrix = np.matmul(target_rot_matrix, np.transpose(cur_rot_matrix))
    else:
        relative_rot_matrix = np.matmul(np.transpose(cur_rot_matrix), target_rot_matrix)

    return relative_rot_matrix


def get_rel_axisangle(curr_quat, target_quat, global_frame=False):
    relative_rot_matrix = get_rel_rotmat(curr_quat, target_quat, global_frame=global_frame)
    relative_rot_quat = mat2quat(relative_rot_matrix)
    return quat2axisangle(relative_rot_quat)


def get_rel_quat(curr_quat, target_quat, global_frame=False):
    rel_rotmat = get_rel_rotmat(curr_quat, target_quat, global_frame=global_frame)
    rel_quat = mat2quat(rel_rotmat)
    return rel_quat


def rotate_quat(curr_quat, rot_mat, global_frame=False):
    curr_rot_matrix = quat2mat(curr_quat)
    if global_frame:
        target_rot_matrix = rot_mat @ curr_rot_matrix
    else:
        target_rot_matrix = curr_rot_matrix @ rot_mat
    target_quat = mat2quat(target_rot_matrix)
    return target_quat
