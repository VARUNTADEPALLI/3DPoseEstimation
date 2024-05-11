import torch
import math
import cv2
import numpy as np
import sys
sys.path.append('/mnt/c/Users/varun/Downloads/3DHumanPoseEstimation/openpose/lightweight/modules')

from keypoints import  extract_keypoints, group_keypoints


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                    cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad

def get_bbox(joint_img):

    x_img, y_img = joint_img[:,0], joint_img[:,1]

    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img)

    x_center = (xmin+xmax)/2.; width = xmax-xmin
    xmin = x_center - 0.5*width*1.2
    xmax = x_center + 0.5*width*1.2
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin
    ymin = y_center - 0.5*height*1.2
    ymax = y_center + 0.5*height*1.2

    bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox

def process_bbox(bbox, aspect_ratio=None, scale=1.0):
    x, y, w, h = bbox
    x1, y1, x2, y2 = x, y, x + (w - 1), y + (h - 1)
    if w * h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        return None

    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.0
    c_y = bbox[1] + h / 2.0
    if aspect_ratio is None:
        aspect_ratio = 384 / 288
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * scale  # *1.25
    bbox[3] = h * scale  # *1.25
    bbox[0] = c_x - bbox[2] / 2.0
    bbox[1] = c_y - bbox[3] / 2.0
    return bbox

def get_center_scale(box_info):
        x, y, w, h = box_info

        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        scale = np.array([
            w * 1.0, h * 1.0
        ], dtype=np.float32)

        return center, scale

def get_affine_transform(center,scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def flip_2d_joint(kp, width, flip_pairs):
    kp[:, 0] = width - kp[:, 0] - 1
    for lr in flip_pairs:
        kp[lr[0]], kp[lr[1]] = kp[lr[1]].copy(), kp[lr[0]].copy()

    return kp

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def j2d_processing(kp, res, bbox, rot, f, flip_pairs):

    center, scale = get_center_scale(bbox)
    trans = get_affine_transform(center, scale, rot, res)

    nparts = kp.shape[0]
    for i in range(nparts):
        kp[i, :2] = affine_transform(kp[i, :2].copy(), trans)

    if f:
        kp = flip_2d_joint(kp, res[0], flip_pairs)
    kp = kp.astype('float32')
    return kp, trans

def joint_coordination(joint_coord):
    joint_coord = np.delete(joint_coord, (1), axis=0)
    idx = [0, 14, 13, 16, 15, 4, 1, 5, 2, 6, 3, 10, 7, 11, 8, 12, 9]
    joint_coord = joint_coord[idx]
    joint_coord = joint_coord.reshape(17, -1)
    # add pelvis joint
    lhip_idx = 11
    rhip_idx = 12
    pelvis = (joint_coord[lhip_idx, :] + joint_coord[rhip_idx, :]) * 0.5
    pelvis = pelvis.reshape(1, 2)
    joint_coord = np.concatenate((joint_coord, pelvis))
    # add neck
    lshoulder_idx = 5
    rshoulder_idx = 6
    neck = (joint_coord[lshoulder_idx, :] + joint_coord[rshoulder_idx, :]) * 0.5
    neck = neck.reshape(1, 2)
    joint_coord = np.concatenate((joint_coord, neck))
    # optimizing cam param
    bbox = get_bbox(joint_coord)
    bbox1 = process_bbox(bbox.copy(), aspect_ratio=1.0, scale=1.25)
    bbox2 = process_bbox(bbox.copy())
    proj_target_joint_img, trans = j2d_processing(joint_coord.copy(), (500, 500), bbox1, 0, 0, None)
    joint_img, _ = j2d_processing(joint_coord.copy(), (384, 288), bbox2, 0, 0, None)
    joint_img = joint_img[:, :2]
    joint_img /= np.array([[384, 288]])
    mean, std = np.mean(joint_img, axis=0), np.std(joint_img, axis=0)
    joint_img = (joint_img.copy() - mean) / std
    joint_img = joint_img[None, :, :]

    return joint_img

GTRS = torch.jit.load(
    "/mnt/c/Users/varun/Downloads/3DPoseEstimation/GTRS/GTRS-mains/gpu.pt", map_location=torch.device("cpu")
)

TwoDPoseDetection = torch.jit.load(
    "/mnt/c/Users/varun/Downloads/3DHumanPoseEstimation/openpose/lightweight/results.pt", map_location=torch.device("cpu")
)


GTRS.eval().to("cpu")
TwoDPoseDetection.eval().to("cpu")




device = "cuda" if torch.cuda.is_available() else "cpu"
cpu = True

net = TwoDPoseDetection.eval()
# read image
img = cv2.imread("/mnt/c/Users/varun/Downloads/3DHumanPoseEstimation/openpose/lightweight/input.jpeg", cv2.IMREAD_COLOR)
stride = 8
upsample_ratio = 4
num_keypoints = 18
net_input_height_size = 256

height, width, _ = img.shape
scale = net_input_height_size / height
pad_value=(0, 0, 0)
img_mean=np.array([128, 128, 128], np.float32)
img_scale=np.float32(1/256)

scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
scaled_img = normalize(scaled_img, img_mean, img_scale)
min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
if not cpu:
    tensor_img = tensor_img.cuda()

stages_output = TwoDPoseDetection(tensor_img)

stage2_heatmaps = stages_output[-2]
heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

stage2_pafs = stages_output[-1]
pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

total_keypoints_num = 0
all_keypoints_by_type = []
for kpt_idx in range(num_keypoints):  # 19th for bg
    total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
for kpt_id in range(all_keypoints.shape[0]):
    all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
    all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
current_poses = []
for n in range(len(pose_entries)):
    if len(pose_entries[n]) == 0:
        continue
    pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
    for kpt_id in range(num_keypoints):
        if pose_entries[n][kpt_id] != -1.0:
            pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
            pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
    current_poses.append(pose_keypoints)

joint_img = joint_coordination(current_poses[0])

joint_img = torch.Tensor(joint_img)

torch.onnx.export(
    GTRS,
    joint_img,
    "/mnt/c/Users/varun/Downloads/3DHumanPoseEstimation/openpose/lightweight/GTRS.onnx",
    verbose=True,
    input_names=["joints"],
    output_names=["3dmesh", "3dpose"],
)
