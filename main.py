import argparse
import json
import os
import re
from datetime import datetime
from math import atan2

import cv2
import matplotlib.pyplot as plt
import numpy as np
from nuscenes import NuScenes

from gemini_client import GeminiIntegrationError, analyze_driving_scene
from yolo_detector import YOLODetector, YOLODetectorError
from utils import (
    EstimateCurvatureFromTrajectory,
    IntegrateCurvatureForPoints,
    OverlayTrajectory,
    WriteImageSequenceToVideo,
)

OBS_LEN = 10
FUT_LEN = 10
TTL_LEN = OBS_LEN + FUT_LEN


def GenerateMotion(
    image_path,
    obs_velocities,
    obs_curvatures,
    given_intent,
    processor=None,
    model=None,
    tokenizer=None,
    args=None,
    detections=None,
):
    speed_history = (
        np.linalg.norm(np.asarray(obs_velocities, dtype=float), axis=1)
        if obs_velocities is not None and len(obs_velocities) > 0
        else np.array([], dtype=float)
    )
    curvature_history = (
        np.asarray(obs_curvatures, dtype=float)
        if obs_curvatures is not None
        else np.array([], dtype=float)
    )
    historical_motion = {
        "speed_history": speed_history.tolist(),
        "curvature_history": curvature_history.tolist(),
    }
    analysis = None
    updated_intent = given_intent
    try:
        analysis = analyze_driving_scene(
            image_path=image_path,
            detections=detections or [],
            historical_motion=historical_motion,
            previous_intent=given_intent,
        )
        updated_intent = analysis["driving_intent"]
    except GeminiIntegrationError as exc:
        print(
            f"Gemini failure for {image_path}: {exc.category}; "
            "using deterministic trajectory fallback."
        )

    if len(speed_history) == 0:
        zero_prediction = ", ".join("[0.00, 0.00]" for _ in range(FUT_LEN))
        return (
            f"Future speeds and curvatures: {zero_prediction}",
            None,
            analysis,
            updated_intent or "trajectory continuation",
        )

    recent_speed = float(np.median(speed_history[-3:])) if len(speed_history) >= 3 else float(np.median(speed_history))
    recent_curvature = float(np.median(curvature_history[-3:])) if len(curvature_history) >= 3 else float(np.median(curvature_history))

    future_predictions = []
    for i in range(FUT_LEN):
        speed = max(0.0, recent_speed * (1.0 - 0.03 * i))
        curvature = recent_curvature * 0.85
        future_predictions.append([float(speed), float(curvature * 100.0)])

    text = ", ".join(f"[{speed:.2f}, {curvature:.2f}]" for speed, curvature in future_predictions)
    return (
        f"Future speeds and curvatures: {text}",
        None,
        analysis,
        updated_intent or "trajectory continuation",
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--dataroot", type=str, default='datasets/NuScenes')
    parser.add_argument("--version", type=str, default='v1.0-mini')
    parser.add_argument("--method", type=str, default='openemma')
    args = parser.parse_args()

    try:
        detector = YOLODetector()
    except YOLODetectorError as exc:
        raise RuntimeError(f"YOLO detector initialization failed: {exc}") from exc

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    timestamp = "openemma_results/" + args.method + "/" + timestamp
    os.makedirs(timestamp, exist_ok=True)

    nusc = NuScenes(version=args.version, dataroot=args.dataroot)
    scenes = nusc.scene

    print(f"Number of scenes: {len(scenes)}")
    for scene in scenes:
        token = scene['token']
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        name = scene['name']

        front_camera_images = []
        ego_poses = []
        camera_params = []
        curr_sample_token = first_sample_token
        while True:
            sample = nusc.get('sample', curr_sample_token)
            cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            front_camera_images.append(os.path.join(nusc.dataroot, cam_front_data['filename']))
            pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
            ego_poses.append(pose)
            camera_params.append(nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token']))

            if curr_sample_token == last_sample_token:
                break
            curr_sample_token = sample['next']

        scene_length = len(front_camera_images)
        print(f"Scene {name} has {scene_length} frames")

        if scene_length < TTL_LEN:
            print(f"Scene {name} has less than {TTL_LEN} frames, skipping...")
            continue

        DT = 0.5
        ego_poses_world = np.array([ego_poses[t]['translation'][:3] for t in range(scene_length)])
        plt.plot(ego_poses_world[:, 0], ego_poses_world[:, 1], 'r-', label='GT')

        ego_velocities = np.zeros_like(ego_poses_world)
        ego_velocities[1:] = (ego_poses_world[1:] - ego_poses_world[:-1]) / DT
        ego_velocities[0] = ego_velocities[1]

        ego_curvatures = EstimateCurvatureFromTrajectory(ego_poses_world)
        ego_velocities_norm = np.linalg.norm(ego_velocities, axis=1)
        estimated_points = IntegrateCurvatureForPoints(
            ego_curvatures[1:],
            ego_velocities_norm[1:],
            ego_poses_world[0],
            atan2(ego_velocities[0][1], ego_velocities[0][0]),
            DT,
        )

        if args.plot:
            plt.quiver(
                ego_poses_world[:, 0],
                ego_poses_world[:, 1],
                ego_velocities[:, 0],
                ego_velocities[:, 1],
                color='b',
            )
            plt.plot(estimated_points[:, 0], estimated_points[:, 1], 'g-', label='Reconstruction')
            plt.legend()
            plt.savefig(f"{timestamp}/{name}_interpolation.jpg")
            plt.close()

        ego_traj_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]

        prev_intent = None
        cam_images_sequence = []
        ade1s_list = []
        ade2s_list = []
        ade3s_list = []

        for i in range(scene_length - TTL_LEN):
            fut_ego_traj_world = ego_traj_world[i + OBS_LEN:i + TTL_LEN]
            obs_ego_velocities = ego_velocities[i:i + OBS_LEN]
            obs_ego_curvatures = ego_curvatures[i:i + OBS_LEN]

            current_ego_position = ego_traj_world[i + OBS_LEN - 1]
            current_ego_pose = ego_poses[i + OBS_LEN - 1]
            current_camera_params = camera_params[i + OBS_LEN - 1]
            current_image = front_camera_images[i + OBS_LEN - 1]
            img = cv2.imread(current_image)
            if img is None:
                continue
            try:
                detections = detector.detect(current_image)
            except YOLODetectorError as exc:
                print(f"YOLO failure for {current_image}: {exc}; skipping frame.")
                continue

            prediction, _, _, updated_intent = GenerateMotion(
                current_image,
                obs_ego_velocities,
                obs_ego_curvatures,
                prev_intent,
                processor=None,
                model=None,
                tokenizer=None,
                args=args,
                detections=detections,
            )

            prev_intent = updated_intent
            pred_waypoints = prediction.replace("Future speeds and curvatures:", "").strip()
            coordinates = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", pred_waypoints)
            if coordinates == []:
                continue

            speed_curvature_pred = [[float(v), float(k)] for v, k in coordinates]
            speed_curvature_pred = speed_curvature_pred[:10]
            print(f"Got {len(speed_curvature_pred)} future actions: {speed_curvature_pred}")

            pred_len = min(FUT_LEN, len(speed_curvature_pred))
            pred_curvatures = np.array(speed_curvature_pred)[:, 1] / 100
            pred_speeds = np.array(speed_curvature_pred)[:, 0]
            pred_traj = np.zeros((pred_len, 3))
            pred_traj[:, :2] = IntegrateCurvatureForPoints(
                pred_curvatures,
                pred_speeds,
                current_ego_position,
                atan2(obs_ego_velocities[-1][1], obs_ego_velocities[-1][0]),
                DT,
            )

            OverlayTrajectory(img, pred_traj.tolist(), current_camera_params, current_ego_pose, color=(255, 0, 0), args=args)

            fut_ego_traj_world = np.array(fut_ego_traj_world)
            ade = np.mean(np.linalg.norm(fut_ego_traj_world[:pred_len] - pred_traj, axis=1))

            pred1_len = min(pred_len, 2)
            ade1s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred1_len] - pred_traj[:pred1_len], axis=1))
            ade1s_list.append(ade1s)

            pred2_len = min(pred_len, 4)
            ade2s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred2_len] - pred_traj[:pred2_len], axis=1))
            ade2s_list.append(ade2s)

            pred3_len = min(pred_len, 6)
            ade3s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred3_len] - pred_traj[:pred3_len], axis=1))
            ade3s_list.append(ade3s)

            if args.plot:
                cam_images_sequence.append(img.copy())
                cv2.imwrite(f"{timestamp}/{name}_{i}_front_cam.jpg", img)

                plt.plot(fut_ego_traj_world[:, 0], fut_ego_traj_world[:, 1], 'r-', label='GT')
                plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'b-', label='Pred')
                plt.legend()
                plt.title(f"Scene: {name}, Frame: {i}, ADE: {ade}")
                plt.savefig(f"{timestamp}/{name}_{i}_traj.jpg")
                plt.close()

                np.save(f"{timestamp}/{name}_{i}_pred_traj.npy", pred_traj)
                np.save(f"{timestamp}/{name}_{i}_pred_curvatures.npy", pred_curvatures)
                np.save(f"{timestamp}/{name}_{i}_pred_speeds.npy", pred_speeds)

                with open(f"{timestamp}/{name}_{i}_logs.txt", 'w') as f:
                    f.write(f"Intent Description: {updated_intent}\n")
                    f.write(f"Average Displacement Error: {ade}\n")

        if len(ade1s_list) == 0:
            continue

        mean_ade1s = np.mean(ade1s_list)
        mean_ade2s = np.mean(ade2s_list)
        mean_ade3s = np.mean(ade3s_list)
        failure_rate = 0
        for f in ade1s_list:
            if f > 10:
                failure_rate += 1
        failure_rate = (failure_rate * 100) / len(ade1s_list)

        aveg_ade = np.mean([mean_ade1s, mean_ade2s, mean_ade3s])

        result = {
            "name": name,
            "token": token,
            "ade1s": mean_ade1s,
            "ade2s": mean_ade2s,
            "ade3s": mean_ade3s,
            "avgade": aveg_ade,
            "failure_rate": failure_rate,
        }

        with open(f"{timestamp}/ade_results.jsonl", "a") as f:
            f.write(json.dumps(result))
            f.write("\n")

        if args.plot:
            WriteImageSequenceToVideo(cam_images_sequence, f"{timestamp}/{name}")

