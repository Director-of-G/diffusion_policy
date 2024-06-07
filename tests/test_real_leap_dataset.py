import numpy as np
import hydra
import zarr

from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)


def parse_dataset():
    with hydra.initialize('../diffusion_policy/config'):
        cfg = hydra.compose('train_diffusion_unet_real_image_workspace')
        OmegaConf.resolve(cfg)
        dataset = hydra.utils.instantiate(cfg.task.dataset)

    from matplotlib import pyplot as plt
    import pdb; pdb.set_trace()
    normalizer = dataset.get_normalizer()
    nactions = normalizer['action'].normalize(dataset.replay_buffer['action'][:])
    dists = np.linalg.norm(nactions, axis=-1)
    _ = plt.hist(dists, bins=100); plt.title('real action velocity')
    plt.show()


def rectify_actions():
    """
        The recorded actions are desired joint pos minus a fixed value.
        And the fixed value is only updated when relaunch data collection.
        This function is used to recover the true action from recorded values.
    """

    zarr_path = "/home/yongpeng/homework/Robotics/project/tmp/real_leap_20240605/replay_buffer.zarr"
    zarr_data = zarr.open(zarr_path, "rw")

    # detect restart idx in the data collection
    timestamps = zarr_data.data.timestamp[:]
    restart_idx = np.where(np.diff(timestamps) > 10.0)[0] + 1
    print(f"data collection restarts {len(restart_idx)} times!")

    robot_joint = zarr_data.data.robot_joint[:]
    action = np.array(zarr_data.data.action[:]).copy()
    data_collect_start_idx = [0] + restart_idx.tolist()
    data_collect_end_idx = restart_idx.tolist() + [len(robot_joint)]

    for start, end in zip(data_collect_start_idx, data_collect_end_idx):
        # this may not be the first desired point in code!
        # init_robot_joint = robot_joint[start]

        # estimate the first desired point
        fake_actions = action[start:end]
        robot_joint_offset = np.mean(robot_joint[start:end] - fake_actions, axis=0)

        true_desired_robot_joint = robot_joint_offset + fake_actions
        true_actions = true_desired_robot_joint - robot_joint[start:end]
        action[start:end] = true_actions

        # from matplotlib import pyplot as plt
        # plt.plot(robot_joint[start:end, 0])
        # plt.plot(true_desired_robot_joint[:, 0])

    from matplotlib import pyplot as plt
    import pdb; pdb.set_trace()

    # del zarr_data.data['action']
    # zarr_data.create_dataset('action', data=action, chunks=(400, 16), dtype="float64")


def rectify_actions_v2():
    """
        The recorded actions are desired joint pos minus a fixed value.
        And the fixed value is only updated when relaunch data collection.
        This function is used to recover the true action from recorded values.
    """

    zarr_path = "/home/yongpeng/homework/Robotics/project/tmp/real_leap_20240605/replay_buffer.zarr"
    zarr_data = zarr.open(zarr_path, "rw")

    robot_joint = zarr_data.data["robot_joint"][:]
    action = zarr_data.data["action"][:]

    desired_robot_joint = robot_joint + action

    from matplotlib import pyplot as plt
    import pdb; pdb.set_trace()

    # plt.plot(robot_joint[:, 0])
    # plt.plot(desired_robot_joint[:, 0])
    # plt.show()

    del zarr_data.data['action']
    zarr_data.create_dataset('action', data=desired_robot_joint, chunks=(400, 16), dtype="float64")


if __name__ == "__main__":
    # parse_dataset()
    # rectify_actions()
    # rectify_actions_v2()
    pass
