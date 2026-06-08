# motion_path = (
#     r"C:\\Users\\tjdrb\\project01\\tram_asap\data\\raw\SKEL\\Rigid\\Model_Hand_R.MOTION"
# )

# with open(motion_path, "r", encoding="utf-8") as f:

#     for i in range(20):

#         line = f.readline()

#         print(f"{i}: {line}")


import os
import numpy as np
from visualization.plot_trajectory import (plot_upper_body_trajectory)

class SKELLoader:

    def __init__(self, data_root):
        self.data_root = data_root

    def load_motion_file(self, motion_path):

        if not os.path.exists(motion_path):
            raise FileNotFoundError(
                f"파일이 존재하지 않습니다: {motion_path}"
            )

        motion_rows = []

        with open(
            motion_path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                # 빈 줄 제거
                if len(line) == 0:
                    continue

                parts = line.split()

                # 숫자 12개 아니면 skip
                if len(parts) != 12:
                    continue

                try:

                    row = [
                        float(x)
                        for x in parts
                    ]

                    motion_rows.append(row)

                except ValueError:
                    continue

        if len(motion_rows) == 0:
            raise ValueError(
                "유효한 motion 데이터가 없음"
            )

        raw_motion = np.array(
            motion_rows,
            dtype=np.float32
        )
        print("Original First Frame:")
        print(raw_motion[0])
        return raw_motion

    def extract_upper_body(self, motion_data):

        if len(motion_data.shape) != 2:
            raise ValueError(
                "잘못된 motion shape"
            )

        if motion_data.shape[1] < 12:
            raise ValueError(
                f"column 부족: {motion_data.shape[1]}"
            )

        return motion_data[:, :12]

    def normalize_motion(self, motion_data):

        mean = np.mean(motion_data, axis=0)

        std = np.std(motion_data, axis=0)

        std[std == 0] = 1e-6

        normalized = (motion_data - mean) / std

        return normalized

    def classify_environment(self, folder_name):

        folder_name = folder_name.lower()

        if "rigid" in folder_name:
            return "rigid"

        elif "semi" in folder_name:
            return "semi_rigid"

        elif "flexible" in folder_name:
            return "flexible"

        return "unknown"

    def generate_time_axis(
        self,
        num_frames,
        duration=5.0
    ):

        return np.linspace(
            0,
            duration,
            num_frames
        )

    def process_motion(
        self,
        environment_folder,
        motion_filename
    ):

        motion_path = os.path.join(
            self.data_root,
            environment_folder,
            motion_filename
        )

        print(f"Loading: {motion_path}")

        raw_motion = self.load_motion_file(
            motion_path
        )

        print(
            "Raw Motion Shape:",
            raw_motion.shape
        )

        upper_body = self.extract_upper_body(
            raw_motion
        )

        normalized = self.normalize_motion(
            upper_body
        )

        environment_type = (
            self.classify_environment(
                environment_folder
            )
        )

        time_axis = self.generate_time_axis(
            len(normalized)
        )

        return {
            "environment": environment_type,
            "time": time_axis,
            "trajectory": upper_body
        }


if __name__ == "__main__":

    DATA_ROOT = (r"C:\\Users\\tjdrb\\project01\\tram_asap\\data\\raw\\SKEL")

    loader = SKELLoader(DATA_ROOT)

    result = loader.process_motion(
        "Rigid",
        "Model_Hand_R.MOTION")

    print()

    print("Environment:", result["environment"])

    print("Trajectory Shape:", result["trajectory"].shape)

    print()
    print("First Frame:", result["trajectory"][0])
    
    plot_upper_body_trajectory(
    result["trajectory"],
    title="Rigid Manipulation Trajectory"
    )
    