import numpy as np


class MotionSequenceDataset:

    def __init__(
        self,
        trajectory,
        window_size=60,
        stride=1
    ):

        self.trajectory = trajectory

        self.window_size = window_size

        self.stride = stride

        self.inputs = []

        self.targets = []

        self.build_dataset()

    def build_dataset(self):

        total_frames = len(
            self.trajectory
        )

        for start_idx in range(
            0,
            total_frames - self.window_size,
            self.stride
        ):

            end_idx = (
                start_idx
                + self.window_size
            )

            input_seq = (
                self.trajectory[
                    start_idx:end_idx
                ]
            )

            target = (
                self.trajectory[
                    end_idx
                ]
            )

            self.inputs.append(
                input_seq
            )

            self.targets.append(
                target
            )

        self.inputs = np.array(
            self.inputs,
            dtype=np.float32
        )

        self.targets = np.array(
            self.targets,
            dtype=np.float32
        )

    def get_data(self):

        return (
            self.inputs,
            self.targets
        )


if __name__ == "__main__":

    dummy = np.random.randn(
        4200,
        12
    )

    dataset = (
        MotionSequenceDataset(
            dummy,
            window_size=60
        )
    )

    X, y = dataset.get_data()

    print("Input Shape:", X.shape)

    print("Target Shape:", y.shape)