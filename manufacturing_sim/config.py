from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_SKEL_ROOT = DATA_ROOT / "raw" / "SKEL"
RAW_AMASS_ROOT = DATA_ROOT / "raw" / "AMASS"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class RobotSpec:
    name: str = "ASAP-TRAM UpperBody Cell Robot"
    shoulder_width_m: float = 0.46
    upper_arm_m: float = 0.34
    forearm_m: float = 0.32
    max_step_m: float = 0.035
    torque_limit_nm: float = 55.0
    workspace_radius_m: float = 0.68


@dataclass(frozen=True)
class ProcessSpec:
    name: str
    source_folder: str
    motion_file: str
    tolerance_m: float
    lag: float
    vibration: float
    material_compliance: float
    contact_noise: float


PROCESS_SPECS = {
    "rigid": ProcessSpec(
        name="rigid",
        source_folder="Rigid",
        motion_file="Model_Hand_R.MOTION",
        tolerance_m=0.045,
        lag=0.08,
        vibration=0.004,
        material_compliance=0.02,
        contact_noise=0.003,
    ),
    "semi_rigid": ProcessSpec(
        name="semi_rigid",
        source_folder="Semi-Rigid",
        motion_file="Model_Hand_R.MOTION",
        tolerance_m=0.045,
        lag=0.28,
        vibration=0.018,
        material_compliance=0.11,
        contact_noise=0.010,
    ),
    "flexible": ProcessSpec(
        name="flexible",
        source_folder="Flexible",
        motion_file="Model_Hand_R.MOTION",
        tolerance_m=0.045,
        lag=0.22,
        vibration=0.034,
        material_compliance=0.16,
        contact_noise=0.014,
    ),
}

