from dataclasses import dataclass


@dataclass(frozen=True)
class DataSource:
    name: str
    role: str
    location: str
    use_in_pipeline: str


LOCAL_DATA_SOURCES = [
    DataSource(
        name="MPI SKEL hand-object motions",
        role="Minimal source data for rigid, semi-rigid, and flexible hand/object manipulation.",
        location="data/raw/SKEL",
        use_in_pipeline="Primary trajectory source for the first executable manufacturing-cell simulation.",
    ),
    DataSource(
        name="AMASS TCDHands",
        role="Upper-body and hand-heavy human motion sequences such as typing, writing, and gestures.",
        location="data/raw/AMASS/TCDHands",
        use_in_pipeline="Optional motion augmentation source for future two-hand assembly imitation.",
    ),
    DataSource(
        name="AMASS BMLmovi/CMU/Transitions",
        role="Whole-body human motion diversity.",
        location="data/raw/AMASS",
        use_in_pipeline="Optional domain randomization source for posture and timing variation.",
    ),
]


EXTERNAL_EXTENSION_SOURCES = [
    DataSource(
        name="IKEA ASM",
        role="Multi-view furniture assembly dataset with RGB, depth, action labels, human pose, object segments, and tracking.",
        location="https://ikeaasm.github.io/",
        use_in_pipeline="Add real assembly step ordering, object state labels, and pose supervision.",
    ),
    DataSource(
        name="Assembly101",
        role="Large multi-view procedural assembly/disassembly dataset with fine-grained action labels and 3D hand poses.",
        location="https://assembly-101.github.io/",
        use_in_pipeline="Add corrective actions, mistakes, and hand-object assembly variations.",
    ),
    DataSource(
        name="AMASS",
        role="Unified human motion capture archive represented with SMPL-family body models.",
        location="https://amass.is.tue.mpg.de/",
        use_in_pipeline="Expand upper-body motion priors and retargeting coverage.",
    ),
]

