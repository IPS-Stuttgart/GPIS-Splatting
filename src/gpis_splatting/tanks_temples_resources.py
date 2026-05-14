from __future__ import annotations

from dataclasses import dataclass

TANKS_TEMPLES_SOURCE_URL = "https://www.tanksandtemples.org/download/"
TANKS_TEMPLES_TUTORIAL_URL = "https://www.tanksandtemples.org/tutorial/"
TANKS_TEMPLES_LICENSE_URL = "https://www.tanksandtemples.org/license/"
TANKS_TEMPLES_IMAGE_BASE_URL = "https://storage.googleapis.com/t2-downloads/image_sets"
TANKS_TEMPLES_GT_BASE_URL = "https://storage.googleapis.com/t2-training-gt-data"
GOOGLE_DRIVE_DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id={file_id}"
TANKS_TEMPLES_TRAINING_BUNDLE_FILE_ID = "0B-ePgl6HF260dU1pejdkeXdMb00"
SUPPORTED_TANKS_TEMPLES_SCENES = ("Barn", "Ignatius", "Truck")
TANKS_TEMPLES_RESOURCE_NAMES = ("images", "reconstruction", "camera_log", "alignment", "crop", "ground_truth")


@dataclass(frozen=True)
class TanksTemplesResource:
    name: str
    relative_path: str
    url: str
    source: str
    archive: bool = False
    archive_member: str | None = None


@dataclass(frozen=True)
class TanksTemplesSceneSpec:
    scene: str
    resources: tuple[TanksTemplesResource, ...]


def google_drive_url(file_id: str) -> str:
    return GOOGLE_DRIVE_DOWNLOAD_URL.format(file_id=file_id)


TANKS_TEMPLES_TRAINING_BUNDLE_URL = google_drive_url(TANKS_TEMPLES_TRAINING_BUNDLE_FILE_ID)


def image_set_resource(scene: str) -> TanksTemplesResource:
    return TanksTemplesResource(
        name="images",
        relative_path=f"image_sets/{scene}.zip",
        url=f"{TANKS_TEMPLES_IMAGE_BASE_URL}/{scene}.zip",
        source="tanks_temples:image_set",
        archive=True,
    )


def training_bundle_resource(name: str, scene: str, relative_dir: str, suffix: str) -> TanksTemplesResource:
    return TanksTemplesResource(
        name=name,
        relative_path=f"{relative_dir}/{scene}{suffix}",
        url=TANKS_TEMPLES_TRAINING_BUNDLE_URL,
        source="tanks_temples:training_bundle",
        archive_member=f"{scene}{suffix}",
    )


def training_bundle_resources(scene: str) -> tuple[TanksTemplesResource, ...]:
    return (
        training_bundle_resource("reconstruction", scene, "reconstruction", ".ply"),
        training_bundle_resource("camera_log", scene, "camera_poses", ".log"),
        training_bundle_resource("alignment", scene, "alignment", ".txt"),
        training_bundle_resource("crop", scene, "crop", ".json"),
    )


TANKS_TEMPLES_SCENES: dict[str, TanksTemplesSceneSpec] = {
    "Barn": TanksTemplesSceneSpec(
        scene="Barn",
        resources=(
            image_set_resource("Barn"),
            *training_bundle_resources("Barn"),
            TanksTemplesResource(
                name="ground_truth",
                relative_path="ground_truth/Barn.ply",
                url=f"{TANKS_TEMPLES_GT_BASE_URL}/Barn/Barn.ply",
                source="tanks_temples:training_ground_truth",
            ),
        ),
    ),
    "Ignatius": TanksTemplesSceneSpec(
        scene="Ignatius",
        resources=(
            image_set_resource("Ignatius"),
            TanksTemplesResource(
                name="reconstruction",
                relative_path="reconstruction/Ignatius.ply",
                url=google_drive_url("1K4TFKLuD-lvJtU6iY-DsxTlp9y85nb6U"),
                source="tanks_temples:training_colmap_reconstruction",
            ),
            TanksTemplesResource(
                name="camera_log",
                relative_path="camera_poses/Ignatius.log",
                url=google_drive_url("172dDxEcJyA6i2Ih3zy3QNWK1R-2sAIi_"),
                source="tanks_temples:training_camera_poses",
            ),
            TanksTemplesResource(
                name="alignment",
                relative_path="alignment/Ignatius.txt",
                url=google_drive_url("1wSCbCrOT7GsGVLDq0aXs4RHhs4FUzUeM"),
                source="tanks_temples:training_alignment",
            ),
            TanksTemplesResource(
                name="crop",
                relative_path="crop/Ignatius.json",
                url=google_drive_url("1_0fESbNxfNI5NWzQ4RBhh460a9_0J54q"),
                source="tanks_temples:training_crop",
            ),
            TanksTemplesResource(
                name="ground_truth",
                relative_path="ground_truth/Ignatius.ply",
                url=f"{TANKS_TEMPLES_GT_BASE_URL}/Ignatius/Ignatius.ply",
                source="tanks_temples:training_ground_truth",
            ),
        ),
    ),
    "Truck": TanksTemplesSceneSpec(
        scene="Truck",
        resources=(
            image_set_resource("Truck"),
            *training_bundle_resources("Truck"),
            TanksTemplesResource(
                name="ground_truth",
                relative_path="ground_truth/Truck.ply",
                url=f"{TANKS_TEMPLES_GT_BASE_URL}/Truck/Truck.ply",
                source="tanks_temples:training_ground_truth",
            ),
        ),
    ),
}
