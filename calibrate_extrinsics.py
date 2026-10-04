import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


# ============================================================
# USER CONFIGURATION
# ============================================================

# Camera intrinsic matrix
K = np.array([
    [808.95296262, 0.0, 365.22137067],
    [0.0, 789.09771349, 261.07802709],
    [0.0, 0.0, 1.0]
], dtype=np.float64)


# World coordinates of:
#   1. top-left inner corner
#   2. top-right inner corner
#   3. bottom-right inner corner
#
# IMPORTANT:
# These must correspond to the inner checkerboard corners,
# not the outer physical corners of the checkerboard.
CORNERS = np.array([
    [1.292727, -3.826339,  0.560793],
    [0.119595, -4.030364,  0.566427],
    [0.119755, -4.018098, -0.223339]
], dtype=np.float64)


# ============================================================
# GLIM TRAJECTORY
# ============================================================

def read_glim_trajectory(path):
    """
    Read GLIM trajectory.

    Expected columns:
        timestamp x y z qx qy qz qw

    Returns:
        timestamps : (N,) seconds
        positions  : (N,3)
        rotations  : scipy Rotation containing N rotations

    The pose convention assumed here is:

        p_world = R_world_lidar @ p_lidar + t_world_lidar
    """

    data = np.loadtxt(path)

    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] < 8:
        raise ValueError(
            "GLIM trajectory must have at least 8 columns: "
            "timestamp x y z qx qy qz qw"
        )

    timestamps = data[:, 0]
    positions = data[:, 1:4]

    # scipy expects quaternion order [x, y, z, w]
    rotations = Rotation.from_quat(data[:, 4:8])

    return timestamps, positions, rotations


def interpolate_pose(timestamp, timestamps, positions, rotations):
    """
    Interpolate the GLIM trajectory at a requested timestamp.

    Position: linear interpolation.

    Orientation: spherical linear interpolation (SLERP).

    Returns:
        position (3,)
        rotation (scipy Rotation)
    """

    if timestamp < timestamps[0] or timestamp > timestamps[-1]:
        raise ValueError(
            f"Timestamp {timestamp:.6f} outside GLIM trajectory range "
            f"[{timestamps[0]:.6f}, {timestamps[-1]:.6f}]"
        )

    # Exact sample
    idx = np.searchsorted(timestamps, timestamp)

    if idx == 0:
        return positions[0].copy(), rotations[0]

    if idx == len(timestamps):
        return positions[-1].copy(), rotations[-1]

    if timestamps[idx] == timestamp:
        return positions[idx].copy(), rotations[idx]

    t0 = timestamps[idx - 1]
    t1 = timestamps[idx]

    alpha = (timestamp - t0) / (t1 - t0)

    # Linear position interpolation
    position = (
        (1.0 - alpha) * positions[idx - 1]
        + alpha * positions[idx]
    )

    # SLERP orientation interpolation
    key_times = np.array([t0, t1])
    key_rots = Rotation.concatenate(
        [rotations[idx - 1], rotations[idx]]
    )

    slerp = Slerp(key_times, key_rots)
    rotation = slerp([timestamp])[0]

    return position, rotation


# ============================================================
# IMAGE TIMESTAMPS
# ============================================================

def timestamp_from_filename(filename, timezone_offset_hours=0.0):
    """
    Extract timestamp from filenames of the form:

        20260827_135740_619_IR_normalized.tiff

    Meaning:

        YYYYMMDD_HHMMSS_milliseconds

    The timestamp is interpreted as UTC plus
    timezone_offset_hours.

    For example, if the filename represents Japan local time,
    use timezone_offset_hours=9.

    Returns:
        Unix timestamp in seconds.
    """

    pattern = r"(\d{8})_(\d{6})_(\d{3})"

    match = re.search(pattern, filename)

    if match is None:
        raise ValueError(
            f"Could not find timestamp in filename: {filename}"
        )

    date_string = match.group(1)
    time_string = match.group(2)
    milliseconds = int(match.group(3))

    year = int(date_string[0:4])
    month = int(date_string[4:6])
    day = int(date_string[6:8])

    hour = int(time_string[0:2])
    minute = int(time_string[2:4])
    second = int(time_string[4:6])

    import datetime

    dt = datetime.datetime(
        year,
        month,
        day,
        hour,
        minute,
        second,
        milliseconds * 1000,
        tzinfo=datetime.timezone.utc,
    )

    timestamp = dt.timestamp()

    # If filename time is local time rather than UTC,
    # convert it to UTC.
    timestamp -= timezone_offset_hours * 3600.0

    return timestamp


# ============================================================
# CHECKERBOARD GEOMETRY
# ============================================================

def create_checkerboard_points(inner_cols, inner_rows):
    """
    Construct world coordinates for all checkerboard inner corners.

    CORNERS contains:
        [0] = top-left
        [1] = top-right
        [2] = bottom-right

    The generated ordering is row-major:

        top-left -> top-right
        then second row
        ...
        bottom-left -> bottom-right

    This matches the usual OpenCV checkerboard corner ordering
    assuming the supplied three corners correspond to that
    orientation.
    """

    if inner_cols < 2 or inner_rows < 2:
        raise ValueError("Checkerboard dimensions must be >= 2 x 2.")

    top_left = CORNERS[0]
    top_right = CORNERS[1]
    bottom_right = CORNERS[2]

    # Direction between adjacent corners in the horizontal direction
    step_x = (top_right - top_left) / (inner_cols - 1)

    # Vertical direction.
    #
    # bottom_right = top_right + (inner_rows - 1) * step_y
    step_y = (bottom_right - top_right) / (inner_rows - 1)

    points = []

    for row in range(inner_rows):
        for col in range(inner_cols):
            point = top_left + col * step_x + row * step_y
            points.append(point)

    return np.asarray(points, dtype=np.float64)


# ============================================================
# CHECKERBOARD DETECTION
# ============================================================

def detect_checkerboard(image, pattern_size):
    """
    Detect checkerboard inner corners.

    Returns:
        corners : (N,1,2) subpixel corner coordinates
        or None if detection fails.
    """

    gray = image

    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # Try the modern SB detector first.
    # It tends to be more robust for imperfect images.
    flags = (
        cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    found, corners = cv2.findChessboardCornersSB(
        gray,
        pattern_size,
        flags=flags
    )

    if found:
        return corners

    # Fallback to the classic detector
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags
    )

    if not found:
        return None

    # Subpixel refinement
    criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    corners = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria,
    )

    return corners


# ============================================================
# TRANSFORM UTILITIES
# ============================================================

def make_transform(rotation, translation):
    """
    Create 4x4 homogeneous transform.
    """

    T = np.eye(4)

    T[:3, :3] = rotation.as_matrix()
    T[:3, 3] = translation

    return T


def invert_transform(T):
    """
    Invert a rigid 4x4 transformation.
    """

    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t

    return T_inv


def transform_point(T, p):
    """
    Transform a 3D point using a 4x4 transform.
    """

    return T[:3, :3] @ p + T[:3, 3]


# ============================================================
# PNP
# ============================================================

def solve_camera_pose(object_points, image_points, K):
    """
    Solve:

        X_camera = R_CW X_world + t_CW

    using PnP.

    Returns:
        T_CW
        reprojection_rmse
    """

    # No distortion coefficients are supplied by the user,
    # so assume zero distortion.
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        raise RuntimeError("solvePnP failed.")

    R_CW, _ = cv2.Rodrigues(rvec)

    T_CW = np.eye(4)
    T_CW[:3, :3] = R_CW
    T_CW[:3, 3] = tvec.ravel()

    # Reprojection error
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        K,
        dist_coeffs,
    )

    projected = projected.reshape(-1, 2)
    observed = image_points.reshape(-1, 2)

    errors = np.linalg.norm(projected - observed, axis=1)

    rmse = np.sqrt(np.mean(errors ** 2))

    return T_CW, rmse


# ============================================================
# MAIN CALIBRATION
# ============================================================

def calibrate(
    image_folder,
    glim_path,
    inner_cols,
    inner_rows,
    timezone_offset_hours=0.0,
    time_offset=0.0,
    max_reprojection_error=3.0,
):
    """
    Perform camera-LiDAR extrinsic calibration.
    """

    # --------------------------------------------------------
    # Load trajectory
    # --------------------------------------------------------

    traj_t, traj_p, traj_R = read_glim_trajectory(glim_path)

    print(f"Loaded {len(traj_t)} GLIM trajectory poses.")

    # --------------------------------------------------------
    # Checkerboard geometry
    # --------------------------------------------------------

    object_points = create_checkerboard_points(
        inner_cols,
        inner_rows,
    )

    print(
        f"Checkerboard: {inner_cols} x {inner_rows} "
        f"inner corners"
    )

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    image_paths = sorted(
        list(Path(image_folder).glob("*.tiff"))
        + list(Path(image_folder).glob("*.tif"))
    )

    if len(image_paths) == 0:
        raise RuntimeError(
            f"No TIFF images found in {image_folder}"
        )

    print(f"Found {len(image_paths)} images.")

    # --------------------------------------------------------
    # Estimate T_CL for every image
    # --------------------------------------------------------

    transforms_CL = []
    reprojection_errors = []

    for image_path in image_paths:

        try:
            image_timestamp = timestamp_from_filename(
                image_path.name,
                timezone_offset_hours,
            )

            # Optional correction for different timestamp scales.
            #
            # For example, if GLIM timestamps are GPST while
            # filenames contain UTC, you can supply the required
            # offset here.
            trajectory_timestamp = (
                image_timestamp + time_offset
            )

        except ValueError as e:
            print(f"[SKIP] {image_path.name}: {e}")
            continue

        try:
            lidar_position, lidar_rotation = interpolate_pose(
                trajectory_timestamp,
                traj_t,
                traj_p,
                traj_R,
            )
        except ValueError as e:
            print(f"[SKIP] {image_path.name}: {e}")
            continue

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            print(f"[SKIP] Could not read {image_path}")
            continue

        # ----------------------------------------------------
        # Detect checkerboard
        # ----------------------------------------------------

        image_points = detect_checkerboard(
            image,
            (inner_cols, inner_rows),
        )

        if image_points is None:
            print(
                f"[SKIP] Checkerboard not found: "
                f"{image_path.name}"
            )
            continue

        # ----------------------------------------------------
        # PnP
        # ----------------------------------------------------

        try:
            T_CW, reprojection_rmse = solve_camera_pose(
                object_points,
                image_points,
                K,
            )
        except RuntimeError as e:
            print(f"[SKIP] {image_path.name}: {e}")
            continue

        # ----------------------------------------------------
        # Reject bad PnP results
        # ----------------------------------------------------

        if reprojection_rmse > max_reprojection_error:
            print(
                f"[SKIP] {image_path.name}: "
                f"reprojection RMSE = "
                f"{reprojection_rmse:.3f} px"
            )
            continue

        # ----------------------------------------------------
        # Construct T_WL from GLIM
        # ----------------------------------------------------

        T_WL = make_transform(
            lidar_rotation,
            lidar_position,
        )

        # ----------------------------------------------------
        # T_CW and T_WL:
        #
        # world -> camera
        # lidar -> world
        #
        # Therefore:
        #
        # lidar -> camera
        #
        # T_CL = T_CW @ T_WL
        # ----------------------------------------------------

        T_CL = T_CW @ T_WL

        transforms_CL.append(T_CL)
        reprojection_errors.append(reprojection_rmse)

        print(
            f"[OK] {image_path.name}: "
            f"reprojection RMSE = "
            f"{reprojection_rmse:.3f} px"
        )

    # --------------------------------------------------------
    # Check results
    # --------------------------------------------------------

    if len(transforms_CL) < 3:
        raise RuntimeError(
            f"Only {len(transforms_CL)} valid observations. "
            f"Need at least 3."
        )

    transforms_CL = np.asarray(transforms_CL)

    print()
    print(
        f"Using {len(transforms_CL)} valid observations "
        f"out of {len(image_paths)} images."
    )

    # --------------------------------------------------------
    # Average translation
    # --------------------------------------------------------

    translations = transforms_CL[:, :3, 3]

    translation_mean = np.mean(translations, axis=0)

    # --------------------------------------------------------
    # Average rotations properly on SO(3)
    # --------------------------------------------------------

    rotations = Rotation.from_matrix(
        transforms_CL[:, :3, :3]
    )

    rotation_mean = rotations.mean()

    R_CL_mean = rotation_mean.as_matrix()

    T_CL_mean = np.eye(4)
    T_CL_mean[:3, :3] = R_CL_mean
    T_CL_mean[:3, 3] = translation_mean

    # --------------------------------------------------------
    # Extract LiDAR -> Camera quaternion and translation
    # --------------------------------------------------------

    lidar_to_camera_translation = T_CL_mean[:3, 3]

    # scipy quaternion order: [qx, qy, qz, qw]
    lidar_to_camera_quaternion = Rotation.from_matrix(
        T_CL_mean[:3, :3]
    ).as_quat()

    print()
    print("LiDAR -> Camera Quaternion [qx, qy, qz, qw]:")
    print(lidar_to_camera_quaternion)

    print()
    print("LiDAR -> Camera Translation [tx, ty, tz] [m]:")
    print(lidar_to_camera_translation)



    # --------------------------------------------------------
    # Calculate variation between observations
    # --------------------------------------------------------

    translation_errors = np.linalg.norm(
        translations - translation_mean,
        axis=1,
    )

    rotation_errors = (
        rotations.inv() * rotation_mean
    ).magnitude()

    print()
    print("========================================")
    print("CAMERA <-> LIDAR CALIBRATION")
    print("========================================")

    print()
    print("T_CL  (LiDAR -> Camera):")
    print(T_CL_mean)

    print()
    print("T_LC  (Camera -> LiDAR):")

    T_LC_mean = invert_transform(T_CL_mean)

    print(T_LC_mean)

    print()
    print(
        "Translation LiDAR -> Camera [m]:"
    )
    print(T_CL_mean[:3, 3])

    print()
    print(
        "Translation Camera -> LiDAR [m]:"
    )
    print(T_LC_mean[:3, 3])

    # Euler angles are convenient for inspection, but the
    # rotation matrix above should be used in actual software.
    euler_deg = Rotation.from_matrix(
        T_CL_mean[:3, :3]
    ).as_euler("xyz", degrees=True)

    print()
    print(
        "LiDAR -> Camera Euler XYZ [deg]:"
    )
    print(euler_deg)

    print()
    print(
        f"Mean reprojection RMSE: "
        f"{np.mean(reprojection_errors):.3f} px"
    )

    print(
        f"Median reprojection RMSE: "
        f"{np.median(reprojection_errors):.3f} px"
    )

    print(
        f"Translation std: "
        f"{np.std(translation_errors):.4f} m"
    )

    print(
        f"Rotation std: "
        f"{np.degrees(np.std(rotation_errors)):.4f} deg"
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    np.savez(
        "camera_lidar_calibration.npz",
        T_CL=T_CL_mean,
        T_LC=T_LC_mean,
        mean_reprojection_rmse=np.mean(reprojection_errors),
        median_reprojection_rmse=np.median(reprojection_errors),
        num_observations=len(transforms_CL),
    )

    print()
    print(
        "Saved: camera_lidar_calibration.npz"
    )

    return T_CL_mean


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Camera-LiDAR extrinsic calibration using a "
            "known-world checkerboard and GLIM trajectory."
        )
    )

    parser.add_argument(
        "--images",
        required=True,
        help="Folder containing TIFF checkerboard images.",
    )

    parser.add_argument(
        "--trajectory",
        required=True,
        help="GLIM trajectory .txt file.",
    )

    parser.add_argument(
        "--cols",
        type=int,
        default=7,
        help="Number of checkerboard inner corners horizontally.",
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Number of checkerboard inner corners vertically.",
    )

    parser.add_argument(
        "--timezone-offset",
        type=float,
        default=9.0,
        help=(
            "Timezone offset of image filenames relative to UTC, "
            "in hours. Example: 9 for Japan local time. "
            "Default: 0."
        ),
    )

    parser.add_argument(
        "--time-offset",
        type=float,
        default=0.0,
        help=(
            "Additional timestamp offset applied to image time "
            "before looking up the GLIM trajectory. "
            "Useful for UTC/GPST or sensor clock offsets. "
            "Default: 0 seconds."
        ),
    )

    parser.add_argument(
        "--max-reprojection-error",
        type=float,
        default=3.0,
        help=(
            "Maximum allowed PnP reprojection RMSE in pixels. "
            "Default: 3.0."
        ),
    )

    args = parser.parse_args()

    calibrate(
        image_folder=args.images,
        glim_path=args.trajectory,
        inner_cols=args.cols,
        inner_rows=args.rows,
        timezone_offset_hours=args.timezone_offset,
        time_offset=args.time_offset,
        max_reprojection_error=args.max_reprojection_error,
    )


if __name__ == "__main__":
    main()