#!/usr/bin/env python3

import argparse
import glob
import os

import cv2
import numpy as np


def calibrate_camera(
    image_dir,
    pattern_size,
    square_size=1.0,
    output_file="camera_calibration.yml",
):
    """
    Calibrate a camera from monochrome TIFF checkerboard images.

    Parameters
    ----------
    image_dir : str
        Directory containing .tif/.tiff calibration images.
    pattern_size : tuple[int, int]
        Number of INNER checkerboard corners (columns, rows).
    square_size : float
        Physical size of one checkerboard square. The absolute value does
        not affect the intrinsic matrix, but determines the units of the
        translation vectors.
    output_file : str
        Output OpenCV YAML calibration file.

    Returns
    -------
    camera_matrix : np.ndarray
        3x3 camera intrinsic matrix.
    dist_coeffs : np.ndarray
        Distortion coefficients.
    """

    # Object points for one checkerboard view.
    #
    # Example for a 9 x 6 inner-corner board:
    #
    # (0,0,0), (1,0,0), ... (8,0,0)
    # (0,1,0), (1,1,0), ... (8,1,0)
    #
    objp = np.zeros(
        (pattern_size[0] * pattern_size[1], 3),
        np.float32,
    )

    objp[:, :2] = np.mgrid[
        0 : pattern_size[0],
        0 : pattern_size[1],
    ].T.reshape(-1, 2)

    objp *= square_size

    object_points = []
    image_points = []

    # Support both common TIFF extensions.
    image_files = sorted(
        glob.glob(os.path.join(image_dir, "*.tif"))
        + glob.glob(os.path.join(image_dir, "*.tiff"))
        + glob.glob(os.path.join(image_dir, "*.TIF"))
        + glob.glob(os.path.join(image_dir, "*.TIFF"))
    )

    if not image_files:
        raise RuntimeError(
            f"No TIFF images found in directory: {image_dir}"
        )

    image_size = None

    # More robust checkerboard detector available in modern OpenCV.
    use_sb_detector = hasattr(cv2, "findChessboardCornersSB")

    for filename in image_files:
        print(f"Processing: {filename}")

        # Load as grayscale.
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print("  WARNING: Could not read image; skipping.")
            continue

        image_size_current = (image.shape[1], image.shape[0])

        if image_size is None:
            image_size = image_size_current
        elif image_size_current != image_size:
            raise RuntimeError(
                f"Image size mismatch: {filename} has "
                f"{image_size_current}, expected {image_size}"
            )

        if use_sb_detector:
            flags = (
                cv2.CALIB_CB_EXHAUSTIVE
                | cv2.CALIB_CB_ACCURACY
                | cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            found, corners = cv2.findChessboardCornersSB(
                image,
                pattern_size,
                flags=flags,
            )
        else:
            # Fall back to the traditional detector.
            flags = (
                cv2.CALIB_CB_ADAPTIVE_THRESH
                | cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            found, corners = cv2.findChessboardCorners(
                image,
                pattern_size,
                flags,
            )

            if found:
                # Refine the detected corner positions to sub-pixel accuracy.
                criteria = (
                    cv2.TERM_CRITERIA_EPS
                    + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                )

                corners = cv2.cornerSubPix(
                    image,
                    corners,
                    winSize=(11, 11),
                    zeroZone=(-1, -1),
                    criteria=criteria,
                )

        if found:
            print("  Checkerboard found.")
            object_points.append(objp.copy())
            image_points.append(corners)
        else:
            print("  Checkerboard NOT found; skipping.")

    if len(object_points) < 3:
        raise RuntimeError(
            f"Only {len(object_points)} valid calibration images found. "
            "At least 3 are required; 10+ images with varied orientations "
            "are recommended."
        )

    print()
    print(f"Valid calibration images: {len(object_points)}")
    print(f"Image size: {image_size[0]} x {image_size[1]}")
    print("Calibrating...")

    # Estimate camera intrinsics and distortion.
    rms_error, camera_matrix, dist_coeffs, rvecs, tvecs = (
        cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
        )
    )

    print()
    print("=== Camera Calibration ===")
    print(f"RMS reprojection error: {rms_error:.6f}")
    print()
    print("Camera matrix:")
    print(camera_matrix)
    print()
    print("Distortion coefficients:")
    print(dist_coeffs.ravel())

    print("\nPer-image reprojection errors:")

    total_squared_error = 0.0
    total_points = 0

    for i in range(len(object_points)):

        projected_points, _ = cv2.projectPoints(
            object_points[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs,
        )

        # Convert both arrays to shape (N, 2)
        observed = np.asarray(image_points[i], dtype=np.float64).reshape(-1, 2)
        projected = np.asarray(projected_points, dtype=np.float64).reshape(-1, 2)

        # Euclidean error for each corner
        errors = np.linalg.norm(observed - projected, axis=1)
    
        image_rms = np.sqrt(np.mean(errors ** 2))

        total_squared_error += np.sum(errors ** 2)
        total_points += len(errors)

        print(f"{i:3d}: {image_rms:.3f} px")

    mean_rms = np.sqrt(total_squared_error / total_points)

    print()
    print(f"Overall mean reprojection error: {mean_rms:.6f} pixels")

    # Save calibration in OpenCV YAML format.
    fs = cv2.FileStorage(
        output_file,
        cv2.FILE_STORAGE_WRITE,
    )

    if not fs.isOpened():
        raise RuntimeError(
            f"Could not open output file: {output_file}"
        )

    fs.write("image_width", image_size[0])
    fs.write("image_height", image_size[1])
    fs.write("camera_matrix", camera_matrix)
    fs.write("distortion_coefficients", dist_coeffs)
    fs.write("rms_reprojection_error", rms_error)
    fs.write("mean_reprojection_error", mean_rms)

    fs.release()

    print()
    print(f"Calibration saved to: {output_file}")

    return camera_matrix, dist_coeffs


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate a camera using TIFF checkerboard images."
    )

    parser.add_argument(
        "image_dir",
        help="Directory containing monochrome TIFF calibration images.",
    )

    parser.add_argument(
        "--corners",
        nargs=2,
        type=int,
        required=True,
        metavar=("COLUMNS", "ROWS"),
        help=(
            "Number of INNER checkerboard corners. "
            "For example, a 10x7-square board has 9x6 inner corners."
        ),
    )

    parser.add_argument(
        "--square-size",
        type=float,
        default=1.0,
        help=(
            "Physical checkerboard square size. Default: 1.0. "
            "This affects extrinsic translation units, not the intrinsic matrix."
        ),
    )

    parser.add_argument(
        "--output",
        default="camera_calibration.yml",
        help="Output calibration YAML file.",
    )

    args = parser.parse_args()

    calibrate_camera(
        image_dir=args.image_dir,
        pattern_size=tuple(args.corners),
        square_size=args.square_size,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()