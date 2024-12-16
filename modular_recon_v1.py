import os
import numpy as np
import tomopy
import tifffile
from pathlib import Path
from scipy import interpolate, ndimage
import cv2
import h5py
import multiprocessing as mp

# User-defined settings
CONFIG = {
    "proj_dir": "./data",
    "folder_to_reconstruct": "projections",
    "gains_dir": "gains",
    "use_h5": False,
    "sinogram_chunk_size": 400,
    "threads": {
        "sino_gen": 16,
        "stripe_removal": 16,
        "reconstruction": 16
    },
    "center_finding": {
        "enabled": True,
        "range": 20,
        "step": 1
    },
    "hotopy": {
        "enabled": True,
        "alpha": 6.0,
        "gamma": 2.0
    },
    "reconstruction": {
        "algorithm": "gridrec",
        "filter_name": "butterworth",
        "filter_params": [0.2, 2],
    },
    "stripe_removal": {
        "enabled": True,
        "method": "remove_all_stripe",
        "params": {
            "snr": 3,
            "large_size": 51,
            "small_size": 21,
        },
    },
    "phase_retrieval": {
        "enabled": False,
        "params": {
            "pixel_size": 0.0005,
            "distance": 7,
            "energy": 14,
            "alpha": 0.0006,
        }
    },
    "output": {
        "dir": "reconstructed",
        "format": "tiff",
        "bit_depth": 32,
    }
}

# Helper Functions
def load_images(folder, ext=".tif"):
    """Load image paths from a directory."""
    return sorted(Path(folder).glob(f"*{ext}"))

def preprocess_images(images, gains, darks):
    """Apply gain and dark correction."""
    corrected = []
    for img_path in images:
        img = tifffile.imread(img_path)
        gain = tifffile.imread(gains[0])
        dark = tifffile.imread(darks[0])
        corrected_img = (img - dark) / (gain - dark)
        corrected.append(corrected_img)
    return np.array(corrected)

def stripe_removal(sinogram, config):
    """Apply stripe removal based on the configuration."""
    if config["method"] == "remove_all_stripe":
        return tomopy.remove_all_stripe(
            sinogram,
            snr=config["params"]["snr"],
            la_size=config["params"]["large_size"],
            sm_size=config["params"]["small_size"],
        )
    elif config["method"] == "remove_stripe_based_sorting":
        return tomopy.remove_stripe_based_sorting(
            sinogram, size=config["params"]["small_size"]
        )
    else:
        raise ValueError("Unsupported stripe removal method.")

def phase_retrieval(sinogram, config):
    """Apply phase retrieval to a sinogram."""
    return tomopy.retrieve_phase(
        sinogram,
        pixel_size=config["params"]["pixel_size"],
        dist=config["params"]["distance"],
        energy=config["params"]["energy"],
        alpha=config["params"]["alpha"],
    )

def reconstruct(projections, theta, center, config):
    """Perform reconstruction."""
    return tomopy.recon(
        projections,
        theta,
        center=center,
        algorithm=config["algorithm"],
        filter_name=config["filter_name"],
        filter_par=config["filter_params"],
    )

# Main Script
def main():
    # Paths and Directories
    proj_dir = Path(CONFIG["proj_dir"])
    recon_dir = proj_dir / CONFIG["output"]["dir"]
    recon_dir.mkdir(parents=True, exist_ok=True)

    # Load Projections
    projections_dir = proj_dir / CONFIG["folder_to_reconstruct"]
    projection_paths = load_images(projections_dir)
    projections = preprocess_images(projection_paths, ["gain.tif"], ["dark.tif"])

    # Angle Setup
    theta = tomopy.angles(len(projections), angle_range=(0, np.pi))

    # Stripe Removal
    if CONFIG["stripe_removal"]["enabled"]:
        projections = stripe_removal(projections, CONFIG["stripe_removal"])

    # Phase Retrieval
    if CONFIG["phase_retrieval"]["enabled"]:
        projections = phase_retrieval(projections, CONFIG["phase_retrieval"])

    # Center Finding
    # Center-Finding Logic

    # toggle center finding with the ["enabled"] key  - swap to diag
    if CONFIG["center_finding"]["enabled"]:
        # Step 1: Use PC to get an initial guess
        initial_center = tomopy.find_center_pc(projections[0], projections[-1], tol=0.25)
        print(f"Initial center from PC: {initial_center}")

    # Step 2: Fine-tune around the initial guess
        center_range = range(
            int(initial_center - CONFIG["center_finding"]["range"] // 2),
            int(initial_center + CONFIG["center_finding"]["range"] // 2),
            CONFIG["center_finding"]["step"],
        )
        for c in center_range:
            print(f"Testing center: {c}")
            # Perform reconstruction with `c` (not shown for simplicity)
    else:
    # Use a manually defined center
        center = CONFIG.get("manual_center", projections.shape[-1] // 2)
        print(f"Using predefined center: {center}")

    # Reconstruction
    reconstruction = reconstruct(projections, theta, center, CONFIG["reconstruction"])

    # Save Output
    for idx, slice_ in enumerate(reconstruction):
        out_path = recon_dir / f"recon_{str(idx).zfill(5)}.{CONFIG['output']['format']}"
        tifffile.imwrite(out_path, slice_.astype(np.float32))

if __name__ == "__main__":
    main()
