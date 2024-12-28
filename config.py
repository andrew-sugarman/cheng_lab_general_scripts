config = {
    # Input/Output Paths
    "proj_dir": "./projections",
    "gain_path": "./gains/gain.tif",
    "gain_path_post": "./gains/gain_post.tif",
    "dark_path": "./gains/dark.tif",
    "dark_path_post": "./gains/dark_post.tif",
    "log_path": "./log.txt",

    # Reconstruction Parameters
    "sinogram_chunk_size": 100,
    "threads_to_use": {
        "sino_gen": 16,
        "stripe_removal": 16,
        "recon": 16
    },
    "center_of_rotation": 1292.5,
    "reconstruction_algorithm": "gridrec",  # Options: gridrec, astra

    # Stripe Removal
    "stripe_removal": {
        "enabled": True,
        "method": "filtering",  # Options: filtering, sorting, interpolation
        "parameters": {
            "sigma": 3,
            "size": 5
        }
    },

    # Skew Correction
    "skew_correction": {
        "enabled": True,
        "angle": 0  # Angle in degrees
    },

    # Movement Correction
    "movement_correction": {
        "enabled": True,
        "y_shift": 1,
        "x_shift": 0
    },

    # Other Options
    "full_recon": False,
    "phase_retrieval": False
}

