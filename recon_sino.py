from tomopy import recon
import numpy as np
from tifffile import memmap

def reconstruct_sinograms(config, logger):
    """
    Reconstruct sinograms from processed projections using memory mapping.

    Args:
        config (dict): Configuration dictionary containing paths and parameters.
        logger (Logger): Logger object for recording reconstruction steps.
    """
    # Iterate through sinogram chunks
    for chunk_idx, sinogram_chunk_path in enumerate(load_sinograms(config)):
        logger.info(f"Reconstructing sinogram chunk {chunk_idx + 1}...")

        # Load sinogram chunk using memory mapping
        sinogram_chunk = memmap(sinogram_chunk_path, mode='r')

        # Perform reconstruction
        reconstructed = recon(
            sinogram_chunk,
            algorithm=config["reconstruction_algorithm"],
            center=config["center_of_rotation"]
        )

        # Save the reconstructed images
        output_path = f"reconstructed_chunk_{chunk_idx + 1}.tif"
        save_with_memmap(output_path, reconstructed)

        logger.info(f"Saved reconstructed chunk {chunk_idx + 1} to {output_path}")


def load_sinograms(config):
    """
    Loads sinogram chunks from preprocessed data.

    Args:
        config (dict): Configuration dictionary containing paths and parameters.

    Yields:
        str: Path to a sinogram chunk.
    """
    # Logic to generate paths for sinogram chunks
    # This assumes sinograms are saved in the specified directory after preprocessing
    sinogram_dir = config["proj_dir"]  # Adjust this path as needed
    for idx in range(0, len(sinogram_dir), config["sinogram_chunk_size"]):
        yield f"{sinogram_dir}/sinogram_chunk_{idx}.tif"


def save_with_memmap(output_path, data):
    """
    Save processed data using memory mapping.

    Args:
        output_path (str): Path to save the processed data.
        data (np.ndarray): Data to save.
    """
    memmap = np.memmap(output_path, dtype=data.dtype, mode='w+', shape=data.shape)
    memmap[:] = data[:]
    memmap.flush()
