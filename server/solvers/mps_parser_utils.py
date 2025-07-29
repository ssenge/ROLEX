import logging
from typing import Optional, Tuple

from pysmps import smps_loader as smps

logger = logging.getLogger(__name__)

def get_mps_dimensions(mps_file_path: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parses an MPS file to extract the number of variables and constraints.

    Args:
        mps_file_path: The absolute path to the MPS file.

    Returns:
        A tuple containing (num_variables, num_constraints). Returns (None, None)
        if the file cannot be parsed or an error occurs.
    """
    try:
        parse_start_time = time.time()
        model_data = smps.load_mps(mps_file_path)
        parse_end_time = time.time()
        logger.info(f"smps.load_mps took {parse_end_time - parse_start_time:.4f} seconds for {mps_file_path}")

        num_variables = len(model_data.get('col_names', []))
        num_constraints = len(model_data.get('row_names', []))

        return num_variables, num_constraints

    except FileNotFoundError:
        logger.error(f"MPS file not found: {mps_file_path}")
        return None, None
    except Exception as e:
        logger.error(f"Error parsing MPS file {mps_file_path}: {e}")
        return None, None
