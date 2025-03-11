import numpy as np

def average_distance_with_filter(reference, estimated, threshold=5.0):
    """
    Calculate the average distance between points in the reference and estimated arrays,
    allowing the estimated array to have more values than the reference array.
    Each reference value is matched to the closest estimated value within the threshold range.
    Tracks unpaired estimated values and unmatched reference values.

    Parameters:
    - reference (list or array): The correct reference values.
    - estimated (list or array): The estimated values.
    - threshold (float): The maximum allowable distance for an estimated value to be considered valid.

    Returns:
    - tuple: (average_distance, unmatched_references, unpaired_estimated)
        - average_distance (float): Average distance between valid points, or None if no matches.
        - unmatched_references (int): Number of reference values without a match.
        - unpaired_estimated (int): Number of estimated values that were never paired.
    """
    used_indices = set()
    valid_distances = []
    unmatched_references = 0

    for ref_val in reference:
        # Find all estimated values within the threshold of the current reference value
        candidates = [
            (abs(ref_val - est_val), idx)
            for idx, est_val in enumerate(estimated)
            if idx not in used_indices and abs(ref_val - est_val) <= threshold
        ]

        if candidates:
            # Select the closest estimated value
            closest_distance, closest_index = min(candidates, key=lambda x: x[0])
            valid_distances.append(closest_distance)
            used_indices.add(closest_index)  # Mark this estimated value as used
        else:
            # No match found for this reference value
            unmatched_references += 1

    # Calculate the number of unpaired estimated values
    unpaired_estimated = len(estimated) - len(used_indices)

    # Compute average distance
    average_distance = np.mean(valid_distances) if valid_distances else None

    return average_distance, unmatched_references, unpaired_estimated



# Example usage
reference = [10, 20, 30, 40, 50]
estimated = [10, 34, 70, 50]  # Example with extra values and some outliers
threshold = 5.0

avg_distance, unmatched_refs, unpaired_est = average_distance_with_filter(reference, estimated, threshold)
print(f"Average Distance: {avg_distance}")
print(f"Unmatched References: {unmatched_refs}")
print(f"Unpaired Estimated: {unpaired_est}")
