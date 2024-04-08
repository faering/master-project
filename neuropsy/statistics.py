import numpy as np


def get_shuffle_indices(num_obs, rng):
    """get_shuffle_indices Shuffle observations between two paired samples and return array of indices
        that have been shuffled.

    Args:
        num_obs (int): Number of observations in each sample.
        seed (int): Seed for random number generator. Defaults to None.

    Returns:
        indices: Boolean array with True at indices that have been shuffled.
    """
    arr_a = [1] * num_obs
    arr_b = [0] * num_obs

    # Combine arr_a and arr_b into a single 2D NumPy array
    combined = np.column_stack((arr_a, arr_b))

    # Shuffle the indices of the rows within each pair
    for pair in combined:
        rng.shuffle(pair)

    indices = np.where(combined[:, 0] == 0)[0]
    return indices


def assign_cluster_labels(clusters_found: np.ndarray, verbose: bool = False) -> np.ndarray:
    """assign_cluster_labels Forward iteration to assign cluster labels to clusters found in the data.

    Args:
        clusters_found (np.ndarray): 1D or 2D array of clusters found in the data.

    Returns:
        cluster_labels (np.ndarray): 1D or 2D array - depending on input array - with unique cluster labels.
    """
    if clusters_found.ndim != 1 and clusters_found.ndim != 2:
        raise ValueError(
            f"Input array must be 1D or 2D, got {clusters_found.ndim}D.")

    ndim = clusters_found.ndim

    # intialise output
    arr_shape = clusters_found.shape
    if ndim == 1:
        cluster_labels = np.reshape(([None] * arr_shape[0]), arr_shape)
    elif ndim == 2:
        cluster_labels = np.reshape(
            ([None] * arr_shape[0] * arr_shape[1]), arr_shape)

    first_run = True
    label_increment = 1
    continue_labelling = True
    cluster_labels_before = cluster_labels.copy()
    itercounter = 0
    while continue_labelling:
        for i in range(arr_shape[0]):

            if ndim == 1:
                # get neighbours
                if first_run:

                    # check if current point is a cluster
                    if clusters_found[i] == 1:
                        # first run, check if any of the previous neighbours have been labelled
                        if i == 0:
                            xlo = None
                            xhi = None
                        elif i == arr_shape[0] - 1:
                            xlo = cluster_labels[i-1]
                            xhi = None
                        else:
                            xlo = cluster_labels[i-1]
                            xhi = None

                        # check if any of the prior neighbours, in relation to forward iteration, have been labelled
                        if xlo is not None:
                            cluster_labels[i] = int(xlo)
                        else:
                            cluster_labels[i] = label_increment
                            label_increment += 1

                # after first run, get all surrounding neighbours [FIXME] maybe remove, since 1D one iteration should be sufficient
                else:
                    # check if current point is a cluster
                    if cluster_labels[i] is not None:

                        if i == 0:
                            xlo = None
                            xhi = cluster_labels[i+1]
                        elif i == arr_shape[0] - 1:
                            xlo = cluster_labels[i-1]
                            xhi = None
                        else:
                            xlo = cluster_labels[i-1]
                            xhi = cluster_labels[i+1]

                        # check if any of the neighbours have been labelled
                        # if any neighbours then set to the minimum value of the neigbouring clusters
                        # note:
                        #   this should make sure overlapping clusters are propageted toward being the same cluster after N iterations
                        if xlo is not None and xhi is not None:
                            cluster_labels[i] = int(np.min([xlo, xhi]))
                        elif xlo is not None:
                            cluster_labels[i] = int(xlo)
                        elif xhi is not None:
                            cluster_labels[i] = int(xhi)
                        else:
                            # cluster label will be the same as before
                            pass

            if ndim == 2:
                for j in range(arr_shape[1]):

                    # get neighbours
                    if first_run:

                        # check if current point is a cluster
                        if clusters_found[i, j] == 1:
                            # first run, check if any of the previous neighbours have been labelled
                            if i == 0 and j == 0:
                                xlo = None
                                xhi = None
                                ylo = None
                                yhi = None
                            elif i == 0 and j == arr_shape[1] - 1:
                                xlo = cluster_labels[i, j-1]
                                xhi = None
                                ylo = None
                                yhi = None
                            elif i == arr_shape[0] - 1 and j == 0:
                                xlo = None
                                xhi = None
                                ylo = None
                                yhi = cluster_labels[i-1, j]
                            elif i == arr_shape[0] - 1 and j == arr_shape[1] - 1:
                                xlo = cluster_labels[i, j-1]
                                xhi = None
                                ylo = None
                                yhi = cluster_labels[i-1, j]
                            elif i == 0:
                                xlo = cluster_labels[i, j-1]
                                xhi = None
                                ylo = None
                                yhi = None
                            elif i == arr_shape[0] - 1:
                                xlo = cluster_labels[i, j-1]
                                xhi = None
                                ylo = None
                                yhi = cluster_labels[i-1, j]
                            elif j == 0:
                                xlo = None
                                xhi = None
                                ylo = None
                                yhi = cluster_labels[i-1, j]
                            elif j == arr_shape[1] - 1:
                                xlo = cluster_labels[i, j-1]
                                xhi = None
                                ylo = None
                                yhi = cluster_labels[i-1, j]
                            else:
                                xlo = cluster_labels[i, j-1]
                                xhi = None
                                ylo = None
                                yhi = cluster_labels[i-1, j]

                            # check if any of the prior neighbours, in relation to forward iteration, have been labelled
                            if xlo is not None and yhi is not None:
                                cluster_labels[i, j] = int(np.min([xlo, yhi]))
                            elif xlo is not None:
                                cluster_labels[i, j] = int(xlo)
                            elif yhi is not None:
                                cluster_labels[i, j] = int(yhi)
                            else:
                                cluster_labels[i, j] = label_increment
                                label_increment += 1

                    # after first run, get all surrounding neighbours
                    else:
                        # check if current point is a cluster
                        if cluster_labels[i, j] is not None:

                            if i == 0 and j == 0:
                                xlo = None
                                xhi = cluster_labels[i, j+1]
                                ylo = cluster_labels[i+1, j]
                                yhi = None
                            elif i == 0 and j == arr_shape[1] - 1:
                                xlo = cluster_labels[i, j-1]
                                xhi = None
                                ylo = cluster_labels[i+1, j]
                                yhi = None
                            elif i == arr_shape[0] - 1 and j == 0:
                                xlo = None
                                xhi = cluster_labels[i, j+1]
                                ylo = None
                                yhi = cluster_labels[i-1, j]
                            elif i == arr_shape[0] - 1 and j == arr_shape[1] - 1:
                                xlo = cluster_labels[i, j-1]
                                xhi = None
                                ylo = None
                                yhi = cluster_labels[i-1, j]
                            elif i == 0:
                                xlo = cluster_labels[i, j-1]
                                xhi = cluster_labels[i, j+1]
                                ylo = cluster_labels[i+1, j]
                                yhi = None
                            elif i == arr_shape[0] - 1:
                                xlo = cluster_labels[i, j-1]
                                xhi = cluster_labels[i, j+1]
                                ylo = None
                                yhi = cluster_labels[i-1, j]
                            elif j == 0:
                                xlo = None
                                xhi = cluster_labels[i, j+1]
                                ylo = cluster_labels[i+1, j]
                                yhi = cluster_labels[i-1, j]
                            elif j == arr_shape[1] - 1:
                                xlo = cluster_labels[i, j-1]
                                xhi = None
                                ylo = cluster_labels[i+1, j]
                                yhi = cluster_labels[i-1, j]
                            else:
                                xlo = cluster_labels[i, j-1]
                                xhi = cluster_labels[i, j+1]
                                ylo = cluster_labels[i+1, j]
                                yhi = cluster_labels[i-1, j]

                            # check if any of the neighbours have been labelled
                            # if any neighbours then set to the minimum value of the neigbouring clusters
                            # note:
                            #   this should make sure overlapping clusters are propageted toward being the same cluster after N iterations
                            if xlo is not None and xhi is not None and ylo is not None and yhi is not None:
                                cluster_labels[i, j] = int(
                                    np.min([xlo, xhi, ylo, yhi]))
                            elif xlo is not None and xhi is not None and ylo is not None:
                                cluster_labels[i, j] = int(
                                    np.min([xlo, xhi, ylo]))
                            elif xlo is not None and xhi is not None and yhi is not None:
                                cluster_labels[i, j] = int(
                                    np.min([xlo, xhi, yhi]))
                            elif xlo is not None and ylo is not None and yhi is not None:
                                cluster_labels[i, j] = int(
                                    np.min([xlo, ylo, yhi]))
                            elif xhi is not None and ylo is not None and yhi is not None:
                                cluster_labels[i, j] = int(
                                    np.min([xhi, ylo, yhi]))
                            elif xlo is not None and xhi is not None:
                                cluster_labels[i, j] = int(np.min([xlo, xhi]))
                            elif xlo is not None and ylo is not None:
                                cluster_labels[i, j] = int(np.min([xlo, ylo]))
                            elif xlo is not None and yhi is not None:
                                cluster_labels[i, j] = int(np.min([xlo, yhi]))
                            elif xhi is not None and ylo is not None:
                                cluster_labels[i, j] = int(np.min([xhi, ylo]))
                            elif xhi is not None and yhi is not None:
                                cluster_labels[i, j] = int(np.min([xhi, yhi]))
                            elif ylo is not None and yhi is not None:
                                cluster_labels[i, j] = int(np.min([ylo, yhi]))
                            elif xlo is not None:
                                cluster_labels[i, j] = int(xlo)
                            elif xhi is not None:
                                cluster_labels[i, j] = int(xhi)
                            elif ylo is not None:
                                cluster_labels[i, j] = int(ylo)
                            elif yhi is not None:
                                cluster_labels[i, j] = int(yhi)
                            else:
                                # cluster label will be the same as before
                                pass

        # after full run
        first_run = False
        label_increment = None

        # check if cluster label matrix is the same after this iteration as it was before
        # and stop while loop if it is the same, otherwise continue until all labels have been assigned
        # correctly.
        if np.array_equal(cluster_labels, cluster_labels_before):
            # print(f"Assignation of cluster labels has converged. Stopping while loop.")
            if verbose:
                print(f"Assignation of cluster labels has converged.")
            continue_labelling = False

            # convert all None to zeros
            if ndim == 1:
                cluster_labels = np.array(
                    [i if i is not None else 0 for i in cluster_labels]).astype(int)
            elif ndim == 2:
                cluster_labels = np.array(
                    [i if i is not None else 0 for sublist in cluster_labels for i in sublist]).reshape(arr_shape).astype(int)

            # corner case:
            # note:
            #   if all points are assigned to a cluster, then the cluster labels will be 1, 2, 3, 4, etc. (and not 0, 1, 2, 3, etc.)
            if 0 not in cluster_labels:
                unique_labels = np.unique(cluster_labels)
            else:
                unique_labels = np.unique(cluster_labels)[1:]

            # correct cluster labels if they are not 1, 2, 3, 4, etc.
            if verbose:
                print(f"Assigned cluster labels: {unique_labels}")
            for i in range(1, len(unique_labels)+1):
                if i in cluster_labels:
                    continue
                else:
                    cluster_labels[cluster_labels == unique_labels[i-1]] = i

            if 0 not in cluster_labels:
                unique_labels_corrected = np.unique(cluster_labels)
            else:
                unique_labels_corrected = np.unique(cluster_labels)[1:]

            if np.array_equal(unique_labels, unique_labels_corrected):
                if verbose:
                    print(f"Cluster labels are correct: {unique_labels}")
            else:
                if verbose:
                    print(
                        f"Cluster labels have been corrected to: {unique_labels_corrected}")

            return cluster_labels
        else:
            cluster_labels_before = cluster_labels.copy()
            itercounter += 1
            if verbose:
                print(f"Iteration {itercounter} completed, continuing.")
            continue


def get_cluster_sizes(t_statistics: np.array, cluster_labels: np.array, verbose: bool = False) -> dict:
    """get_cluster_sizes Compute cluster sizes based on t-statistics and cluster labels.

    Args:
        t_statistics (np.array): The t-values from t-test.
        cluster_labels (np.array): The cluster labels assigned to the clusters found in the data.

    Returns:
        cluster_sizes (dict): Dictionary with cluster labels as keys and cumulative t-values as values.
    """
    if t_statistics.shape != cluster_labels.shape:
        raise ValueError(
            "t-statistics and cluster labels must have the same shape.")

    ndim = cluster_labels.ndim

    # initialise output
    arr_shape = t_statistics.shape

    if 0 not in cluster_labels:
        unique_labels = np.unique(cluster_labels).astype(int)
    else:
        unique_labels = np.unique(cluster_labels).astype(int)[
            1:]  # ignore zero

    # initialise dict for storing cumulative cluster t-values
    cluster_sizes = {str(label): 0 for label in unique_labels}

    if ndim == 1:
        for i in range(arr_shape[0]):
            # check if current point is a cluster
            if cluster_labels[i] != 0:
                # increment the cluster size by the t-value
                cluster_sizes[str(cluster_labels[i])] += t_statistics[i]
    elif ndim == 2:
        for i in range(arr_shape[0]):
            for j in range(arr_shape[1]):
                # check if current point is a cluster
                if cluster_labels[i, j] != 0:
                    # increment the cluster size by the t-value
                    cluster_sizes[str(cluster_labels[i, j])
                                  ] += t_statistics[i, j]

    # round values to 2 decimal places
    cluster_sizes = {key: round(value, 2)
                     for key, value in cluster_sizes.items()}

    if verbose:
        for label in unique_labels:
            print(f"\tCluster {label} size: {cluster_sizes[str(label)]}")

    return cluster_sizes
