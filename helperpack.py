import numpy as np

class Rereference():
    """Re-reference the channel data.

    Args:
        Vk (array): Vector with channel data.
        Vref (array): Must be a 1D-array. Used for 'monopolar' re-referencing. Default is None.
        Vk_nb (ndarray): The neighbouring channel or channels data. Used for neighbouring referencing such
                         as 'bipolar' and 'laplacian'. Must be 1D if bipolar method is used and 2D if 
                         laplacian is used. Default is None.
        Vk_new (array): Vector with re-referenced channel data.
    """

    def __init__(self):
        self.Vk_new = None

    def monopolar(self, Vk, Vref):
        """Monopolar re-referencing of channel data uses a single reference contact as reference.
        
        Args:
            Vk (array, list): 1D vector with channel data.
            Vref (array, list): 1D vector with reference channel data.
            
        Returns:
            Vk_new (array): 1D vector with re-referenced channel data.
        """
        print("Re-referencing method: 'monopolar'")
        # Type and dimensionality check for Vk
        if not (isinstance(Vk, list) or isinstance(Vk, np.ndarray)):
            raise ValueError("Vk must be a list or numpy array.")
            return None
        elif isinstance(Vk[0], list) or isinstance(Vk[0], np.ndarray):
            raise ValueError("Vk must be a 1D array or list.")
            return None
        # Type and dimensionality check for Vref
        if not (isinstance(Vref, list) or isinstance(Vref, np.ndarray)):
            raise ValueError(
                "Vref must be a float when using method 'monopolar'.")
            return None
        elif isinstance(Vref[0], list) or isinstance(Vref[0], np.ndarray):
            raise ValueError("Vref must be a 1D array or list.")
            return None
        # Length check for Vk and Vref
        if len(Vk) != len(Vref):
            raise ValueError(
                f"Vk and Vref must have the same length, got ({len(Vk)}) and ({len(Vref)}).")
            return None
        # Ensure Vk and Vref are numpy arrays
        if isinstance(Vk, list):
            print(f"Vk is a list, converting to numpy array.")
            Vk = np.array(Vk)
        if isinstance(Vref, list):
            print(f"Vref is a list, converting to numpy array.")
            Vref = np.array(Vref)
        # Re-reference channel
        self.Vk_new = Vk - Vref
        print("Successfully re-referenced channel.")
        return self.Vk_new

    def bipolar(self, Vk, Vk_nb):
        """Bipolar re-referencing of channel data uses a neighbouring contact as reference.
        
        Args:
            Vk (array, list): 1D vector with channel data.
            Vk_nb (array, list): 1D vector with neighbouring channel data.
            
        Returns:
            Vk_new (array): 1D vector with re-referenced channel data.
        """
        print("Re-referencing method: 'bipolar'")
        # Type and dimensionality check for Vk
        if not (isinstance(Vk, list) or isinstance(Vk, np.ndarray)):
            raise ValueError("Vk must be a list or numpy array.")
            return None
        elif isinstance(Vk[0], list):
            raise ValueError("Vk must be a 1D array.")
            return None
        # Type and dimensionality check for Vk_nb
        if not (isinstance(Vk_nb, list) or isinstance(Vk_nb, np.ndarray)):
            raise ValueError("Vk_nb must be a list or numpy array.")
            return None
        elif isinstance(Vk_nb[0], list):
            raise ValueError("Vk_nb must be a 1D array.")
            return None
        # Length check for Vk and Vk_nb
        if len(Vk) != len(Vk_nb):
            raise ValueError(
                f"Vk and Vk_nb must have the same length, got ({len(Vk)}) and ({len(Vk_nb)}).")
            return None
        # Ensure Vk and Vk_nb are numpy arrays
        if isinstance(Vk, list):
            print(f"Vk is a list, converting to numpy array.")
            Vk = np.array(Vk)
        if isinstance(Vk_nb, list):
            print(f"Vk_nb is a list, converting to numpy array.")
            Vk_nb = np.array(Vk_nb)
        # Re-reference channel
        self.Vk_new = Vk - Vk_nb
        print("Successfully re-referenced channel.")
        return self.Vk_new

    def laplacian(self, Vk, Vk_nb):
        """Laplacian re-referencing of channel data uses the average of neighbouring contacts as reference.
        
        Args:
            Vk (array, list): 1D vector with channel data.
            Vk_nb (array, list): 2D vector with neighbouring channel data.
            
        Returns:
            Vk_new (array): 1D vector with re-referenced channel data.
        """
        print("Re-referencing method: 'laplacian'")
        # Type and dimensionality check for Vk
        if not (isinstance(Vk, list) or isinstance(Vk, np.ndarray)):
            raise ValueError("Vk must be a list or numpy array.")
            return None
        elif isinstance(Vk[0], list):
            raise ValueError("Vk must be a 1D array.")
            return None
        # Type and dimensionality check for Vk_nb
        if not (isinstance(Vk_nb, list) or isinstance(Vk_nb, np.ndarray)):
            raise ValueError("Vk_nb must be a list or numpy array.")
            return None
        elif not isinstance(Vk_nb[0], list):
            raise ValueError("Vk_nb must be a 2D array.")
            return None
        elif len(Vk_nb) != 2:
            raise ValueError(
                f"Vk_nb must be a 2D array with 2 rows, got ({len(Vk_nb)}) rows.")
        elif len(Vk_nb[0]) != len(Vk_nb[1]):
            raise ValueError(
                f"Vk_nb neighbouring channels must be of same length, got ({len(Vk_nb[0])}) and ({len(Vk_nb[1])}).")
            return None
        # Length check for Vk and Vk_nb
        if len(Vk) != len(Vk_nb[0]):
            raise ValueError(
                f"Vk and Vk_nb must have the same length, got ({len(Vk)}) and ({len(Vk_nb[0])}).")
            return None
        elif len(Vk) != len(Vk_nb[1]):
            raise ValueError(
                f"Vk and Vk_nb must have the same length, got ({len(Vk)}) and ({len(Vk_nb[1])}).")
            return None
        # Ensure Vk and Vk_nb are numpy arrays
        if isinstance(Vk, list):
            print(f"Vk is a list, converting to numpy array.")
            Vk = np.array(Vk)
        if isinstance(Vk_nb, list):
            print(f"Vk_nb is a list, converting to numpy array.")
            Vk_nb = np.array(Vk_nb)
        # Re-reference channel
        self.Vk_new = Vk - 0.5*(Vk_nb[0] + Vk_nb[1])
        print("Successfully re-referenced channel.")
        return self.Vk_new
