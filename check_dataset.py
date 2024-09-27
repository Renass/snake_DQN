import h5py
import matplotlib.pyplot as plt
import numpy as np

FILENAME = '/home/renas/pythonprogv2/snake_DQN/datasets/dataset_2024_09_28_06:59.h5'

def load_and_visualize_dataset(filename, index=0):
    # Open the HDF5 file
    with h5py.File(filename, 'r') as hdf:
        # Load the states and next_states datasets
        states = np.array(hdf['states'])
        next_states = np.array(hdf['next_states'])

    # Ensure the index is within bounds
    if index >= len(states):
        print(f"Index out of range. The dataset contains {len(states)} entries.")
        return
    
    # Extract the state and next_state at the specified index
    state = states[index]  # Shape: [3, 30, 30] (channels first)
    next_state = next_states[index]  # Shape: [3, 30, 30] (channels first)

    # Transpose to [30, 30, 3] for plotting (channels last for matplotlib)
    state = np.transpose(state, (1, 2, 0))
    next_state = np.transpose(next_state, (1, 2, 0))

    # Plot the state and next_state as RGB images
    plt.figure(figsize=(10, 5))

    # Plot the current state
    plt.subplot(1, 2, 1)
    plt.title("State")
    plt.imshow(state.astype(np.uint8))  # Convert to uint8 for displaying as an image
    plt.axis('off')  # Hide axis for better visualization

    # Plot the next state
    plt.subplot(1, 2, 2)
    plt.title("Next State")
    plt.imshow(next_state.astype(np.uint8))  # Convert to uint8 for displaying as an image
    plt.axis('off')

    # Display the images
    plt.show()

if __name__ == '__main__':
    index = 1  # Change index to view different state-next_state pairs
    load_and_visualize_dataset(FILENAME, index=index)

