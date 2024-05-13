import h5py

# Open the file
with h5py.File('./cache/cwgangp_cyberAttack_critic_model_weights_step_100.h5', 'r') as file:
    # Iterate through each key in the file
    for key in file.keys():
        print("Inspecting key:", key)
        item = file[key]

        # Check if the item is a group and list its contents
        if isinstance(item, h5py.Group):
            print("  Group contents:", list(item.keys()))
            for subkey in item.keys():
                subitem = item[subkey]
                # Check further subkeys, here you might want to handle more layers of nesting
                if isinstance(subitem, h5py.Dataset):
                    print(f"    Dataset {subkey} shape: {subitem.shape}")
                elif isinstance(subitem, h5py.Group):
                    print(f"    Group {subkey} further contents: {list(subitem.keys())}")

        # Check if the item is a dataset and print its shape
        elif isinstance(item, h5py.Dataset):
            print("  Dataset shape:", item.shape)