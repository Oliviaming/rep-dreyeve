import h5py

def patch_checkpoint(checkpoint_path):
    """
    Patches a checkpoint file to remove the 'output_dim' parameter from the EltWiseProduct layer.
    :param checkpoint_path: Path to the checkpoint file.
    """
    with h5py.File(checkpoint_path, 'r+') as f:
        # Navigate to the model configuration
        model_config = f.attrs['model_config']
        if isinstance(model_config, bytes):
            model_config = model_config.decode('utf-8')
        
        # Load the model configuration as a dictionary
        import json
        model_config = json.loads(model_config)

        # Locate the EltWiseProduct layer(s) and remove 'output_dim'
        for layer in model_config['config']['layers']:
            if layer['class_name'] == 'EltWiseProduct':
                if 'config' in layer and 'output_dim' in layer['config']:
                    del layer['config']['output_dim']

        # Save the modified configuration back to the checkpoint
        f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')

# Example usage
checkpoint_path = 'checkpoints/e3cfa94f-cc94-4098-88ce-8b1656ce43e7/weights.mlnet.05-0.0052.h5'
patch_checkpoint(checkpoint_path)
print(f"Patched checkpoint: {checkpoint_path}")