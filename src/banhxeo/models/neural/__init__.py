import torch
import torch.nn as nn

from banhxeo import GPU_DEVICE
from banhxeo.models import LanguageModel, ModelConfig


class NeuralLanguageModel(LanguageModel, nn.Module):
    def __init__(self, model_config: ModelConfig):
        LanguageModel.__init__(self, model_config)
        nn.Module.__init__(self)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Neural models must implement the 'forward' pass.")

    def predict(self, processed_input_data, **kwargs):
        self.eval()
        with torch.inference_mode(mode=True):  # same as no_grad
            raise NotImplementedError(
                "Subclasses should implement user-friendly prediction."
            )

    def get_all_layer_name(self):
        return self.named_children()

    @torch.inference_mode(mode=True)
    def get_layer_output(self, layer_name: str, input_data_batch):
        """
        Attempts to get the output of a specific named layer.
        This requires the model to have named its layers in a retrievable way
        (e.g., as attributes like self.embedding, self.rnn_layer)
        """
        self.eval()
        try:
            target_layer = dict(self.get_all_layer_name())[layer_name]
            # More robust: use forward hooks if you want to capture output from any nn.Module
            output_capture = {}

            def hook_fn(module, input, output):
                output_capture["output"] = output

            handle = target_layer.register_forward_hook(hook_fn)
            _ = self.forward(input_data_batch)  # Run a forward pass to trigger the hook
            handle.remove()  # Clean up the hook
            return output_capture.get("output")
        except KeyError:
            print(f"Error: Layer '{layer_name}' not found as a direct child module.")
            return None
        except Exception as e:
            print(f"Error getting layer output for '{layer_name}': {e}")
            return None

    def to_gpu(self):
        if GPU_DEVICE is None:
            raise
        self.to(GPU_DEVICE)
