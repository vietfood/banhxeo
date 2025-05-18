from typing import Optional

import torch
import torch.nn as nn

from banhxeo.models import LanguageModel, ModelConfig


class NeuralLanguageModel(LanguageModel, nn.Module):
    def __init__(self, model_config: ModelConfig):
        LanguageModel.__init__(self, model_config)
        nn.Module.__init__(self)

    def forward(self, *args, **kwargs):
        """
        The core of a neural model. Subclasses MUST implement this.
        This is where the data flows through the layers.
        - Comment each step.
        - Show tensor shapes (optionally via debug prints).
        - Break down complex parts into smaller, understandable operations.
        """
        raise NotImplementedError("Neural models must implement the 'forward' pass.")

    def predict(self, processed_input_data, **kwargs):
        """
        This typically involves:
        1. Setting the model to evaluation mode (`self.eval()`).
        2. Disabling gradient calculations (`with torch.no_grad():`).
        3. Passing data through `self.forward()`.
        4. Post-processing outputs (e.g., applying softmax, getting argmax).
        The exact signature and output will depend on the task (classification, generation).
        """
        self.eval()
        with torch.inference_mode(mode=True):  # same as no_grad
            raise NotImplementedError(
                "Subclasses should implement user-friendly prediction."
            )

    def get_all_layer_name(self):
        return self.named_children()

    @torch.no_grad()
    def get_layer_output(self, layer_name: str, input_data_batch):
        """
        Attempts to get the output of a specific named layer.
        This requires the model to have named its layers in a retrievable way
        (e.g., as attributes like self.embedding, self.rnn_layer) or use forward hooks.
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
