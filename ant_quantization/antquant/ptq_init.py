
import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_modules import Quantizer,Conv2dQuantizer, LinearQuantizer
from multihead_attention import MultiheadAttentionQuantizer
from quant_utils import enable_quantization, disable_quantization, disable_input_quantization


def set_quant_state_till_layer(model, layer, weight_quant: bool, act_quant: bool):
    """
    Set quant state of layers before the given layer, and disable quant for the rest.
    """
    disable_quantization(model)

    for name, module in model.named_modules():
        if module is layer:
            break
        if isinstance(module, Quantizer):
            if weight_quant:
                enable_quantization(module)
                if not act_quant:
                    disable_input_quantization(module)
            else:
                disable_quantization(module)

def register_act_func(module):
    """
    Register succesive activation function to quantized layers. 
    """
    prev_quantmodule = None
    def identity(x):
        return x
    #FIXME: use act function later

    for name, child_module in module.named_children():
        if isinstance(child_module, (
            Conv2dQuantizer,
            LinearQuantizer,
            # MultiheadAttentionQuantizer,
        )):
            prev_quantmodule = child_module
            prev_quantmodule.act_func = identity
        elif isinstance(child_module, (
            nn.ReLU,
            nn.ReLU6,
            nn.GELU,
        )):
            if isinstance(child_module, nn.ReLU):
                act_func = F.relu
            elif isinstance(child_module, nn.ReLU6):
                act_func = F.relu6
            elif isinstance(child_module, nn.GELU):
                act_func = F.gelu
            else:
                raise NotImplementedError
            if prev_quantmodule is not None:
                prev_quantmodule.act_func = act_func
        else:
            register_act_func(child_module)
            

def ptq_init_model(model: nn.Module, module: nn.Module, cali_data: torch.Tensor,
                   opt_target: str = 'tensor', opt_metric: str = 'mse', batch_size: int = 32,
                   asym: bool = False, act_quant: bool = False, include_act_func: bool = False):
    """
    Initialize quantization parameters of the model.

    :param model: top model to quantize
    :param module: current module to parse
    """
    disable_quantization(model)
    register_act_func(model)

    for name, child_module in module.named_children():
        if isinstance(child_module, (
            Conv2dQuantizer,
            LinearQuantizer,
            # MultiheadAttentionQuantizer,
        )):
            ptq_init_layer(model, child_module, cali_data, opt_target=opt_target, opt_metric=opt_metric,
                           batch_size=batch_size, asym=asym, act_quant=act_quant, 
                           include_act_func=include_act_func)
        else:
            ptq_init_model(model, child_module, cali_data, opt_target=opt_target, opt_metric=opt_metric,
                           batch_size=batch_size, asym=asym, act_quant=act_quant, 
                           include_act_func=include_act_func)
    
    enable_quantization(model)
    if not act_quant:
        disable_input_quantization(model)


def ptq_init_layer(model: nn.Module, layer: nn.Module, cali_data: torch.Tensor, 
                   batch_size: int = 32, asym: bool = False, act_quant: bool = False,
                   weight_opt_target: str = 'tensor', weight_opt_metric: str = 'mse',
                   inp_opt_target: str = 'tensor', inp_opt_metric: str = 'mse',
                   ):
    """
    Initialize quantization parameters of the layer.

    :param model: quantized model, with all previous layers inited quant params
    :param layer: current layer to initailize PTQ params
    :param cali_data: calibration data from training set
    :param batch_size: how much calibration data is fed at one time
    :param asym: if set True, enable quantized model of previous layers to collect inp
    :param act_quant: enable activation quantization for previous layers
    :param opt_target: minimize tensor error or output error
    :param opt_metric: how to determine the amount of quantization error
    """
    assert hasattr(layer, "act_func"), f" Layer {layer} doesn't register activation function"

    set_quant_state_till_layer(model, layer, weight_quant=False, act_quant=False)

    # store input and output activation
    cached_inps, cached_outs = save_inp_oup_data(model, layer, cali_data, 
                                                 asym=asym, act_quant=act_quant, batch_size=batch_size)

    # store gradient if necessary
    if weight_opt_metric != 'fisher_diag' and inp_opt_metric != 'fisher_diag':
        grad_out = save_grad_data(model, layer, cali_data, act_quant=act_quant, batch_size=batch_size)
    else:
        grad_out = None

    # init weight tensor quantizer
    # For simplicity, input tensor quantizer is not initialized when selecting weight's scaling factor.
    # When running asymmetric reconstruction, this corresponds to Case 3 in QDrop.
    quant_weight = layer.quant_weight
    quant_weight.init_quant_para(layer.weight, data_b=cached_inps, org_outs=cached_outs, 
                                 grad_data=None, grad_out=grad_out, act_func=layer.act_func,
                                 opt_target=weight_opt_target, opt_metric=weight_opt_metric)

    # init input tensor quantizer
    # To align with BRECQ and original ANT, we still use MSE.
    quant_input = layer.quant_input
    quant_input.init_quant_para(cached_inps, layer.weight, org_outs=cached_outs, 
                                grad_data=None, grad_out=grad_out, act_func=layer.act_func,
                                opt_target=inp_opt_target, opt_metric=inp_opt_metric)


# Data collection util functions copied and modified from BRECQ/quant/data_utils.py

def save_inp_oup_data(model, layer, cali_data: torch.Tensor,
                      asym: bool = False, act_quant: bool = False, batch_size: int = 32, keep_gpu: bool = True):
    """
    Save input data and output data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param layer: QuantModule or QuantBlock
    :param cali_data: calibration data set
    :param asym: if Ture, save quantized input and full precision output
    :param act_quant: use activation quantization
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: input and output data
    """
    device = next(model.parameters()).device
    get_inp_out = GetLayerInpOut(model, layer, device=device, asym=asym, act_quant=act_quant)
    cached_batches = []
    torch.cuda.empty_cache()

    for i in range(int(cali_data.size(0) / batch_size)):
        cur_inp, cur_out = get_inp_out(cali_data[i * batch_size:(i + 1) * batch_size])
        cached_batches.append((cur_inp.cpu(), cur_out.cpu()))

    cached_inps = torch.cat([x[0] for x in cached_batches])
    cached_outs = torch.cat([x[1] for x in cached_batches])
    torch.cuda.empty_cache()
    if keep_gpu:
        cached_inps = cached_inps.to(device)
        cached_outs = cached_outs.to(device)
    return cached_inps, cached_outs


def save_grad_data(model, layer, cali_data: torch.Tensor,
                   damping: float = 1., act_quant: bool = False, batch_size: int = 32,
                   keep_gpu: bool = True):
    """
    Save gradient data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param layer: QuantModule or QuantBlock
    :param cali_data: calibration data set
    :param damping: damping the second-order gradient by adding some constant in the FIM diagonal
    :param act_quant: use activation quantization
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: gradient data
    """
    device = next(model.parameters()).device
    get_grad = GetLayerGrad(model, layer, device, act_quant=act_quant)
    cached_batches = []
    torch.cuda.empty_cache()

    for i in range(int(cali_data.size(0) / batch_size)):
        cur_grad = get_grad(cali_data[i * batch_size:(i + 1) * batch_size])
        cached_batches.append(cur_grad.cpu())

    cached_grads = torch.cat([x for x in cached_batches])
    cached_grads = cached_grads.abs() + 1.0
    # scaling to make sure its mean is 1
    # cached_grads = cached_grads * torch.sqrt(cached_grads.numel() / cached_grads.pow(2).sum())
    torch.cuda.empty_cache()
    if keep_gpu:
        cached_grads = cached_grads.to(device)
    return cached_grads


class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


class GetLayerInpOut:
    def __init__(self, model, layer,
                 device: torch.device, asym: bool = False, act_quant: bool = False):
        self.model = model
        self.layer = layer
        self.asym = asym
        self.device = device
        self.act_quant = act_quant
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=True)

    def __call__(self, model_input):
        self.model.eval()
        set_quant_state_till_layer(self.model, self.layer, False, False)

        handle = self.layer.register_forward_hook(self.data_saver)
        with torch.no_grad():
            try:
                _ = self.model(model_input.to(self.device))
            except StopForwardException:
                pass

            if self.asym:
                # Recalculate input with network quantized
                self.data_saver.store_output = False
                set_quant_state_till_layer(self.model, self.layer, weight_quant=True, act_quant=self.act_quant)
                try:
                    _ = self.model(model_input.to(self.device))
                except StopForwardException:
                    pass
                self.data_saver.store_output = True

        handle.remove()

        set_quant_state_till_layer(self.model, self.layer, False, False)
        self.model.train()

        return self.data_saver.input_store[0].detach(), self.data_saver.output_store.detach()
    

class GradSaverHook:
    def __init__(self, store_grad=True):
        self.store_grad = store_grad
        self.stop_backward = False
        self.grad_out = None

    def __call__(self, module, grad_input, grad_output):
        if self.store_grad:
            self.grad_out = grad_output[0]
        if self.stop_backward:
            raise StopForwardException


class GetLayerGrad:
    def __init__(self, model, layer,
                 device: torch.device, act_quant: bool = False):
        self.model = model
        self.layer = layer
        self.device = device
        self.act_quant = act_quant
        self.data_saver = GradSaverHook(True)

    def __call__(self, model_input):
        """
        Compute the gradients of block output, note that we compute the
        gradient by calculating the KL loss between fp model and quant model

        :param model_input: calibration data samples
        :return: gradients
        """
        self.model.eval()

        handle = self.layer.register_backward_hook(self.data_saver)
        with torch.enable_grad():
            try:
                self.model.zero_grad()
                inputs = model_input.to(self.device)
                self.model.set_quant_state_till_layer(False,  False)
                out_fp = self.model(inputs)
                set_quant_state_till_layer(self.model, self.layer, weight_quant=True, act_quant=self.act_quant)
                out_q = self.model(inputs)
                loss = F.kl_div(F.log_softmax(out_q, dim=1), F.softmax(out_fp, dim=1), reduction='batchmean')
                loss.backward()
            except StopForwardException:
                pass

        handle.remove()
        set_quant_state_till_layer(self.model, self.layer, False, False)
        self.model.train()
        return self.data_saver.grad_out.data