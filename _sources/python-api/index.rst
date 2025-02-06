.. _pytorch_api:

Python API
==========




.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Core Modules

   torch
   tensors
   tensor_attributes
   tensor_view
   torch.autograd <autograd>
   size
   storage


.. toctree::
   :maxdepth: 1
   :caption: Neural Networks
   
   nn
   nn.functional
   nn.init
   nn.attention

.. toctree::
   :maxdepth: 1
   :caption: Device Management

   cuda
   cpu
   mps
   xpu
   accelerator
   mtia
   mtia.memory
   meta
   torch.cuda.memory <torch_cuda_memory>

.. toctree::
   :maxdepth: 1
   :caption: Data Handling

   torch.utils.data <data>
   torch.utils.dlpack <dlpack>
   torch.utils.model_zoo <model_zoo>

.. toctree::
   :maxdepth: 1
   :caption: Distributed Computing
   torch.distributed <distributed>
   torch.distributed.tensor <distributed.tensor>
   torch.distributed.algorithms.join <distributed.algorithms.join>
   torch.distributed.elastic <distributed.elastic>
   torch.distributed.fsdp <fsdp>
   torch.distributed.fsdp.fully_shard <distributed.fsdp.fully_shard>
   torch.distributed.tensor.parallel <distributed.tensor.parallel>
   torch.distributed.pipelining <distributed.pipelining>
   torch.distributed.checkpoint <distributed.checkpoint>
   rpc

.. toctree::
   :maxdepth: 1
   :caption: Optimization
   
   optim
   torch.distributed.optim <distributed.optim>
   
.. toctree::
   :maxdepth: 1
   :caption: Optimization and Compilation

   torch.compile <torch.compiler>
   torch.jit <jit>
   torch.fx
   torch.fx.experimental
   torch.func <func>
   
.. toctree::
   :maxdepth: 1
   :caption: Utilities
   torch.utils <utils>
   torch.utils.benchmark <benchmark_utils>
   torch.utils.bottleneck <bottleneck>
   torch.utils.checkpoint <checkpoint>
   torch.utils.cpp_extension <cpp_extension>
   torch.utils.deterministic <deterministic>
   torch.utils.jit <jit_utils>
   torch.utils.mobile_optimizer <mobile_optimizer>
   torch.utils.tensorboard <tensorboard>
   torch.utils.module_tracker <module_tracker>

.. toctree::
   :maxdepth: 1
   :caption: Specialized Modules
   torch.amp <amp>
   torch.library <library>
   torch.fft <fft>
   futures
   torch.hub <hub>
   torch.linalg <linalg>
   torch.monitor <monitor>
   torch.signal <signal>
   torch.special <special>
   torch.overrides
   torch.package <package>
   profiler
   onnx
   torch.random <random>
   masked
   torch.nested <nested>
   sparse
   torch.testing <testing>

.. toctree::
   :maxdepth: 1
   :caption: Additional Topics
   complex_numbers
   ddp_comm_hooks
   quantization
   type_info
   named_tensor
   name_inference
   torch.__config__ <config_mod>
   torch.__future__ <future_mod>
   logging
   torch_environment_variables
