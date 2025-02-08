.. PyTorch documentation master file, created by
   sphinx-quickstart on Fri Dec 23 13:31:47 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/pytorch/pytorch

PyTorch documentation
===================================

**PyTorch** is an open-source machine learning library that excels in deep
learning on GPUs, offering a dynamic computational graph for flexible
model building and real-time graph modification. PyTorch provides
efficient tensor computation with strong GPU acceleration, enabling
the handling of large-scale data.

.. mermaid::

    graph TD;
    A[PyTorch] --> B[Tensors]
    A --> C[Autograd]
    A --> D[Neural Networks]
    A --> E[Optimizers]
    B --> F[Operations]
    C --> G[Backward Propagation]
    D --> H[Modules]
    D --> I[Layers]
    E --> J[SGD]
    E --> K[Adam]
    F --> L[Matrix Multiplication]
    F --> M[Element-wise Operations]
    H --> N[Sequential]
    H --> O[Custom Modules]

Beginner's Essentials
~~~~~~~~~~~~~~~~~~~~~

.. grid:: 3
   :gutter: 3 3 5 3
   
   .. grid-item-card:: Get Started
      :link: https://pytorch.org/get-started/locally/

      Visit the **PyTorch Get Started** page to find the appropriate
      installation command for your operating system and environment.


   .. grid-item-card:: Learn the Basics
      :link: https://pytorch.org/tutorials/beginner/basics/intro.html

      Familiarize yourself with PyTorch's core concepts by
      exploring the Learn the **Basics tutorial series**, which
      cover tensors, autograd, and building neural networks.

   .. grid-item-card:: Build Neural Networks
      :link: https://pytorch.org/tutorials/beginner/basics/intro.html

      Dive into the **Neural Networks** tutorial to learn how to
      define and train models using PyTorch's torch.nn module.

   .. grid-item-card:: Python API
      :link: torch_api
      :link-type: ref

      Explore the **PyTorch Python API Documentation** to get detailed
      information on all available classes, functions, and modules.

Building Proficiency
~~~~~~~~~~~~~~~~~~~~

.. grid:: 3
   :gutter: 3 3 5 3

   .. grid-item-card:: Distributed Training
      :link: https://pytorch.org/docs/stable/distributed.html
      :link-type: url

      Learn how to scale your models across multiple GPUs and machines
      using PyTorch's distributed training capabilities.


   .. grid-item-card:: CUDA and GPU Acceleration
      :link: https://pytorch.org/docs/stable/notes/cuda.html
      :link-type: url

      Explore how to leverage CUDA for GPU acceleration to
      significantly speed up model training and inference.

   .. grid-item-card:: Performance Optimization
      :link: Discover how to use torch.compile to optimize model
      performance by compiling PyTorch models for faster execution.
      :link-type: url

      Discover how to use torch.compile to optimize model
      performance by compiling PyTorch models for faster execution.

Deep Dive
~~~~~~~~~

.. grid:: 3
   :gutter: 3 3 5 3

   .. grid-item-card:: Features for Large-scale deployments
      :link: https://pytorch.org/docs/stable/notes/large_scale_deployments.html
      :link-type: url

      Learn about how to use PyTorch at scale within a larger system.

Contributing to PyTorch
~~~~~~~~~~~~~~~~~~~~~~~

Ready to contribute? Learn about the process of contributing to the
PyTorch codebase on our `Community page <https://pytorch.org/docs/stable/community/design.html>`__
and in the `Ultimate Guide to PyTorch Contributions <https://github.com/pytorch/pytorch/wiki/The-Ultimate-Guide-to-PyTorch-Contributions>`__.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Python API
   :hidden:

   python-api/index
    

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Developer Notes
   :hidden:

   notes/index
 
.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Community
   :hidden:

   community/index
    
.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Language Bindings
   :hidden:

   lang-bindings/index

