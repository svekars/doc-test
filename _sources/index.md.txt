# PyTorch documentation

**PyTorch** is an open-source machine learning library that excels in deep
learning on GPUs, offering a dynamic computational graph for flexible
model building and real-time graph modification. PyTorch provides
efficient tensor computation with strong GPU acceleration, enabling
the handling of large-scale data.

```{mermaid}
graph TD;
    A(["PyTorch"]) --> B["Tensors"]
    A --> C["JIT (Just-in-Time)"]
    A --> D["Torch Compile"]
    A --> E["Inductor"]
    A --> F["Torch Export"]
    A --> G["Executorch"]

    subgraph Components [Components]
        direction LR
        B
        C
        D
        E
        F
        G
    end

    subgraph Functions [What they do]
        direction LR
        H(["Multi-dimensional matrices for data storage and operations"])
        M(["Compiles PyTorch models for optimization and execution"])
        R(["Converts PyTorch models to optimized native code"])
        U(["Generates efficient code from Torch IR"])
        Y(["Captures graph for non-Python execution"])
        AA(["Executes models on edge devices efficiently"])
    end

    B --> H
    C --> M
    D --> R
    E --> U
    F --> Y
    G --> AA

    classDef topNode fill:#F9BFB4,stroke:#DE3412,stroke-width:2px,color:#DE3412,font-weight:bold,font-size:25px,fill-opacity:0.2;
    classDef middleRow fill:#F3E8FB,stroke:#812CE5,stroke-width:2px,color:#812CE5,font-weight:bold,font-size:18px,fill-opacity:0.5,rx:10,ry:10;
    classDef lastRow fill:none,stroke:#812CE5,stroke-width:0px,color:#B932CC,font-weight:bold,font-size:18px,fill-opacity:0.1,text-align:center;
    classDef title fill:none,stroke:none,color:#333,font-weight:bold,font-size:22px;


    class A topNode;
    class B,C,D,E,F,G middleRow;
    class H,M,R,U,Y,AA lastRow;

    linkStyle default stroke:#BBBBBB,stroke-width:2px;

    
    %% Change the background color of the Components subgraph
    style Components fill:#E9D8FD,stroke:#812CE5,stroke-width:2px,fill-opacity:0.1;

    %% Change the background color of the Functions subgraph
    style Functions fill:#B932CC,stroke:#B932CC,stroke-width:0px,fill-opacity:0.03;
```

## Beginner's Essentials


::::{grid} 3
:gutter: 3 3 5 3

:::{grid-item-card} Get Started
:link: https://pytorch.org/get-started/locally/

Visit the **PyTorch Get Started** page to find the appropriate installation command for your operating system and environment.
:::

:::{grid-item-card} Learn the Basics
:link: https://pytorch.org/tutorials/beginner/basics/intro.html

Familiarize yourself with PyTorch's core concepts by exploring the Learn the **Basics tutorial series**, which cover tensors, autograd, and building neural networks.
:::

:::{grid-item-card} Build Neural Networks
:link: https://pytorch.org/tutorials/beginner/basics/intro.html

Dive into the **Neural Networks** tutorial to learn how to define and train models using PyTorch's torch.nn module.
:::

:::{grid-item-card} Python API
:link: torch_api
:link-type: ref

Explore the **PyTorch Python API Documentation** to get detailed information on all available classes, functions, and modules.
:::
::::


## Building Proficiency

::::{grid} 3
:gutter: 3 3 5 3

:::{grid-item-card} Distributed Training
:link: https://pytorch.org/docs/stable/distributed.html
:link-type: url

Learn how to scale your models across multiple GPUs and machines using PyTorch's distributed training capabilities.
:::

:::{grid-item-card} CUDA and GPU Acceleration
:link: https://pytorch.org/docs/stable/notes/cuda.html
:link-type: url

Explore how to leverage CUDA for GPU acceleration to significantly speed up model training and inference.
:::

:::{grid-item-card} Performance Optimization
:link: Discover how to use torch.compile to optimize model performance by compiling PyTorch models for faster execution.
:link-type: url

Discover how to use torch.compile to optimize model performance by compiling PyTorch models for faster execution.
::::

## Deep Dive

::::{grid} 3
:gutter: 3 3 5 3

:::{grid-item-card} Features for Large-scale deployments
:link: https://pytorch.org/docs/stable/notes/large_scale_deployments.html
:link-type: url

Learn about how to use PyTorch at scale within a larger system.
:::
::::


## Contributing to PyTorch

Ready to contribute? Learn about the process of contributing to the
PyTorch codebase on our [Community page](https://pytorch.org/docs/stable/community/design.html)
and in the [Ultimate Guide to PyTorch Contributions](https://github.com/pytorch/pytorch/wiki/The-Ultimate-Guide-to-PyTorch-Contributions).

```{toctree}
:glob:
:maxdepth: 1
:caption: Python API
:hidden:

python-api/index
```    

```{toctree}
:glob:
:maxdepth: 1
:caption: Developer Notes
:hidden:

notes/index
``` 
