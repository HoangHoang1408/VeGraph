# Verify in the Graph (VeGraph)

## Overview
This repository contains the implementation of the methods and experiments described in our paper ["Verify-in-the-Graph: Entity Disambiguation Enhancement for Complex Claim Verification with Interactive Graph Representation"](https://arxiv.org/abs/2305.12369).

## Installation
We use [Poetry](https://python-poetry.org) to manage dependencies and [Docker](https://docs.docker.com/build-cloud/) to host Retrieval Server and LLM Server.

1. **Install Poetry**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Clone the repository and install dependencies**
   ```bash
   git clone https://github.com/HoangHoang1408/VeGraph.git
   cd VeGraph
   poetry install
   ```

3. **Prepare the LLM Server**
   ```bash
   # Add instructions for starting the LLM server
   ```

4. **Prepare the Retrieval Server**
   ```bash
   # Add instructions for starting the retrieval server
   ```

## Data Preparation
1. **Data Download**
   ```bash
   # Add instructions for downloading necessary datasets
   ```

2. **Data Processing**
   ```bash
   # Add instructions for data preprocessing steps
   ```

## Usage
Describe how to use the main components of your implementation:

1. **Initial Processing**
   ```bash
   # Example command
   bash scripts/initial_processing_factkg.sh
   ```

2. **Entity Annotation**
   ```bash
   # Example command
   python src/annotate_in_out_entities.py
   ```

## Experiments
Describe how to reproduce the experiments from your paper:

1. **Experiment 1**
   ```bash
   # Add command
   ```

2. **Experiment 2**
   ```bash
   # Add command
   ```

## Results
Describe your main results and how to reproduce them.

## License
This project is licensed under the MIT License - see the [LICENCE](LICENCE) file for details.

## Citation
If you use this code in your research, please cite our paper:
```bibtex
@article{
  # Add your citation here
}