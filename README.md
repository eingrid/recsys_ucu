# UCU Rec_systems

This project is a part of recommender system course in UCU. It leverages the MovieLens 1M dataset as a foundation for learning and experimentation.

## Directory Structure

```plaintext
recsys_ucu/
├── data/                 # Directory for storing data
├── experiments/          # Directory containing Jupyter notebooks
├── scripts/              # Directory for scripts such as get_data.sh
├── environment.yml       # Conda environment file
├── LICENSE               # License file
├── src/                  # Source code in .py format
└── README.md             # This README file
```

## Getting Started

### Prerequisites

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)


### Install environment and get data

To get started, follow these steps:

1. Clone the repository to your local machine.
2. Run the `get_data.sh` script to download the required data. You can use the following command:
    ```bash
    ./get_data.sh
    ```
3. Install the conda environment using the `environment.yml` file. You can use the following command:
    ```bash
    conda env create -f environment.yml
    ```
4. Activate the conda environment:
    ```bash
    conda activate recsys_ucu
    ```
5. You are now ready to run the notebooks. Open the desired notebook and execute the cells.

## Notebooks

You can find all notebooks in the experiments folder. Each notebook is designed to be self-contained and can be run independently. Feel free to explore and modify the notebooks according to your needs.


## License

This project is licensed under the [MIT License](LICENSE).