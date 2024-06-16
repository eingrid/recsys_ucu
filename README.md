
# UCU Recommendation Systems

This project is part of the Recommender System course at UCU. It utilizes the MovieLens 1M dataset as a foundation for learning and experimentation.

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

### Installation and Data Retrieval

To get started, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/eingrid/recsys_ucu.git
    cd recsys_ucu
    ```
2. **Run the `get_data.sh` script** to download the required data:
    ```bash
    ./scripts/get_data.sh
    ```
3. **Install the Conda environment** using the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```
4. **Activate the Conda environment**:
    ```bash
    conda activate recsys_ucu
    ```
5. You are now ready to run the notebooks. Open your desired notebook and execute the cells.

## Notebooks

You can find all notebooks in the `experiments` folder. Each notebook is designed to be self-contained and can be run independently. Feel free to explore and modify the notebooks according to your needs.

## License

This project is licensed under the [MIT License](LICENSE).
