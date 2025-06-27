# StrikeSim: Union Strike Simulation Model

A computational model to simulate how union structure affects strike growth, success, and failure. This model helps unions decide whether to call a strike by predicting the development of industrial action over time.

## Overview

StrikeSim generates random networks representing the internal structures of unions and workplaces. Nodes represent people and edges represent interactions. Along these networks flows hope and fear affecting workers' willingness to participate in industrial action. The model tracks monetary flows including wages, strike pay, and employer revenue. Over time, the combination of morale and workers' financial position affects whether the strike grows or shrinks, and whether the employer cedes to demands.

## Key Features

- **Agent-Based Model**: Workers, employers, and unions as individual agents with distinct behaviors
- **Network Effects**: Social interactions influence worker morale and participation decisions
- **Financial Flows**: Complete tracking of wages, strike pay, union dues, and employer revenue
- **Morale System**: Three different morale specifications (sigmoid, linear, no-motivation)
- **Policy Simulation**: Union and employer policy changes during strikes
- **Network Loading**: Support for loading pre-defined networks from .gexf files
- **Data Export**: HDF5 and CSV export for analysis
- **Visualization**: Network and time series visualizations
- **Monte Carlo**: Support for multiple simulation runs with different random networks

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd strikesim
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Simulation

Run a single simulation:

```bash
python run.py
```

This will:
- Initialize the simulation with default parameters
- Run the strike simulation
- Generate visualizations (networks.png, time_series.png)
- Save data (strikesim_data.h5, strikesim_summary.csv)

### Monte Carlo Simulation

To run multiple simulations with different random networks, uncomment the Monte Carlo section in `run.py`:

```python
# Uncomment to run Monte Carlo simulation
mc_results = run_monte_carlo()
```

### Configuration

Modify `settings.py` to adjust model parameters:

- **Calendar**: Start date, duration, working days
- **Workers**: Number, wages, target wages, initial morale
- **Financial**: Employer balance, strike fund, revenue markup
- **Networks**: Union and employer network structure
- **Morale**: Specification type and parameters
- **Policies**: Concession thresholds, strike pay rates

## Model Components

### Agents

- **Worker**: Individual workers with morale, wages, savings, and participation history
- **Employer**: Company with balance, revenue calculation, and concession policies
- **Union**: Organization with strike fund, dues collection, and policy management

### Networks

- **Employer Network**: Hierarchical structure (executive → department → team)
- **Union Network**: Bargaining committee and worker connections with configurable density
- **Network Loading**: Option to load pre-defined networks from .gexf files in `networks/employers/` and `networks/unions/` directories

### Financial Flows

- Wages paid to working workers
- Strike pay distributed to striking workers
- Union dues collected from members
- Employer revenue based on working days and markup
- Balance tracking for all agents

### Morale System

Three specifications implemented:

1. **Sigmoid**: Non-linear response to wage gaps and savings
2. **Linear**: Linear combination of factors
3. **No-Motivation**: Simplified specification

Morale combines private factors (financial position) with social factors (network interactions).

### Network Configuration

The model supports two ways to define networks:

1. **Generated Networks** (default): Random networks generated based on parameters in `settings.py`
2. **Loaded Networks**: Pre-defined networks loaded from .gexf files

To use pre-defined networks, modify `settings.py`:

```python
# Set to None to use generated networks (default)
employer_network_file = None  # e.g., 'defra' to load networks/employers/defra.gexf
union_network_file = None     # e.g., 'union_network' to load networks/unions/union_network.gexf

# Or specify network files:
employer_network_file = 'defra'  # Loads networks/employers/defra.gexf
union_network_file = 'my_union'  # Loads networks/unions/my_union.gexf
```

The model will automatically fall back to generated networks if the specified files don't exist.

### Testing Network Loading

Run the test script to verify network loading functionality:

```bash
python test_network_loading.py
```

This will show available network files and test different loading scenarios.

## Output Files

- `strikesim_data.h5`: Full time series data in HDF5 format
- `strikesim_summary.csv`: Summary statistics
- `monte_carlo_results.csv`: Results from multiple simulations
- `networks.png`: Visualization of union and employer networks
- `time_series.png`: Time series plots of key metrics

## Model Validation

The model is designed to be calibrated using:
- Academic case studies of historical strikes
- Publicly available information from newspaper articles and press releases
- Union data (without requiring access to sensitive information)

## Research Applications

This model can be used to:
- Test different union strategies before calling strikes
- Understand how network structure affects strike outcomes
- Analyze the impact of policy changes on strike success
- Compare different morale specifications
- Study the role of financial factors in strike dynamics

## Technical Details

- **Language**: Python 3.7+
- **Dependencies**: NetworkX, Pandas, NumPy, Matplotlib, Seaborn, H5Py
- **Architecture**: Object-oriented with clear separation of concerns
- **Data Storage**: HDF5 for full data, CSV for summaries
- **Visualization**: Matplotlib/Seaborn for plots and networks

## Contributing

This model is designed for research and union decision-making. Contributions are welcome, particularly:
- Additional morale specifications
- More sophisticated network generation
- Enhanced policy systems
- Improved visualization tools
- Model validation studies

## License

[Add appropriate license information]

## Citation

If you use this model in research, please cite:
[Add citation information when paper is published]
