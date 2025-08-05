# harfed
Heterogeneity, Attacks, and Robustness in Federated Learning

This repository contains a Streamlit-based interactive application named **Harfed**, dedicated to configuring and running Federated Learning (FL) experiments with detailed visualization and monitoring. The app features an intuitive UI with comprehensive visualizations and real-time monitoring capabilities. It has been utilized in the research and experiments presented in our published paper: [Exploring Heterogeneity, Attacks, and Robustness in Federated Learning](https://ieeexplore.ieee.org/document/10857574).

---

## Overview

The Harfed application provides a streamlined interface for setting up and conducting Federated Learning experiments, particularly focusing on:
1. **Heterogeneous data distributions**: Support for advanced dataset partitioning strategies like Dirichlet and Pathological.
2. **Robustness testing**: Configuring experiments with malfunctions or adversarial behavior (e.g., targeted or untargeted attacks).
3. **Monitoring and visualization**: GPU utilization tracking, dataset partition visualizations, accuracy trends, and attack success rates over training rounds.

---

## Features

### **Main Functionalities**:
- **Dataset Configuration**: 
  - Selection of datasets and subsets.
  - Customizable partitioning strategies (e.g., Dirichlet, IID, etc.).
- **Attack Configuration**:
  - Integration of Local Differential Privacy (LDP) settings.
  - Fraction of malicious clients.
- **Federated Learning Strategies**:
  - Popular FL algorithms like FedAvg, FedMedian, FedProx, etc.
  - Flexible settings for model architectures, client fractions, and training rounds.
- **Real-time Monitoring**:
  - GPU resource tracking (memory, utilization, temperature).
  - Live experiment logs.
- **Results Visualization**:
  - Accuracy and attack success rate (ASR) plots over FL rounds.
  - Dataset partition label distribution charts.

---

## Usage Instructions

1. **Install dependencies**:
   Ensure you have Python 3.10+ and install the required packages using `pip`:
   ```
   pip install -r requirements.txt
   ```

2. **Launch the Streamlit app**:
   Run the following command to start the Streamlit application:
   ```
   streamlit run app.py
   ```

3. **Configure the experiment**:
   - Use the sidebar panels to configure all aspects of your experiment:
     - 📊 **Dataset Configuration**: Select datasets, subsets, and partitioning strategies
     - 🛡️ **Attack Configuration**: Configure attack types and malicious client fractions
     - 🔒 **Differential Privacy**: Set privacy parameters
     - ⚙️ **Federated Strategy**: Choose FL algorithms, models, and training parameters
     - 🖥️ **GPU Configuration**: Manage GPU resources and allocation

4. **Visualize data distribution**:
   - Enable the dataset partition visualization to see how data is distributed across clients
   - View label distributions across partitions with interactive charts
   - Examine sample data from clients with the image gallery feature
   - Download visualizations and distribution data for your records

5. **Run the experiment**:
   - Review the experiment summary in the expandable section
   - Click the "▶️ Run Experiment" button to start the training process
   - Monitor progress with the real-time progress bar and status updates
   - View live logs during execution to track training progress

6. **Analyze results**:
   - Examine final metrics (accuracy, attack success rate) in the results dashboard
   - Explore interactive charts showing performance over training rounds
   - View GPU utilization statistics with detailed graphs
   - Download all results, charts, and data in various formats (CSV, PNG, PDF)

---

## Citation

If you find this code useful in your research, please cite the paper:
[Exploring Heterogeneity, Attacks, and Robustness in Federated Learning](https://ieeexplore.ieee.org/document/10857574)