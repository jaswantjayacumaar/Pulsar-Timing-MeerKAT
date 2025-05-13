# Pulsar-Timing-MeerKAT

# Pulsar Timing and ν˙ Variations Analysis

## Project Overview

This project focuses on analyzing the spin-down rate (ν˙) variations of pulsars using a combination of data from different observatories. The analysis incorporates two main methods for studying ν˙ variations: **Bayesian Inference** and **Fitwaves** analysis. The primary goal of this research was to evaluate the impact of combining datasets from the **Jodrell Bank Observatory (JBO)** and **MeerKAT telescope** for improved precision in determining pulsar parameters and studying spin-down rate changes.

The datasets and Python scripts used in this study are made publicly available on this repository. This work also showcases the use of **TEMPO2** and **Enterprise** tools for pulsar timing and Bayesian analysis.

## Research Workflow

### Step 1: **Obtaining Post-fit Residuals**

Using the **TEMPO2** package, post-fit residuals for selected pulsars are obtained after fitting the data with initial parameters.

### Step 2: **Bayesian Analysis**

The **Enterprise** Python package is used for Bayesian analysis. This analysis provides a robust estimation of ν˙ variations by fitting pulsar timing data from both JBO and MeerKAT.

### Step 3: **Fitwaves Analysis**

Using the **psrsalsa** package, the **Fitwaves** method is applied to pulsar timing data, specifically focusing on MeerKAT observations due to its high cadence.

### Step 4: **Plot Generation**

Custom Python scripts are used to generate residuals and nudot plots for visualizing the pulsar timing and spin-down rate variations.

---

## Requirements

Install the following packages:

* **TEMPO2** and **TEMPO3**: A package for pulsar timing analysis.
* **run_enterprise**: A package for Bayesian analysis in pulsar timing.
* **psrsalsa**: A tool for pulsar timing analysis (used for Fitwaves).
* **Python 3** with the following dependencies:

  * `emcee`
  * `matplotlib`
  * `numpy`
  * `scipy`

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/jaswantjayacumaar/Pulsar-Timing-MeerKAT.git
   ```

2. Install the required dependencies for **TEMPO2**, **run_enterprise**, and **psrsalsa** as outlined in their respective documentation.

---

## Usage/Commands

### Obtaining Post-fit Residuals

To generate post-fit residuals for a pulsar, use the following command with the `tempo2` software:

```bash
# Generate post-fit residuals
tempo2 -gr plk -f J1801-2920.par pf_J1801-2920.tim
```

To distinguish observations from different observatories (Jodrell Bank Observatory and MeerKAT), run:

```bash
# To distinguish observations by observatory
tempo2 -gr plk -f J1801-2920.par pf_J1801-2920.tim -colour -be
```

To convert the time units in the `.par` file from Barycentric Dynamical Time (TDB) to Barycentric Coordinate Time (TCB), use the `TRANSFORM` plugin in `tempo2`:

```bash
# Convert time units in .par file
tempo2 -gr transform old_filename.par new_filename.par
```

### Bayesian Analysis

**Note:** Before running the scripts, make sure to `cd` into the project directory containing the dataset and script files. All paths are relative to that working directory.


In this case, navigate to the appropriate subdirectory using:

```bash
cd /Bayesian\ Analysis/Bayesian_min_Gamma_4/JBO/J1801-2920
```

1. Begin the Bayesian analysis with the final `.par` and `.tim` files:

```bash
# Start Bayesian analysis with final .par and .tim files
/run_enterprise/run_enterprise.py final_J1801-2920.par J1801-2920_jbo.tim -j -A -8 --emcee -N 1000 --nwalk 32 -t4 --plot-chain
```

2. To limit the minimum value of Gamma to 4, use the following command:

```bash
# Set minimum Gamma value to 4
/run_enterprise/run_enterprise.py final_J1801-2920.par J1801-2920_jbo.tim -j -A -8 --emcee -N 1000 --nwalk 32 -t4 --plot-chain --red-gamma-min 4
```

3. Run `tempo2` with the `.par.post` files and `.tim` data for post-fit residuals:

```bash
# Run tempo2 for post-fit residuals
tempo2 -gr plk -f final_J1801-2920.par.post J1801-2920_jbo.tim
```

4. Save a new `.par` file after generating the post-fit residuals (`final2*.par`).

5. Generate plots of the pulsar data:

```bash
# Generate plots using the make_pulsar_plots.py script
/run_enterprise/old_scripts/make_pulsar_plots.py final2_J1913-0440.par J1801-2920_jbo.tim final2_J1801-2920.par
```

### Fitwaves Analysis

**Note:** Before running the scripts, make sure to `cd` into the project directory containing the dataset and script files. All paths are relative to that working directory.

In this case, navigate to the appropriate subdirectory using:

```bash
cd /Fitwaves\ Analysis/MeerKAT_Timing/J2048-1616
```

1. Create a barycentric time file (`bary.tim`):

```bash
# Create barycentric time file for fitwaves analysis
/psrsalsa/tempo3 -baryssbdump bary.tim final_J2048-1616.par J2048-1616.tim
```

2. Edit the macrofile to set the number of fitwaves to be fitted (adjust the number in the first line).

3. Run the `nudot.from.tempo3.py` script using the `.par` file, macrofile, barycentric time file, and the number of iterations:

```bash
# Run nudot.from.tempo3.py for fitwaves analysis
python nudot.from.tempo3.py -e final_J2048-1616.par -m macrofile.fit -b bary.tim -n 20
```

4. Plot the results of the fitwaves analysis:

```bash
# Plot the nudot vs. time for the pulsar
python plot.nudot.tempo3.py -F1 nudot.full.sample.txt -mjd MJDS.txt -p J2048-1616
```

5. Save the initial file and zoom in for a more precise view.

### Viewing Chi-squared Values

To view Chi-squared values for all observations, use the following terminal command inside the `PARTIMFS` folder:

```bash
# View Chi-squared values for all observations
for i in iteration.*.par ; do echo -n "$i " ; tempo2 -f $i $(basename $i .par).tim | grep Chisqr ; done
```

### Generating Nudot & Residual Plots

1. Copy the following files from their respective directories for the nudot plots:

   * `nudot.asc` from `JBO_Bayes`
   * `nudot.asc` from `MeerKAT_Bayes`
   * `nudot.asc` from `Combined_Bayes`
   * `nudot.mjd.txt` from `MeerKAT_fitwaves`

2. To obtain the residuals, split the data from the combined dataset. Run the following commands in the "Combined folder":

```bash
# Extract residuals for Jodrell Bank data
tempo2 -output general2 -f final2*.par J2048-1616.tim -s '{bat} {post} {err} {clkchain}\n' | grep JB | awk '{print $1,$2,$3}'  > jb_res.txt
```

```bash
# Extract residuals for MeerKAT data
tempo2 -output general2 -f final2*.par J2048-1616.tim -s '{bat} {post} {err} {clkchain}\n' | grep 'meerkat->' | awk '{print $1,$2,$3}' > mk_res.txt
```

3. Copy both `jb_res.txt` and `mk_res.txt`.

4. Run the `final_plots.ipynb` to generate the final plots.

5. If necessary, use the `arrange_mk.ipynb` script to reorder the MeerKAT data (`mk_res.txt`) by MJD.

---

## Conclusion

This repository presents a comprehensive analysis of pulsar timing residuals and spin-down rate (ν˙) variations using data from the Jodrell Bank Observatory and MeerKAT telescope. By employing Bayesian inference and Fitwaves analysis, this study demonstrates the power of combining datasets from different observatories and the usefulness of advanced analysis techniques in pulsar timing research. Future expansions could involve analyzing a larger set of pulsars and further refining the methodologies used.
