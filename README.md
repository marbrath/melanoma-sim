## Simulering
Ny prosedyre fra 26.04.23.

```bash
module load R/4.1.2-foss-2021b
export SLURM_NTASKS=128
python simulate_data.py <num fams per year> <max children> <seed begin> <seed end>
./generate-addsim-package-for-sim.sh <max children> <seed begin> <seed end>
python simulation_optimize.py <max children> <seed begin> <seed end>
```
