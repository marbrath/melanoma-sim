## Simulering
Ny prosedyre fra 26.04.23.

### addsim
```bash
module load R/4.1.2-foss-2021b
export SLURM_NTASKS=128
python simulate_data.py <num fams per year> <max children> <seed begin> <seed end>
python simulate_shuffle_data.py <max children> <seed begin> <seed end>
./generate-addsim-package-for-sim.sh <max children> <seed begin> <seed end>
python simulation_optimize.py <max children> <seed begin> <seed end>
```

### addsim reordered
Flytter alle syke så de ligger først.
```bash
# antar at man har kjørt det over for addsim først
python simulation_reordered_optimize.py <max children> <seed begin> <seed end>
```

### addsim shuffled
Stokker om på foreldrene og barna randomly.
```bash
# antar at man har kjørt det over for addsim først
python simulation_shuffled_optimize.py <max children> <seed begin> <seed end>
```

### aggf5
__Funker foreløpig bare med 5 barn.__

```bash
module load R/4.1.2-foss-2021b
export SLURM_NTASKS=128
python simulate_data.py <num fams per year> 5 <seed begin> <seed end>
./generate-aggf5-package-for-sim.sh 5
python simulation_aggf5_optimize.py 5 <seed begin> <seed end>
```
