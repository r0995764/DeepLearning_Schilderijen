#!/bin/bash -l
#SBATCH --cluster=genius
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --account=lp_edu_vives_ti_deep_learning

# 1. Modules laden
# Op Genius laden we de modules die we nodig hebben
module load TensorFlow/2.13.0-foss-2023a
module load matplotlib/3.7.2-gfbf-2023a
module load scikit-learn/1.3.1-gfbf-2023a
# Seaborn zit vaak niet in de standaard modules, dus installeren we die even snel lokaal
pip install --user seaborn

# 2. Map opzetten
WORKDIR=$VSC_SCRATCH/Schilderijen_Project/Evaluatie
mkdir -p $WORKDIR
cd $WORKDIR

# 3. Bestanden kopiëren van je data map naar hier
# We hebben de dataset en het model nodig
cp $VSC_DATA/Schilderijen_Project/dataset_processed_vsc.zip dataset.zip
cp $VSC_DATA/Schilderijen_Project/transfer_model_vgg16.h5 .
cp $VSC_DATA/Schilderijen_Project/evaluate_vsc.py .

# 4. Script draaien
python evaluate_vsc.py

# 5. Resultaten terugzetten
cp confusion_matrix.png classification_report.txt $VSC_DATA/Schilderijen_Project/