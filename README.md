# 🎨 Deep Learning - Schilderijen Classificatie

Een deep learning project dat schilderijen classificeert van vier beroemde kunstenaars: **Mondriaan**, **Picasso**, **Rembrandt** en **Rubens**.

## 📊 Resultaten

Het beste model (VGG16 met Transfer Learning) behaalt een nauwkeurigheid van **~94%** op de testset.
De dataset is niet mee gepusht, maar zou door notebooks in volgorde uit te voeren moeten correct komen te staan. In het filmpje kunt u zien wat ik hiermee heb gedaan.
## 📁 Project Structuur

```
├── datasets/
│   ├── raw/          # Originele gescrapete afbeeldingen
│   ├── cleaned/      # Opgeschoonde dataset
│   └── processed/    # Train/validation/test splits
├── models/           # Getrainde modellen (.keras, .h5)
├── notebooks/        # Jupyter notebooks
├── VSC_training_files/  # Scripts voor Vlaamse Supercomputer
└── requirements.txt
```

## 🚀 Installatie & Setup

### 1. Clone de repository
```bash
git clone https://github.com/r0995764/DeepLearning_Schilderijen.git
cd DeepLearning_Schilderijen
```

### 2. Maak een virtuele omgeving aan (aanbevolen)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OF
venv\Scripts\activate     # Windows
```

### 3. Installeer dependencies
```bash
pip install -r requirements.txt
```

### 4. Start Jupyter Notebook
```bash
jupyter notebook
```

## 📓 Notebooks

| Notebook | Beschrijving |
|----------|--------------|
| `01_datascraping.ipynb` | Web scraping van schilderijen via WikiArt |
| `02_data_cleaning_exploration.ipynb` | Data cleaning en exploratie |
| `03_preprocessing_split.ipynb` | Preprocessing en train/val/test split |
| `04_model_training_baseline.ipynb` | Baseline CNN model |
| `04b_data_augmentation-baseline.ipynb` | Model met data augmentation |
| `05_model_finetuning.ipynb` | Transfer Learning met VGG16 |
| `06_demo_gradio.ipynb` | **Interactieve demo** met Gradio |

## 🎯 Demo

Start de Gradio demo om zelf schilderijen te classificeren:

```bash
jupyter notebook notebooks/06_demo_gradio.ipynb
```

Upload een afbeelding van een schilderij en het model voorspelt welke kunstenaar het heeft gemaakt!

## 🛠️ Technologieën

- **TensorFlow/Keras** - Deep learning framework
- **VGG16** - Pre-trained model voor transfer learning
- **Gradio** - Interactieve web interface
- **Scikit-learn** - Evaluatie metrics
- **BeautifulSoup** - Web scraping

## 👥 Auteur

Mathieu - Vives Hogeschool

## 📝 Licentie

Dit project is gemaakt voor educatieve doeleinden.
