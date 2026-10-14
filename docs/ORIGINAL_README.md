# Intelligent Missing Data Analyzer

Benchmarks several numeric imputation strategies on a dataset by masking known values, reconstructing them, and comparing reconstruction error.

## Methods
- mean imputation
- median imputation
- KNN imputation
- iterative regression imputation

## Usage
```bash
pip install -r requirements.txt
python src/analyze_missing.py sample/data.csv --mask-rate 0.15
```

The recommendation is based on mean RMSE across evaluated numeric columns. It is a modeling aid, not an automatic substitute for understanding why data is missing.
