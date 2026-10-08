# HOUSE PRICES: ADVANCED REGRESSION TECHNIQUES

Регрессия: предсказание цены дома (`SalePrice`) по 79 признакам.

## Установка

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
pandas
numpy
scikit-learn
matplotlib
seaborn
pyyaml
lightgbm
catboost
xgboost
torch
```

## Запуск

```bash
python main.py --model catboost

```

Результат сохраняется в `submissions/submission_housing.csv`.

## Пайплайн

```
load_data (raw csv)
    ↓
build_housing_features  (FE: TotalSF, TotalBath, Age, RemodAge, TotalRooms, HasGarage, HasBsmt, HasPool, Has2ndFloor, OverallQuality)
    ↓
split_train_val  (80/20, до препроцессинга - без утечки)
    ↓
TabularPreprocessor:
    • drop_columns (Id)
    • log_target: log1p(SalePrice)
    • none_categorical: PoolQC, MiscFeature, Alley, Fence, ...  =>  "None"
    • none_numeric: MasVnrArea, GarageYrBlt, Bsmt*, Garage*  =>  0
    • group_median: LotFrontage по Neighborhood (fit на train)
    • fillna: Electrical  =>  mode (fit на train)
    • ordinal: ExterQual, BsmtQual, KitchenQual, GarageFinish, ... (порядок из конфига)
    • one-hot: auto - все оставшиеся object-колонки (fit на train, transform на val/test)
    ↓
train_classic  (CV + fit)
    ↓
make_submission  (expm1 для обратной log-трансформации)
```

## Результаты

| Модель                      | CV RMSE (log-таргет) |
|-----------------------------|----------------------|
| LinearRegression (baseline) | 0.1636               |
| Ridge (α=10)                | 0.1524               |
| Lasso (α=0.01)              | 0.1485               |
| Random Forest               | 0.1423               |
| MLP [64, 32]                | 0.1431               |
| LightGBM                    | 0.1315               |
| XGBoost                     | 0.1269               |
| **CatBoost** 👑             | **0.1195**           |

**Финальная модель:** CatBoost (CV RMSE 0.1195 ± 0.0162, holdout 0.1360).
**Kaggle Score:** 0.12656 (RMSLE).

## Структура проекта

```text
housing-ml/
├── 📁 configs/
│   └── 📄 housing.yaml             # YAML-конфиг проекта
│
├── 📁 data/                        # Сырые данные (в .gitignore)
│   ├── 📄 train_housing.csv
│   └── 📄 test_housing.csv
│
├── 📁 notebooks/
│   ├── 📓 EDA_Housing.ipynb
│   └── 📓 Modeling_Housing.ipynb
│
├── 📁 src/                         # Модули пайплайна
│   ├── 🐍 __init__.py
│   ├── 🐍 config.py                # Загрузка YAML-конфигов
│   ├── 🐍 data.py                  # Загрузка данных и сплит
│   ├── 🐍 features.py              # FE + TabularPreprocessor
│   ├── 🐍 models.py                # Фабрика моделей + MLP (PyTorch)
│   ├── 🐍 train.py                 # CV, обучение моделей и DNN
│   ├── 🐍 predict.py               # Генерация submission
│   └── 🐍 utils.py                 # Метрики, сиды, утилиты
│
├── 📁 submissions/                 # Результаты (в .gitignore, оставлен .gitkeep)
│   └── 📄 .gitkeep
│
├── 🐍 main.py
├── 📄 requirements.txt
├── 📄 .gitignore
└── 📄 README.md
```

## Ключевые принципы

- **Препроцессор фитится только на train.** В val/test применяется `transform`,
  статистики (медианы, моды, скейлер, OHE-категории) не пересчитываются.
  Утечек нет.
- **One-hot fit на train, transform на val/test.** Новые категории кодируются
  как «все нули» (`handle_unknown="ignore"`) - без рассинхрона форм данных.
- **Лог-трансформация таргета:** `log1p(SalePrice)` при обучении, `expm1(pred)`
  при генерации submission. Метрика RMSE считается на лог-таргете (= RMSLE).
- **Early stopping + возврат к лучшему чекпоинту** в DNN (`train_dnn` в `src/models.py`).
- **Единый источник истины:** гиперпараметры и пути - только в `configs/housing.yaml`.
  `src/config.py` только загружает и разрешает пути.

## Ноутбуки

### EDA
- **`EDA_Housing.ipynb`** - распределение `SalePrice` (скошено вправо  =>  log1p),
  пропуски-«нет фичи» (PoolQC, MiscFeature, Alley), топ-числовые (OverallQual,
  GrLivArea), топ-категориальные (ExterQual, BsmtQual), выводы о FE и
  ordinal-кодировании.

### Моделирование
- **`Modeling_Housing.ipynb`** - линейные модели, деревья и MLP через 5-fold CV,
  MLP требует стандартизации таргета (иначе обучение расходится),
  финальная модель - CatBoost.

## Тюнинг гиперпараметров

Секция `tuning` в `configs/housing.yaml` - заготовка под Optuna (хотел, но передумал, вроде и так неплохой перфоманс)
Сейчас `enabled: false`: гиперпараметры подобраны вручную в ноутбуке моделирования.

