# Training README

Initial planned directory structure:
src/training/
├── __init__.py
├── pipeline.py
├── trainers/
│   ├── __init__.py
│   ├── base_trainer.py
│   ├── statistical_trainer.py
│   ├── ml_trainer.py
│   ├── dl_trainer.py
│   └── foundation_trainer.py
├── checkpointing/
│   ├── __init__.py
│   └── checkpoint_manager.py
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py
└── utils/
    ├── __init__.py
    ├── gpu_utils.py
    └── model_utils.py
