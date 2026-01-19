# BaseTSFM Design Critique - Full Conversation

## Initial Question

**User:** Just looking for an explanation, no need to change anything yet.... Why does our BaseTSFM class have _save_model_weightes AND save_model, as well as load_model_weights() and load_model()?

Imagine you're an advance AI engineer, could you take a critical look at the design of BaseTSFM and criticize it? See what can be made better? What can we clean up and get rid of, what's really not necessary there that is leading to confusion?

---

## Critical Analysis of BaseTSFM Design

### Current Method Separation - Why It Exists

The distinction exists because you're trying to separate **concerns**:

1. **save_model()** / **load_model()**: Handle **metadata** (config.json, metadata.json, training history)
2. **_save_checkpoint()** / **_load_checkpoint()**: Handle **model-specific files** (weights, optimizer state)

**The Problem**: This creates confusion about responsibility boundaries and violates the principle of having a single clear contract.

---

### Key Design Issues

#### 🔴 **1. Confusing Responsibility Split**
The comment in `_save_checkpoint()` says:
> "This method should ONLY handle writing checkpoint files. The base class save_model() handles config.json and metadata.json."

But in **save_model()**, you ALSO call `_save_checkpoint()`. This means:
- `save_model()` is orchestrating the full save operation
- But child classes must implement `_save_checkpoint()` without knowing what the parent already saved

**Better approach**: Either:
- Make `_save_checkpoint()` do **everything** model-specific (including metadata formatting), OR
- Have `save_model()` be entirely in base class and call multiple smaller abstract methods

#### 🔴 **2. The evaluate() Method is Poorly Designed**
```python
def evaluate(self, test_data: Any, return_predictions: bool = False) -> Dict[str, Any]:
    ...
    y_true = self._extract_ground_truth(test_data)
    ...
```

Then `_extract_ground_truth()` raises `NotImplementedError` by default. This means:
- The base class **pretends** to implement evaluate()
- But it actually doesn't work without child overrides
- This violates the Liskov Substitution Principle

**Better**: Make `evaluate()` abstract, or don't provide a base implementation that doesn't work.

#### 🔴 **3. Redundant Configuration Storage**

In `save_model()`, you save config **twice**:
```python
# Save config.json
json.dump(self.config.to_dict(), f, indent=2)

# Then in metadata.json:
metadata = {
    ...
    "config": self.config.to_dict(),  # Again!
}
```

**Why?** This is unnecessary duplication that can lead to inconsistencies.

#### 🔴 **4. Mixed Abstraction Levels in _train_model()**

You have:
```python
@abstractmethod
def _train_model(self, ...) -> Dict[str, Any]:
    raise NotImplementedError(...)  # But it's already @abstractmethod!
```

The `raise NotImplementedError` is redundant with `@abstractmethod` - Python already handles this.

#### 🔴 **5. Distributed Training Cleanup in __del__**

```python
def __del__(self):
    """Ensure cleanup happens even if not called explicitly."""
    try:
        self._cleanup_distributed()
    except Exception:
        pass
```

**Major issue**: Relying on `__del__` for critical cleanup (like distributed process groups) is **dangerous** because:
- `__del__` timing is unpredictable (depends on GC)
- In distributed scenarios, this can cause deadlocks or GPU memory leaks
- Silent exception swallowing hides real problems

**Better**: Use context managers or explicit cleanup in try/finally blocks (which you already do in `fit()`).

#### 🔴 **6. The fit() Method Does Too Much**

The `fit()` method:
- Sets up distributed training
- Enables LoRA
- Calls `_train_model()`
- Saves metadata
- Cleans up distributed resources

This is the **Template Method pattern**, which is fine, but some of this logic should be movable to child classes based on their needs (e.g., not all models need distributed setup in the same place).

#### 🔴 **7. is_fitted State Management is Fragile**

You set `self.is_fitted = True` in `fit()` after training, but:
- What if training partially fails and you catch the error?
- What if someone calls `load_model()` - when is `is_fitted` set?
- In `load_model()`, you load it from metadata, but what if the metadata is corrupted?

**Better**: Make `is_fitted` a property that checks if model weights exist, not a stored boolean.

#### 🔴 **8. Training Strategy is Redundant**

You have **both**:
- `TrainingStrategy` enum in `ModelConfig`
- Abstract method `get_training_strategy()` that models must implement

Why not just use the enum from config? Or if you need dynamic behavior, why store it in config at all?

---

### What Can Be Cleaned Up?

#### ✅ **Simplifications:**

1. **Merge or clarify save/load methods**:
   ```python
   # Option A: Single abstract method does everything
   @abstractmethod
   def save_model_artifacts(self, output_dir: str) -> None:
       """Save ALL model files (weights + metadata)"""
       pass
   
   # Option B: More granular but clearer
   @abstractmethod
   def _save_weights(self, output_dir: str) -> None:
       pass
   
   def save_model(self, output_dir: str) -> None:
       self._save_config(output_dir)
       self._save_metadata(output_dir)
       self._save_weights(output_dir)
   ```

2. **Remove config duplication**: Only save config.json once, reference it in metadata

3. **Remove NotImplementedError from abstract methods**: `@abstractmethod` is enough

4. **Remove __del__ distributed cleanup**: You already have it in `try/finally` in `fit()`

5. **Make evaluate() fully abstract**: Don't provide a broken base implementation

6. **Remove `training_strategy` from config**: Just use `get_training_strategy()` method

7. **Simplify _compute_metrics**: Move to a separate utility module - it's not model-specific

#### ✅ **Better Patterns:**

1. **Use context manager for distributed training**:
   ```python
   @contextmanager
   def distributed_context(self):
       self._setup_distributed()
       try:
           yield
       finally:
           self._cleanup_distributed()
   ```

2. **Make is_fitted a property**:
   ```python
   @property
   def is_fitted(self) -> bool:
       return self.model is not None and hasattr(self.model, 'trained_weights')
   ```

3. **Separate concerns better**: Create `ModelPersistence`, `DistributedTrainer`, `LoRAAdapter` as separate mixins or helper classes

---

### The Biggest Issue: Over-Engineering

This base class tries to handle:
- Model lifecycle (init, train, predict, eval, save, load)
- Distributed training (DDP, DeepSpeed, FSDP)
- LoRA integration
- Metrics computation
- Metadata management
- Git tracking
- Configuration management

**That's 7+ responsibilities in one class!** This violates the Single Responsibility Principle.

**Better architecture**:
```python
class BaseTSFM(ABC):
    """Only model lifecycle"""
    
class DistributedMixin:
    """Distributed training setup/cleanup"""
    
class LoRAMixin:
    """LoRA integration"""
    
class MetricsMixin:
    """Compute standard metrics"""
    
class TTMForecaster(BaseTSFM, DistributedMixin, LoRAMixin, MetricsMixin):
    """Compose what you need"""
```

This would make each piece testable, optional, and understandable in isolation.

---

### Summary

**Main confusions to eliminate:**
1. ✂️ Merge `save_model`/`_save_checkpoint` or make the boundary crystal clear
2. ✂️ Remove config duplication (config.json + metadata.json)
3. ✂️ Make `evaluate()` abstract (don't fake it)
4. ✂️ Remove `__del__` - distributed cleanup is already in try/finally
5. ✂️ Consider breaking into smaller mixins for distributed/LoRA/metrics

The design is **functional but overly complex**. The save/load confusion stems from trying to be too clever about separation of concerns without clear documentation of the contract.

---

## Naming Convention Analysis

**User:** Can you also analyze my naming conventions here? I'm not sure everything is named with more industry standard conventions.

Looking at your code from a Python/ML industry standards perspective, here are the issues:

---

### 🔴 **Major Naming Issues**

#### **1. Class Names - Acronym Handling**

```python
class BaseTSFM(ABC):  # ❌ Mixing case styles
```

**Problem**: In Python PascalCase, acronyms should be treated as words, not all-caps.

**Industry Standard**:
- `BaseTsfm` (treat as word), or better yet:
- `BaseTimeSeriesFoundationModel` (fully spelled out - clearest)
- `BaseForecaster` (shorter, general)

**Same issue**: `LoRAConfig` → should be `LoraConfig`

---

#### **2. Getters Should Be Properties**

```python
@abstractmethod
def get_training_strategy(self) -> TrainingStrategy:  # ❌
    pass

@abstractmethod
def supports_lora(self) -> bool:  # ❌
    pass

def get_model_info(self) -> Dict[str, Any]:  # ❌
    pass
```

**Problem**: These are queries without side effects - they should be properties, not methods.

**Industry Standard**:
```python
@property
@abstractmethod
def training_strategy(self) -> TrainingStrategy:
    pass

@property
@abstractmethod
def supports_lora(self) -> bool:
    pass

@property
def model_info(self) -> Dict[str, Any]:
    pass
```

**Rationale**: The Python style guide (PEP 8) says properties should be used for computed attributes that don't have side effects. Reserve `get_*()` methods for operations that might be expensive or have side effects.

---

#### **3. Abbreviated Variable Names**

```python
d_model: int = 512        # ❌ Too terse
n_heads: int = 8          # ❌ Abbreviation
n_layers: int = 6         # ❌ Abbreviation
fp16: bool = True         # ❌ Jargon without context
```

**Industry Standard**:
```python
hidden_dim: int = 512              # or model_dim
num_attention_heads: int = 8       # Fully spelled
num_layers: int = 6                # Fully spelled
use_mixed_precision: bool = True   # Descriptive
```

**Note**: `d_model` is common in transformer papers, but in code, clarity > brevity.

---

#### **4. Sklearn Convention for Fitted Attributes**

```python
self.is_fitted = False  # ❌ Missing trailing underscore
self.model = None       # ❌ Missing trailing underscore
```

**Industry Standard** (from sklearn):
```python
self.is_fitted_ = False  # ✅ Trailing underscore
self.model_ = None       # ✅ Trailing underscore
```

**Rationale**: In sklearn, any attribute set during `fit()` gets a trailing underscore to distinguish learned attributes from initialization parameters. This helps users know what's safe to access before/after fitting.

---

#### **5. Inconsistent Directory Parameter Names**

```python
def save_model(self, output_dir: str, ...):     # Uses "output_dir"
    pass

@classmethod
def load_model(cls, model_dir: str, ...):       # Uses "model_dir"
    pass
```

**Problem**: Asymmetry makes the API harder to remember.

**Industry Standard**: Pick one:
```python
# Option 1: Both use "path"
save_model(self, model_path: str)
load_model(cls, model_path: str)

# Option 2: Both use "directory"
save_model(self, directory: str)
load_model(cls, directory: str)
```

---

#### **6. Ambiguous "Strategy" Naming**

```python
fit_strategy: str = "fine_tune"  # ❌ What kind of strategy?
training_strategy: TrainingStrategy = TrainingStrategy.TRANSFORMERS
```

**Problem**: Two different concepts both called "strategy":
- `fit_strategy` = training mode (zero-shot/fine-tune/from-scratch)
- `training_strategy` = training framework (Transformers/PyTorch/Custom)

**Industry Standard**:
```python
training_mode: str = "fine_tune"  # What you're doing
training_backend: TrainingBackend = TrainingBackend.TRANSFORMERS  # How you're doing it
```

Or rename the enum:
```python
class TrainingBackend(Enum):  # Clearer than "TrainingStrategy"
    HUGGINGFACE_TRAINER = "transformers"
    PYTORCH_NATIVE = "pytorch"
    CUSTOM = "custom"
```

---

#### **7. Boolean Naming - Negative Flags**

```python
use_cpu: bool = False  # ❌ Double negative: "not use_cpu" = use GPU
```

**Industry Standard**: Positive flags are clearer:
```python
force_cpu: bool = False  # "force_cpu=True" is clear
# Or better yet:
device: str = "auto"  # "cpu", "cuda", "auto"
```

---

#### **8. Method Name Verbosity**

```python
def _extract_ground_truth(self, test_data: Any) -> np.ndarray:  # ❌ Verbose
    pass

def _save_training_metadata(self, output_dir: str, ...) -> None:  # ❌ Redundant
    pass
```

**Industry Standard**:
```python
def _get_targets(self, test_data: Any) -> np.ndarray:  # Concise
    pass

def _save_metadata(self, output_dir: str, ...) -> None:  # Context is clear
    pass
```

---

#### **9. Generic Parameter Names**

```python
def predict(self, data: Any, ...) -> Union[np.ndarray, Dict[str, Any]]:
    #              ^^^^ Too generic
```

**Industry Standard**:
```python
def predict(self, input_data: Any, ...) -> Union[np.ndarray, Dict[str, Any]]:
# Or even better:
def predict(self, X: Any, ...) -> Union[np.ndarray, Dict[str, Any]]:  # sklearn convention
```

---

#### **10. Config vs Configuration**

```python
class ModelConfig:      # Uses "Config"
class LoRAConfig:       # Uses "Config"
class DistributedConfig:  # Uses "Config"
```

**Industry Standard**: This is actually fine! `Config` is a widely accepted abbreviation. But be consistent - don't mix `Config` and `Configuration` in the same codebase.

---

### 📊 **Summary of Naming Improvements**

| Current | Better | Reason |
|---------|--------|--------|
| `BaseTSFM` | `BaseTimeSeriesModel` or `BaseTsfm` | PascalCase treats acronyms as words |
| `LoRAConfig` | `LoraConfig` | Same reason |
| `get_training_strategy()` | `training_strategy` (property) | No side effects → property |
| `supports_lora()` | `supports_lora` (property) | No side effects → property |
| `get_model_info()` | `model_info` (property) | No side effects → property |
| `d_model` | `hidden_dim` or `model_dim` | Clarity over brevity |
| `n_heads` | `num_attention_heads` | Spell out "number" |
| `n_layers` | `num_layers` | Spell out "number" |
| `fp16` | `use_mixed_precision` | Self-documenting |
| `is_fitted` | `is_fitted_` | Sklearn convention for fitted attrs |
| `self.model` | `self.model_` | Sklearn convention |
| `output_dir` + `model_dir` | Both → `path` or `directory` | Consistency |
| `fit_strategy` | `training_mode` | Clearer intent |
| `training_strategy` | `training_backend` | Distinguish from mode |
| `use_cpu` | `force_cpu` or `device` | Positive flags |
| `_extract_ground_truth` | `_get_targets` | Concise |
| `data` parameter | `input_data` or `X` | More specific |

---

### 🎯 **Most Critical Changes**

1. **Properties not methods** for `training_strategy`, `supports_lora`, `model_info`
2. **Trailing underscores** for fitted attributes (`is_fitted_`, `model_`)
3. **Consistent directory parameters** (pick one name)
4. **Rename strategies** to distinguish mode from backend

These align with **sklearn**, **PyTorch**, and **transformers** library conventions, making your API more intuitive for ML practitioners.

---

## Refactoring Checklist Request

**User:** Can you please create a checklist for all of these recommended changes to these classes?

## Plan: Base Model Framework Refactoring

A comprehensive refactoring of the base model framework to improve design patterns, naming conventions, and maintainability across ~24 files and 5 active model implementations.

### Steps

1. **Phase 1: Design Cleanup - Simplify save/load methods** - Remove confusing split between `save_model()`/`_save_checkpoint()`. Consolidate to clearer contract: base class handles metadata (config.json only), abstract method `_save_model_weights()` handles weights. Update [base_model.py](src/models/base/base_model.py) and all 5 model implementations in [ttm/model.py](src/models/ttm/model.py), [tsmixer/model.py](src/models/tsmixer/model.py), [moment/model.py](src/models/moment/model.py), [timesfm/model.py](src/models/timesfm/model.py), [moirai/model.py](src/models/moirai/model.py).

2. **Phase 2: Design Cleanup - Remove redundant patterns** - Remove `__del__` distributed cleanup (already in try/finally), remove `NotImplementedError` from `@abstractmethod` decorators, remove duplicate config storage in metadata.json, make `evaluate()` fully abstract. Update [base_model.py](src/models/base/base_model.py) lines 490-578, 985-993.

3. **Phase 3: Naming - Convert getters to properties** - Change `get_training_strategy()`, `supports_lora()`, `get_model_info()` to `@property` decorators. Update [base_model.py](src/models/base/base_model.py) and all 5 model implementations that override these methods.

4. **Phase 4: Naming - Add sklearn-style trailing underscores** - Rename fitted attributes: `is_fitted` → `is_fitted_`, `model` → `model_`, `training_history` → `training_history_`, `best_metrics` → `best_metrics_`. Update [base_model.py](src/models/base/base_model.py) and propagate through all 5 model files plus [evaluator.py](src/evaluation/evaluator.py).

5. **Phase 5: Naming - Improve config attribute names** - Rename: `d_model` → `hidden_dim`, `n_heads` → `num_attention_heads`, `n_layers` → `num_layers`, `fp16` → `use_mixed_precision`, `use_cpu` → `force_cpu`, `fit_strategy` → `training_mode`, `training_strategy` → `training_backend`. Update [base_model.py](src/models/base/base_model.py) ModelConfig and all 5 model config classes.

6. **Phase 6: Naming - Fix class name acronyms** - Rename `BaseTSFM` → `BaseTimeSeriesModel` (already done in code, but update docs), `LoRAConfig` → `LoraConfig`, `TrainingStrategy` → `TrainingBackend`. Update 3 core files, 10 model files, 7 scripts, 4 docs ([base_model_framework_README.md](docs/base_model_framework_README.md), [base_model_lora_compatibility.md](docs/base_model_lora_compatibility.md), [ARCHITECTURE.md](configs/data/ARCHITECTURE.md), [src_train_model_reorg_plan.md](src_train_model_reorg_plan.md)).

7. **Phase 7: Naming - Standardize parameter names** - Make save/load methods consistent: both use `path` parameter. Rename `_extract_ground_truth()` → `_get_targets()`, `data` parameter → `input_data`. Update [base_model.py](src/models/base/base_model.py) and propagate to all implementations.

8. **Phase 8: Add deprecation warnings** - Create backward compatibility aliases for old names with `DeprecationWarning` to support existing code during transition. Add to [base_model.py](src/models/base/base_model.py) and [\_\_init\_\_.py](src/models/base/__init__.py).

9. **Phase 9: Update all imports** - Update import statements in all 24 affected files to use new names. Use find/replace with regex to catch all variations in model files, scripts, and examples.

10. **Phase 10: Update documentation** - Revise 4 documentation files to use new naming conventions consistently. Update code examples, API references, and architecture diagrams to reflect refactored design.

### Further Considerations

1. **Breaking changes strategy** - Should we maintain backward compatibility aliases indefinitely, or set a deprecation timeline? Consider semantic versioning impact.
2. **Test coverage** - No existing tests reference base class. Should we add comprehensive unit tests for the refactored framework before proceeding?
3. **Mixin architecture** - The base class handles 7+ responsibilities (see analysis). Should we extract `DistributedMixin`, `LoraMixin`, `MetricsMixin` for better separation of concerns in a future phase?
4. **Empty model folders** - 5 planned models (chronos, tide, timellm, timegpt, ts2vec) have no implementation. Should we create template files with new conventions now to avoid future updates?

---

## Phase 1 Detailed Implementation

**User:** Regarding phase 1, could you write the changes here for me that you'd recommend in a little more detail? You can write the code for me here in the chat, please don't start making changes in the files yet...

## Phase 1: Detailed Refactoring Plan - Save/Load Method Cleanup

### Current Problems

1. **Duplicate config storage**: Config saved in both `config.json` AND `metadata.json`
2. **Confusing responsibility split**: `save_model()` does metadata + calls `_save_checkpoint()`, but unclear what each owns
3. **Inconsistent naming**: `_save_checkpoint()` vs `save_model()` - one is about checkpoints, other about full model

### Recommended Changes

---

#### **Option A: Simplified Two-Method Pattern (RECOMMENDED)**

**Philosophy**: Base class handles ALL metadata files. Subclasses only handle model weights.

##### **In `base_model.py`:**

```python
# ============= NEW ABSTRACT METHOD =============
@abstractmethod
def _save_model_weights(self, path: str) -> None:
    """Save ONLY the model weights/state to the given path.
    
    Args:
        path: Directory path where model weights should be saved.
        
    Note:
        This method should ONLY save model weights and optimizer state.
        The base class handles config.json and training_metadata.json.
        
    Example implementations:
        - PyTorch: torch.save(self.model_.state_dict(), os.path.join(path, "model.pt"))
        - HuggingFace: self.model_.save_pretrained(path)
        - TensorFlow: self.model_.save_weights(os.path.join(path, "weights.h5"))
    """
    pass

@abstractmethod
def _load_model_weights(self, path: str) -> None:
    """Load ONLY the model weights/state from the given path.
    
    Args:
        path: Directory path containing model weights.
        
    Note:
        This method should ONLY load model weights and optimizer state.
        The base class handles config.json and training_metadata.json.
        You can assume self.config is already loaded when this is called.
    """
    pass


# ============= PUBLIC SAVE METHOD (SIMPLIFIED) =============
def save_model(self, path: str) -> None:
    """Save the complete model to disk.
    
    Saves:
    - config.json: Model configuration
    - training_metadata.json: Training history, metrics, git info
    - Model weights: Via subclass _save_model_weights() implementation
    
    Args:
        path: Directory path where the model will be saved.
        
    Example:
        >>> model.fit(train_data)
        >>> model.save_model("./trained_models/my_model")
    """
    os.makedirs(path, exist_ok=True)
    
    # 1. Save configuration (ONLY place config is saved)
    config_path = os.path.join(path, "config.json")
    with open(config_path, "w") as f:
        json.dump(self.config.to_dict(), f, indent=2)
    
    # 2. Save training metadata (NO config duplication)
    metadata = {
        "model_type": self.__class__.__name__,
        "is_fitted": self.is_fitted_,
        "training_history": self.training_history_,
        "best_metrics": self.best_metrics_,
        "timestamp": pd.Timestamp.now().isoformat(),
        "lora_enabled": self.lora_config.enabled,
        "distributed_enabled": self.distributed_config.enabled,
        "training_backend": self.training_backend.value,
    }
    
    # Add git info if available
    try:
        import git
        repo = git.Repo(search_parent_directories=True)
        metadata["git_commit"] = repo.head.commit.hexsha
        metadata["git_branch"] = repo.active_branch.name
        metadata["git_dirty"] = repo.is_dirty()
    except (ImportError, Exception):
        pass
    
    metadata_path = os.path.join(path, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # 3. Save model weights (delegated to subclass)
    self._save_model_weights(path)
    
    info_print(f"✅ Model saved to {path}")


# ============= PUBLIC LOAD METHOD (SIMPLIFIED) =============
@classmethod
def load_model(cls, path: str, config: Optional[ModelConfig] = None) -> "BaseTimeSeriesModel":
    """Load a saved model from disk.
    
    Loads:
    - config.json: Model configuration (unless override provided)
    - training_metadata.json: Training history and metrics
    - Model weights: Via subclass _load_model_weights() implementation
    
    Args:
        path: Directory containing the saved model.
        config: Optional ModelConfig to override the saved configuration.
            Use this to modify hyperparameters when loading.
    
    Returns:
        Loaded model instance ready for prediction or continued training.
        
    Example:
        >>> model = TTMForecaster.load_model("./trained_models/my_model")
        >>> predictions = model.predict(test_data)
    """
    # 1. Load config if not provided
    if config is None:
        config_path = os.path.join(path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"No config.json found at {config_path}. "
                "Either provide a config or ensure the model was saved correctly."
            )
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = ModelConfig.from_dict(config_dict)
    
    # 2. Create instance (calls _initialize_model)
    instance = cls(config)
    
    # 3. Load model weights (delegated to subclass)
    instance._load_model_weights(path)
    
    # 4. Load training metadata
    metadata_path = os.path.join(path, "training_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        instance.training_history_ = metadata.get("training_history", {})
        instance.best_metrics_ = metadata.get("best_metrics", {})
        instance.is_fitted_ = metadata.get("is_fitted", False)
    
    info_print(f"✅ Model loaded from {path}")
    return instance
```

---

##### **In `ttm/model.py` (and other model implementations):**

**BEFORE (current confusing version):**
```python
def _save_checkpoint(self, output_dir: str) -> None:
    """Save TTM model checkpoint using HuggingFace format."""
    if self.model is not None:
        self.model.save_pretrained(output_dir)
        info_print(f"TTM model saved to {output_dir}")

def _load_checkpoint(self, model_dir: str) -> None:
    """Load TTM model checkpoint from a directory."""
    try:
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_dir)
        info_print(f"TTM model checkpoint loaded from {model_dir}")
    except Exception as e:
        error_print(f"Failed to load model checkpoint: {str(e)}")
        raise
```

**AFTER (clear and simple):**
```python
def _save_model_weights(self, path: str) -> None:
    """Save TTM model weights using HuggingFace format.
    
    Args:
        path: Directory where weights will be saved.
    """
    if self.model_ is None:
        raise ValueError("Cannot save weights: model has not been initialized")
    
    self.model_.save_pretrained(path)
    info_print(f"TTM weights saved to {path}")


def _load_model_weights(self, path: str) -> None:
    """Load TTM model weights from HuggingFace format.
    
    Args:
        path: Directory containing saved weights.
    """
    from transformers import AutoModel
    
    self.model_ = AutoModel.from_pretrained(path)
    info_print(f"TTM weights loaded from {path}")
```

---

### Key Improvements

#### ✅ **1. Clear Separation of Concerns**
- **Base class**: Handles `config.json` and `training_metadata.json`
- **Subclass**: Handles model weights ONLY

#### ✅ **2. No Config Duplication**
- Config stored ONCE in `config.json`
- `training_metadata.json` has metrics and git info, NOT config

#### ✅ **3. Consistent Naming**
- `save_model()` / `load_model()` → User-facing public API
- `_save_model_weights()` / `_load_model_weights()` → Internal implementation hooks

#### ✅ **4. Better Parameter Names**
- Consistent use of `path` instead of mixing `output_dir` and `model_dir`

#### ✅ **5. Clearer Docstrings**
- Each method explicitly states what it's responsible for
- Examples show usage patterns

---

### Migration Path for Other Models

**For PyTorch-native models** (e.g., TSMixer):
```python
def _save_model_weights(self, path: str) -> None:
    if self.model_ is None:
        raise ValueError("Cannot save weights: model not initialized")
    
    weights_path = os.path.join(path, "model_weights.pt")
    torch.save(self.model_.state_dict(), weights_path)
    
    # Optionally save optimizer state
    if hasattr(self, 'optimizer_'):
        optimizer_path = os.path.join(path, "optimizer_state.pt")
        torch.save(self.optimizer_.state_dict(), optimizer_path)

def _load_model_weights(self, path: str) -> None:
    weights_path = os.path.join(path, "model_weights.pt")
    state_dict = torch.load(weights_path, map_location='cpu')
    self.model_.load_state_dict(state_dict)
    
    # Optionally load optimizer state
    optimizer_path = os.path.join(path, "optimizer_state.pt")
    if os.path.exists(optimizer_path) and hasattr(self, 'optimizer_'):
        optimizer_state = torch.load(optimizer_path, map_location='cpu')
        self.optimizer_.load_state_dict(optimizer_state)
```

**For models with custom serialization** (e.g., TimeGPT API-based):
```python
def _save_model_weights(self, path: str) -> None:
    # TimeGPT doesn't have local weights, save API credentials/config instead
    api_config = {
        "api_key": self.api_key,
        "model_id": self.model_id,
        "fine_tuned_checkpoint": self.checkpoint_id
    }
    
    api_path = os.path.join(path, "api_config.json")
    with open(api_path, "w") as f:
        json.dump(api_config, f, indent=2)

def _load_model_weights(self, path: str) -> None:
    api_path = os.path.join(path, "api_config.json")
    with open(api_path, "r") as f:
        api_config = json.load(f)
    
    self.api_key = api_config["api_key"]
    self.model_id = api_config["model_id"]
    self.checkpoint_id = api_config["fine_tuned_checkpoint"]
```

---

### File Structure After Save

```
my_model/
├── config.json                    # ← Configuration ONLY (no duplication)
├── training_metadata.json         # ← Metrics, git info, timestamps
└── [model-specific files]         # ← Weights (varies by framework)
    ├── pytorch_model.bin          # HuggingFace models
    ├── config.json                # (HF models also save their own config)
    └── ...
    OR
    ├── model_weights.pt           # PyTorch native
    └── optimizer_state.pt
```

---

### Benefits Summary

| Before | After |
|--------|-------|
| Config in 2 places | Config in 1 place |
| Unclear method responsibilities | Crystal clear responsibilities |
| `_save_checkpoint` naming confusion | `_save_model_weights` is explicit |
| Mixed `output_dir`/`model_dir` | Consistent `path` parameter |
| Comment says "base handles config" but both save it | Actually enforced in code |

This is **much** cleaner and follows the **Template Method Pattern** correctly!

---

## Property vs Method Question

**User:** So my BaseTimeSeriesFoundationModel class has the @property @abstractmethod decorator for the training_backend that is a passthrough to the child class right? 

But my linter is giving me a yellow underline when I have TTM training_backend populated in TTMForecaster. The linter is saying that it incorrectly overrides the property of the base class?

The issue is likely with **how you're overriding the property** in the child class. When overriding an abstract property, you need to use the `@property` decorator in the child class as well.

Let me show you the correct pattern:

## ❌ **Incorrect - This causes linter errors:**

```python
# In TTMForecaster
def training_backend(self) -> TrainingBackend:  # ❌ Missing @property
    return TrainingBackend.TRANSFORMERS
```

Or:

```python
# In TTMForecaster.__init__
self.training_backend = TrainingBackend.TRANSFORMERS  # ❌ Trying to override property with attribute
```

## ✅ **Correct - This is what you need:**

```python
# In TTMForecaster
@property
def training_backend(self) -> TrainingBackend:
    """Return the training backend used by TTM."""
    return TrainingBackend.TRANSFORMERS
```

---

Can you share the exact code from your `TTMForecaster` where you're implementing `training_backend`? Also, what line in `base_model.py` has the abstract property definition?

I can then tell you exactly what to fix. The pattern should be:

**Base class:**
```python
@property
@abstractmethod
def training_backend(self) -> TrainingBackend:
    """Return the training backend this model uses."""
    pass
```

**Child class:**
```python
@property
def training_backend(self) -> TrainingBackend:
    """TTM uses HuggingFace Transformers backend."""
    return TrainingBackend.TRANSFORMERS
```

Note: You only need `@property` in the child, not `@abstractmethod` again.

---

## Property Access Question

**User:** Now that I've made training strategy to be my training backend and for it to be a property instead of a method, how am I supposed to reference self.training_backend() that I had before?

Simple! Remove the parentheses. Properties are accessed like attributes, not called like methods.

## Change Required:

**Before (method call):**
```python
self.training_backend()  # ❌ With parentheses
self.get_training_strategy()  # ❌ Old method name
```

**After (property access):**
```python
self.training_backend  # ✅ No parentheses
```

---

## Common places you'll need to update:

1. **In base class `fit()` method** - where it logs the strategy:
```python
# Before
info_print(f"Training strategy: {self.get_training_strategy().value}")

# After
info_print(f"Training backend: {self.training_backend.value}")
```

2. **In `save_model()` / metadata saving:**
```python
# Before
"training_strategy": self.get_training_strategy().value

# After
"training_backend": self.training_backend.value
```

3. **In `get_model_info()` method:**
```python
# Before
"training_strategy": self.get_training_strategy().value

# After  
"training_backend": self.training_backend.value
```

4. **Any conditional checks:**
```python
# Before
if self.get_training_strategy() == TrainingStrategy.TRANSFORMERS:

# After
if self.training_backend == TrainingBackend.TRANSFORMERS:
```

---

Would you like me to search through your codebase and find all the places that need updating? I can do a grep search to identify every instance of `get_training_strategy()` or `training_backend()` that needs the parentheses removed.
