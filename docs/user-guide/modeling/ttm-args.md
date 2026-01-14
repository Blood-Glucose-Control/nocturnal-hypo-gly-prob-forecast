# Tiny Time Mixer (TTM)
Reference: [https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.ttm.TinyTimeMixerForecaster.html](https://www.sktime.net/en/latest/api_reference/auto_generated/sktime.forecasting.ttm.TinyTimeMixerForecaster.html)


Minimum trainning code
```
forecaster = TinyTimeMixerForecaster(
    model_path=None,
    training_mode="full",
    config={
            "context_length": 8,
            "prediction_length": 2
    },
    training_args={
        "num_train_epochs": 1,
        "output_dir": "test_output",
        "per_device_train_batch_size": 32,
    },
)
```

```
  """
    TinyTimeMixer Forecaster for Zero-Shot Forecasting of Multivariate Time Series.

    Wrapping implementation in [1]_ of method proposed in [2]_. See [3]_
    for tutorial by creators.

    TinyTimeMixer (TTM) are compact pre-trained models for Time-Series Forecasting,
    open-sourced by IBM Research. With less than 1 Million parameters, TTM introduces
    the notion of the first-ever "tiny" pre-trained models for Time-Series Forecasting.

    ** Fit Strategies **: "zero-shot" | "minimal" | "full"
    - *zero-shot* for direct predictions without training
    - *minimal* fine-tuning for lightweight adaptation to new data,
    - *full*: fine-tuning for comprehensive model training.

    **Initialization Process**:

    1. **Model Path**: The ``model_path`` parameter points to a local folder or
       huggingface repo that contains both *configuration files*
       and *pretrained weights*.

    2. **Default Configuration**: The model loads its default configuration from the
       *configuration files*.

    3. **Custom Configuration**: Users can provide a custom configuration via the
       ``config`` parameter during model initialization.

    4. **Configuration Override**: If custom configuration is provided,
       it overrides the default configuration.

    5. **Forecasting Horizon**: If the forecasting horizon (``fh``) specified during
       ``fit`` exceeds the default ``config.prediction_length``,
       the configuration is updated to reflect ``max(fh)``.

    6. **Model Architecture**: The final configuration is used to construct the
       *model architecture*.

    7. **Pretrained Weights**: *pretrained weights* are loaded from the ``model_path``,
       these weights are then aligned and loaded into the *model architecture*.

    8. **Weight Alignment**: However sometimes, *pretrained weights* do not align with
       the *model architecture*, because the config was changed which created a
       *model architecture* of different size than the default one.
       This causes some of the weights in *model architecture* to be reinitialized
       randomly instead of using the pre-trained weights.

    **Training Strategies**:

    - **Zero-shot Forecasting**: When all the *pre-trained weights* are correctly
      aligned with the *model architecture*, fine-tuing part is bypassed and
      the model preforms zero-short forecasting.

    - **Minimal Fine-tuning**: When not all the *pre-trained weights* are correctly
      aligned with the *model architecture*, rather some weights are re-initialized,
      these re-initialized weights are fine-tuned on the provided data.

    - **Full Fine-tuning**:  The model is *fully fine-tuned* on new data, updating *all
      parameters*. This approach offers maximum adaptation to the dataset but requires
      more computational resources.

    Parameters
    ----------
    model_path : str, default="ibm/TTM"
        Path to the Huggingface model to use for forecasting.
        This can be either:

        - The name of a Huggingface repository (e.g., "ibm/TTM")

        - A local path to a folder containing model files in a format supported
          by transformers. In this case, ensure that the directory contains all
          necessary files (e.g., configuration, tokenizer, and model weights).

        - If this parameter is *None*, training_mode should be *full* to allow
          full fine tuning of the model loaded from pretrained/provided config,
          else ValueError is raised.

    revision: str, default="main"
        Revision of the model to use:

        - "main": For loading model with context_length of 512
          and prediction_length of 96.

        - "1024_96_v1": For loading model with context_length of 1024
          and prediction_length of 96.

        This param becomes irrelevant when model_path is None

    validation_split : float, default=0.2
        Fraction of the data to use for validation

    config : dict, default={}
        Configuration to use for the model. See the ``transformers``
        documentation for details.

    training_args : dict, default={}
        Training arguments to use for the model. See ``transformers.TrainingArguments``
        for details.
        Note that the ``output_dir`` argument is required.

    compute_metrics : list, default=None
        List of metrics to compute during training. See ``transformers.Trainer``
        for details.

    callbacks : list, default=[]
        List of callbacks to use during training. See ``transformers.Trainer``

    broadcasting : bool, default=False
        if True, multiindex data input will be broadcasted to single series.
        For each single series, one copy of this forecaster will try to
        fit and predict on it. The broadcasting is happening inside automatically,
        from the outerside api perspective, the input and output are the same,
        only one multiindex output from ``predict``.

    use_source_package : bool, default=False
        If True, the model and configuration will be loaded directly from the source
        package ``tsfm_public.models.tinytimemixer``. This is useful if you
        want to bypass the local version of the package or when working in an
        environment where the latest updates from the source package are needed.
        If False, the model and configuration will be loaded from the local
        version of package maintained in sktime because of model's unavailability
        on pypi.
        To install the source package, follow the instructions here [4]_.

    training_mode : str, default="minimal"
        Strategy to use for fitting (fine-tuning) the model. This can be one of
        the following:
        - "zero-shot": Uses pre-trained model as it is. If model path is *None*
          with this strategy, ValueError is raised.
        - "minimal": Fine-tunes only a small subset of the model parameters,
          allowing for quick adaptation with limited computational resources.
          If model path is *None* with this strategy, ValueError is raised.
        - "full": Fine-tunes all model parameters, which may result in better
          performance but requires more computational power and time. Allows
          model path to be *None*.

    References
    ----------
    .. [1] https://github.com/ibm-granite/granite-tsfm/tree/main/tsfm_public/models/tinytimemixer
    .. [2] Ekambaram, V., Jati, A., Dayama, P., Mukherjee, S.,
           Nguyen, N.H., Gifford, W.M., Reddy, C. and Kalagnanam, J., 2024.
           Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced
           Zero/Few-Shot Forecasting of Multivariate Time Series. CoRR.
    .. [3] https://github.com/ibm-granite/granite-tsfm/blob/main/notebooks/tutorial/ttm_tutorial.ipynb
    .. [4] https://github.com/ibm-granite/granite-tsfm/tree/ttm

    Examples
    --------
    >>> from sktime.forecasting.ttm import TinyTimeMixerForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = TinyTimeMixerForecaster() # doctest: +SKIP
    >>> # performs zero-shot forecasting, as default config (unchanged) is used
    >>> forecaster.fit(y, fh=[1, 2, 3]) # doctest: +SKIP
    >>> y_pred = forecaster.predict() # doctest: +SKIP

    >>> from sktime.forecasting.ttm import TinyTimeMixerForecaster
    >>> from sktime.datasets import load_tecator
    >>>
    >>> # load multi-index dataset
    >>> y = load_tecator(
    ...     return_type="pd-multiindex",
    ...     return_X_y=False
    ... )
    >>> y.drop(['class_val'], axis=1, inplace=True)
    >>>
    >>> # global forecasting on multi-index dataset
    >>> forecaster = TinyTimeMixerForecaster(
    ...     model_path=None,
    ...     training_mode="full",
    ...     config={
    ...             "context_length": 8,
    ...             "prediction_length": 2
    ...     },
    ...     training_args={
    ...         "num_train_epochs": 1,
    ...         "output_dir": "test_output",
    ...         "per_device_train_batch_size": 32,
    ...     },
    ... ) # doctest: +SKIP
    >>>
    >>> # model initialized with random weights due to None model_path
    >>> # and trained with the full strategy.
    >>> forecaster.fit(y, fh=[1, 2, 3]) # doctest: +SKIP
    >>> y_pred = forecaster.predict() # doctest: +SKIP
    """
```
