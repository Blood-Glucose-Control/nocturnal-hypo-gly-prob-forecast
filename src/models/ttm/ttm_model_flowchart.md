```mermaid
graph LR
    Start([User Code]) --> Init[__init__]
    Init --> InitModel[_initialize_model]

    Start --> Fit[fit - inherited from base]
    Fit --> TrainModel[_train_model]
    TrainModel --> PrepareData[_prepare_data]
    TrainModel --> CreateTrainingArgs[_create_training_arguments]
    TrainModel --> GetCallbacks[_get_callbacks]
    TrainModel --> ComputeMetrics[_compute_trainer_metrics]
    CreateTrainingArgs --> GetDistributedArgs[_get_distributed_training_args]
    PrepareData --> CreateColSpec[_create_column_specifiers]

    Start --> Predict[predict]
    Predict --> PrepareData2[_prepare_data]
    PrepareData2 --> CreateColSpec2[_create_column_specifiers]

    Start --> Evaluate[evaluate]
    Evaluate --> PrepareData3[_prepare_data]
    Evaluate --> ComputeMetrics2[_compute_trainer_metrics]
    PrepareData3 --> CreateColSpec3[_create_column_specifiers]

    Start --> PredictZeroShot[predict_zero_shot]
    PredictZeroShot --> Predict2[predict]
    Predict2 --> PrepareData4[_prepare_data]

    Start --> SaveModel[save_model - inherited from base]
    SaveModel --> SaveCheckpoint[_save_checkpoint]

    Start --> LoadModel[load_model - inherited from base]
    LoadModel --> LoadCheckpoint[_load_checkpoint]

    Start --> GetInfo[get_ttm_specific_info]
    GetInfo --> GetModelInfo[get_model_info - inherited]

    Start --> GetTrainingBackend[training_backend property]
    Start --> SupportsLora[supports_lora property]

    %% Base TSFM Public: Inherited public methods from BaseTimeSeriesFoundationModel
    style Fit fill:#0072B2,color:#000
    style SaveModel fill:#0072B2,color:#000
    style LoadModel fill:#0072B2,color:#000
    style GetModelInfo fill:#0072B2,color:#000

    %% Base TSFM Private: Inherited/base private methods (none shown directly)

    %% TTM Abstract Public: TTM-specific public methods and properties
    style Init fill:#F0E442,color:#000
    style Predict fill:#F0E442,color:#000
    style Predict2 fill:#F0E442,color:#000
    style Evaluate fill:#F0E442,color:#000
    style PredictZeroShot fill:#F0E442,color:#000
    style GetInfo fill:#F0E442,color:#000
    style GetTrainingBackend fill:#F0E442,color:#000
    style SupportsLora fill:#F0E442,color:#000

    %% TTM Abstract Private: TTM-specific private implementation methods
    style InitModel fill:#CC79A7,color:#000
    style TrainModel fill:#CC79A7,color:#000
    style PrepareData fill:#CC79A7,color:#000
    style PrepareData2 fill:#CC79A7,color:#000
    style PrepareData3 fill:#CC79A7,color:#000
    style PrepareData4 fill:#CC79A7,color:#000
    style SaveCheckpoint fill:#CC79A7,color:#000
    style LoadCheckpoint fill:#CC79A7,color:#000
    style CreateColSpec fill:#CC79A7,color:#000
    style CreateColSpec2 fill:#CC79A7,color:#000
    style CreateColSpec3 fill:#CC79A7,color:#000
    style CreateTrainingArgs fill:#CC79A7,color:#000
    style GetCallbacks fill:#CC79A7,color:#000
    style ComputeMetrics fill:#CC79A7,color:#000
    style ComputeMetrics2 fill:#CC79A7,color:#000
    style GetDistributedArgs fill:#CC79A7,color:#000
    ```
