# Requirements

## main()

### step4 call:
- Update TTMConfig to be generic
- Clean/clarify the skip_training path in step4 call

## Step Functions

### Step 4
- convert create_ttm_zero_shot_config()
- TTMForecaster inside step4, needs to be made generic

### Step 5
TTMConfig needs to be updated to be generic.
- Some logger.info() print outs might not work for all configs. (Is freeze_backbone())
- TTMForecaster needs to be generic
- model.fit should be implemented by child class, no change needed in worflow script
- model.save ... implemented by child ...

### Step 6
TTMConfig needs to be generic
Change TTMForcaster to be child specific
    - load() functionality does not need to changed

### Step 7
No change needed for .training_history, comes from base class
- model.fit is implemented from child class no change necessary. Child class must conform to the shape defined by TTM example.
- mode.save is implemented by the base class, no change needed.
