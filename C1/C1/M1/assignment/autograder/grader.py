from types import ModuleType, FunctionType
from typing import Dict, List, Optional
from dlai_grader.grading import test_case, object_to_grade
from dlai_grader.types import grading_function, grading_wrapper, learner_submission

##### Imports for tests

import torch
import torch.nn as nn
import torch.optim as optim
import math
import grader_utils

############


def part_1(
    learner_mod: learner_submission, solution_mod: Optional[ModuleType] = None
) -> grading_function:
    @object_to_grade(learner_mod, "rush_hour_feature")
    def g(learner_func: FunctionType) -> List[test_case]:
        
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "rush_hour_feature has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        
        
        sample_tensor = torch.tensor([
            [1.60,      8.20,        0,          7.22],
            [13.09,     16.80,       1,          32.41],
            [6.97,      8.02,        1,          17.47],
        ], dtype=torch.float32)
        
        sample_hours = sample_tensor[:, 1]
        sample_weekends = sample_tensor[:, 2]
        
        learner_sample = learner_func(sample_hours, sample_weekends)
        
        # ##### Return type check #####
        t = test_case()
        if not isinstance(learner_sample, torch.Tensor):
            t.failed = True
            t.msg = "Incorrect is_rush_hour_mask type returned from rush_hour_feature"
            t.want = torch.Tensor
            t.got = type(learner_sample)
            return [t]
        
        # ##### Shape check #####
        t = test_case()
        if learner_sample.shape != (3, ):
            t.failed = True
            t.msg = "is_rush_hour_mask returned from rush_hour_feature has wrong shape. Follow the exercise instructions to make sure you are correctly implementing all of the conditions and operations"
            t.want = "torch.Size([3])"
            t.got = learner_sample.shape
            return [t]
        
        sample_tensor = torch.tensor([
            [1.60,      8.20,        0,          7.22],
            [10.66,     16.07,       0,          37.17],
            [18.24,     13.47,       0,          38.36] 
        ], dtype=torch.float32)
        
        sample_hours = sample_tensor[:, 1]
        sample_weekends = sample_tensor[:, 2]
        
        learner_sample = learner_func(sample_hours, sample_weekends)
        
        
        expected = torch.tensor([1., 1., 0.])
        
        # ##### Expected Values Check #####
        t = test_case()
        if not torch.equal(learner_sample, expected):
            t.failed = True
            t.msg = "rush_hour_feature returned incorrect values. Follow the exercise instructions to make sure you are correctly implementing all of the conditions and operations"
            t.want = expected
            t.got = learner_sample
        cases.append(t)

        return cases

    return g



def part_2(
    learner_mod: learner_submission, solution_mod: Optional[ModuleType] = None
) -> grading_function:
    @object_to_grade(learner_mod, "prepare_data")
    def g(learner_func: FunctionType) -> List[test_case]:
        
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "prepare_data has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]
        
        
        file_path = './data/data_with_features.csv'
        subset_df = grader_utils.load_rows(file_path)
        
        learner_features, learner_targets, learner_results = learner_func(subset_df)
        
        # ##### Return type check (prepared_features) #####
        t = test_case()
        if not isinstance(learner_features, torch.Tensor):
            t.failed = True
            t.msg = "Incorrect prepared_features type returned from prepare_data"
            t.want = torch.Tensor
            t.got = type(learner_features)
            return [t]
        
        # ##### Return type check (prepared_targets) #####
        t = test_case()
        if not isinstance(learner_targets, torch.Tensor):
            t.failed = True
            t.msg = "Incorrect prepared_targets type returned from prepare_data"
            t.want = torch.Tensor
            t.got = type(learner_targets)
            return [t]
        
        # ##### DType check (full_tensor) #####
        t = test_case()
        if learner_results["full_tensor"].dtype != torch.float32:
            t.failed = True
            t.msg = "Incorrect dtype for full_tensor"
            t.want = torch.float32
            t.got = learner_results["full_tensor"].dtype
            return [t]
        
        expected_raw_distances = torch.tensor([18.2600,  5.5500,  3.7500, 10.3000, 19.7300,  5.6000, 13.7700, 15.4700, 5.5200, 14.8400])
        expected_raw_hours = torch.tensor([13.3400, 14.2100, 14.8600, 12.0300, 15.4800, 19.1300,  8.0000, 19.0400, 8.0000, 19.2200])
        expected_raw_weekends = torch.tensor([1., 1., 1., 1., 0., 1., 0., 0., 0., 0.])
        expected_raw_targets = torch.tensor([39.2000, 10.6100, 12.6300, 22.9800, 39.2200, 18.2900, 52.7100, 41.5200, 28.7400, 32.0300])
        
        expected_values = [expected_raw_distances, expected_raw_hours, expected_raw_weekends, expected_raw_targets]
        
        # ##### Returned column (slicing) values checks #####
        keys_to_check = ['raw_distances', 'raw_hours', 'raw_weekends', 'raw_targets']
        for key, expected_tensor in zip(keys_to_check, expected_values):
            learner_tensor = learner_results[key]
            t = test_case()
            if not torch.equal(learner_tensor, expected_tensor):
                t.failed = True
                t.msg = f"The tensor for '{key}' is incorrect. Make sure you are correctly slicing to separate out {key} column"
                t.want = expected_tensor
                t.got = learner_tensor
                cases.append(t)
                
        # ##### Return cases, if any, before moving on #####    
        if cases:
            return cases
        
        # ##### Checking if "unsqueeze(1)" has been applied to *_col variables #####
        keys_to_check = ['distances_col', 'hours_col', 'weekends_col', 'rush_hour_col']
        for key in keys_to_check:            
            t = test_case()
            if learner_results[key].shape != torch.Size([10, 1]):
                t.failed = True
                t.msg = f"Incorrect shape for '{key}'. Make sure you are applying unsqueeze(1) to it"
                t.want = torch.Size([10, 1])
                t.got = learner_results[key].shape
            cases.append(t)

        return cases

    return g



def part_3(
    learner_mod: learner_submission, solution_mod: Optional[ModuleType] = None
) -> grading_function:
    @object_to_grade(learner_mod, "init_model")
    def g(learner_func: FunctionType) -> List[test_case]:
        
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "init_model has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        model, optimizer, loss_function = learner_func()
        
        
        # ##### Return Type Check (Model) #####
        t = test_case()
        if not isinstance(model, nn.Sequential):
            t.failed = True
            t.msg = "Incorrect model type returned from init_model"
            t.want = nn.Sequential
            t.got = type(model)
            return [t]
        
        # ##### Return Type Check (optimizer) #####
        t = test_case()
        if not isinstance(optimizer, optim.SGD):
            t.failed = True
            t.msg = "Incorrect optimizer type returned from init_model"
            t.want = optim.SGD
            t.got = type(optimizer)
            return [t]
        
        # ##### Return Type Check (loss_function) #####
        t = test_case()
        if not isinstance(loss_function, nn.MSELoss):
            t.failed = True
            t.msg = "Incorrect loss_function type returned from init_model"
            t.want = nn.MSELoss
            t.got = type(loss_function)
            return [t]
        
        # ##### Total Number of model's layers Check #####
        t = test_case()
        if len(model) != 5:
            t.failed = True
            t.msg = "model has an incorrect number of layers"
            t.want = 5
            t.got = len(model)
            return [t]
        
        
        # ##### Check if model's layers are as expected #####
        layers_list = [nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear]
        
        for layer_num, layer in enumerate(model):
            t = test_case()
            if not isinstance(layer, layers_list[layer_num]):
                t.failed = True
                t.msg = f"model's ({layer_num}) layer is incorrect"
                t.want = layers_list[layer_num]
                t.got = layer
                cases.append(t)
            
        # ##### Return cases, if any, before moving on #####    
        if cases:
            return cases
        
        
        # ##### Check model's Linear layers are of expected dimension #####
        layer_dims = [[4, 64], 0, [64, 32], 0, [32, 1]]

        for layer_num, layer in enumerate(model):
            # Check if the remainder when dividing by 2 is 0
            if layer_num % 2 == 0:
                t = test_case()
                if layer.in_features != layer_dims[layer_num][0] or layer.out_features != layer_dims[layer_num][1]:
                    t.failed = True
                    t.msg = f"({layer_num}): Linear layer has incorrect dimensions"
                    t.want = f"Linear({layer_dims[layer_num][0]}, {layer_dims[layer_num][1]})"
                    t.got = f"Linear({layer.in_features}, {layer.out_features})"
                cases.append(t)
        
        
        # ##### Learning Rate Value Check #####
        lr = optimizer.defaults["lr"]
        t = test_case()
        if lr != 0.01:
            t.failed = True
            t.msg = "incorrect learning rate set in optimizer"
            t.want = 0.01
            t.got = lr
        cases.append(t)

        return cases

    return g



def part_4(
    learner_mod: learner_submission, solution_mod: Optional[ModuleType] = None
) -> grading_function:
    @object_to_grade(learner_mod, "train_model")
    def g(learner_func: FunctionType) -> List[test_case]:
        
        cases: List[test_case] = []

        t = test_case()
        if not isinstance(learner_func, FunctionType):
            t.failed = True
            t.msg = "train_model has incorrect type"
            t.want = FunctionType
            t.got = type(learner_func)
            return [t]

        try:
            learner_trained_model, learner_loss = learner_func(grader_utils.features, grader_utils.targets, 10000, verbose=False)
            
            # ##### Check return type of model #####
            t = test_case()
            if not isinstance(learner_trained_model, nn.Sequential):
                t.failed = True
                t.msg = "train_model did not return a Sequential model"
                t.want = nn.Sequential
                t.got = type(learner_trained_model)
                return [t]
            
            # ##### Check if loss is changing #####
            t = test_case()
            if learner_loss[0] == learner_loss[1]:
                t.failed = True
                t.msg = "Loss did not change during taining. Make sure you are calculating the loss correctly"
                t.want = "Different loss values"
                t.got = f"Loss at 5000th epoch: {learner_loss[0]}, Loss at 10000th epoch: {learner_loss[1]}"
                return [t]
            
        except Exception as e:
            t = test_case()
            t.failed = True
            t.msg = f"train_model raised an exception"
            t.want = f"The model to train."
            t.got = f"Training ran into an error: \"{e}\""
            return [t]
        

        # ##### Define an input for the model #####
        inputs = torch.tensor([[-0.0624, -0.2469,  0.0000,  1.0000]])
        with torch.no_grad():
            outputs = learner_trained_model(inputs)
            
        
        # ##### Check model output shape #####
        t = test_case()
        if outputs.shape[0] != 1:
            t.failed = True
            t.msg = "model output has incorrect shape"
            t.want = 1
            t.got = outputs.shape[0]
            return [t]
        
        
        expected_output = 38.212249755859375
        
        # ##### Check expected value
        prediction = outputs[0].item()
        t = test_case()
        close = math.isclose(prediction, expected_output, abs_tol=0.4)
        if not close:
            t.failed = True
            t.msg = f"model's output is not close enough to expected output. Make sure your loss is decreasing as expected"
            t.want = f"{expected_output} +- 0.4"
            t.got = prediction
        cases.append(t)

        return cases

    return g



def handle_part_id(part_id: str) -> grading_wrapper:
    grader_dict: Dict[str, grading_wrapper] = {
        "Afj9W": part_1,
        "oNcJE": part_2,
        "lSTlq": part_3,
        "rYq1X": part_4,
    }
    return grader_dict[part_id]
