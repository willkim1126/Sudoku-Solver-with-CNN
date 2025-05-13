# Disclaimer: This page only discuss AWS implementation of my CNN model. For initial model and its report, please visit Initial Model folder.

**Author:** Junghwan Kim  
**Contact:** junghwk11@gmail.com

## Introduction

After developing an enhanced version of my Sudoku Solver, I recognized several opportunities for further improvement, particularly in terms of scalability and generalization. To address these, I migrated the project to Amazon Web Services (AWS), leveraging the increased memory and computing power of SageMaker to build a more robust convolutional neural network (CNN) for solving Sudoku puzzles.

## Data Collection and Preprocessing

Using AWS SageMaker enabled me to increase my training sample size from 200,000 to 1 million puzzles. To ensure true randomness and eliminate potential correlations within the dataset, I employed reservoir sampling to select 1 million random samples. This expanded dataset was stored in an S3 bucket, which streamlined data management and integration with SageMaker.

Unlike previous iterations where I used Scikit-Learn pipelines for preprocessing and model training, this version relied on classic Python functions and objects. This change introduced some subtle differences in the preprocessing steps and the training workflow. Each puzzle was first converted from a string representation to a 9×9 matrix. I then applied MinMax scaling to normalize the puzzle values, which is essential for faster convergence and more stable parameter updates during training. The solutions were one-hot encoded to ensure compatibility with the loss function. As a new step, I converted both the scaled puzzles and one-hot encoded solutions into PyTorch tensors, which allowed for more efficient batch processing. To manage memory efficiently, all preprocessing was performed in chunks of 20,000 samples.

## Model Architecture

The architecture of the CNN remained largely unchanged from earlier versions. The model consists of 16 layers, each utilizing 3×3 convolutions with 512 filters, followed by batch normalization and ReLU activations. The input to the model is a single-channel 9×9 grid representing the unsolved Sudoku puzzle. The initial convolutional layer expands the input to 512 feature maps, followed by 14 repeated blocks of Conv2D, batch normalization, and ReLU. The final layer is a 1×1 convolution that outputs 9 channels, corresponding to the possible digits for each cell.

## Training Procedure

The model was trained over five epochs with a batch size of 64, a learning rate of 0.001, and the Adam optimizer. CrossEntropyLoss was used as the loss function. The training process included processing data in manageable chunks, performing forward passes through the network, calculating loss, backpropagating gradients, updating parameters, and evaluating accuracy at the end of each epoch. Model checkpoints were saved after each epoch, and validation was performed on a held-out test set.

## Training Results

Throughout training, the model demonstrated steady improvement. By the final epoch, training accuracy reached nearly 90%, with a validation accuracy of 86.85%. The relatively small gap between training and validation accuracy suggests that the model generalized well and exhibited minimal overfitting.

| Epoch | Training Accuracy | Validation Accuracy |
|-------|------------------|--------------------|
| 1     | 73.48%           | 80.76%             |
| 2     | 83.94%           | 83.49%             |
| 3     | 86.27%           | 84.98%             |
| 4     | 88.14%           | 86.13%             |
| 5     | 89.26%           | 86.85%             |

## Evaluation on Test Sets

To assess the model’s real-world performance, I conducted two separate test evaluations:

**Test 1:**  
I sampled 1,000 Sudoku puzzles from the original dataset and saved them in S3 as 'sudoku_test_data.csv'. The model achieved a test accuracy of 99.10% on this set. While this result was impressive, I suspected that the test set might not be fully independent from the training data, which could inflate the accuracy.

**Test 2 (Out-of-Distribution):**  
To ensure a more rigorous evaluation, I sourced a completely new dataset from Kaggle containing 1 million Sudoku puzzles. From this, I sampled 2,000 puzzles and saved them as 'new_sudoku_test.csv' in S3. When evaluated on this out-of-distribution test set, the model achieved a test accuracy of 91.31%. This result not only exceeded the validation accuracy observed during training but also demonstrated the model’s strong ability to generalize to entirely new data distributions.

## Discussion

The high test accuracy on a new, independent dataset indicates that the model has learned fundamental Sudoku-solving strategies rather than merely memorizing patterns from the training data. The deep 16-layer CNN architecture, with consistent use of 512 filters and 3×3 convolutions, effectively captures the spatial relationships and constraints inherent to Sudoku puzzles. Batch normalization between convolutional layers contributed to more stable training and improved generalization. Training on a large and diverse dataset further enhanced the model’s robustness.

## Conclusion

The AWS-based implementation of my convolutional neural network Sudoku Solver was a significant success. Compared to earlier versions, which achieved test accuracies of 73% and 79%, the AWS implementation improved performance by over 12 percentage points. The combination of a scalable cloud environment, an optimized data pipeline, and a deep CNN architecture enabled the model to generalize well, achieving over 91% accuracy on a completely independent test set. This project demonstrates the effectiveness of deep learning and cloud computing for solving complex, rule-based problems like Sudoku.
