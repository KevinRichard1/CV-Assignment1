# AI Usage
## AI Coding Tools Used
- Cursor
- ChatGPT

## Examples of AI Usage
- Exploring model architectures
    - Ex: Used AI to search various CNN architectures and summarize their strengths/weaknesses
- Searching for PyTorch API
    - Ex: Used AI to identify the syntax for implementing augmentations
- Writing boilerplate code
    - Ex: Used AI to understand how to modify starter code for various experiments

## Inaccurate Suggestions by AI
When writing the evaluation script, I encountered an error where the dataset had 16 classes, but the model had been trained on only 10 classes. The AI suggestion was to drop the 6 classes that did not exist in the training set. However, after inspecting the data, I realized that the original dataset download had missed 6 classes, and retraining successfully fixed the issue.

## Verifying AI Solutions
Before running AI solutions, I read through generated code to ensure it follows the respective prompt. In cases where the model deviated from the original plan or added unnecessary code, I modified the generated code.

## Experimental and Architectural Decisions
An example of a decision I made was using ConvNeXt as the model architecture. The initial AI recommendation was to use a ResNet or EfficientNet due to their smaller size. I opted to try ConvNeXt to evaluate how a modern CNN architecture performs with this dataset.