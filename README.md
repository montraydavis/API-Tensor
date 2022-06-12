# API-Tensor
Using Tensorflow2 Machine Learning to automate API execution. API Tensors allow you to fully automate the API execution process.

This project aims at allowing developers to automate the API Automation process. Instead of writing, wiring, and maintaining API inputs, use Tensorflow and OpenAI Gym to 'learn' which data correlates to an API endpoints' input(s) -- powered by Artificial Intelligence.

In short, using OpenAI Gym, API-Tensor will repeatedly execute an endpoint using random values from a supplied list of possible values.

# Example

In this scenario, we will attempt to authenticate with our API with only knowing which table(s) in the database that required inputs come from.

In this case, we know that data comes from the 'users' and 'user_types' table(s).

Users

| FIRST_NAME  | LAST_NAME   | USER_NAME   | AUTH_PASSWORD |
| ----------- | ----------- | ----------- | -----------   |
| ...         | ...         | ...         | ...           |

User_Types
| USER_TYPE  | USER_TYPE_ID |
| ----------- | ----------- |
| ...         | ...         |

Our endpoint requires two parameters (which do not match our table column names):

1. Username
2. Password

[View the notebook for the continued example.](https://github.com/montraydavis/API-Tensor/blob/main/api-tensor/src/notebook.ipynb)

v.0.0.1:

Initial Release

- Simple Endpoint Execution (HTTP [POST] (application/json) only)