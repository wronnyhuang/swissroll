# reproducing poison min by using sigopt

## same noise seed and noise level (ecerything identical)
2019.5.6

Reran the sigopt hyperparameter search. using the same code (checkouted the old commit) as before.
Success search of the hyperparameters led to generalization gap of 82% which is about the same as before.
https://app.sigopt.com/experiment/82011

The comet experiments are all here
https://www.comet.ml/wronnyhuang/swissroll-2-sigopt/

## different noise seed and noise level

it also worked. sigopt does an amazing job.
sigopt link: https://app.sigopt.com/experiment/81936

comet experiment series link: https://www.comet.ml/wronnyhuang/swissroll-3-sigopt

comet of best result (99.5% train 15% test)
https://www.comet.ml/wronnyhuang/swissroll-3-sigopt/dc9f249ba8ab4fe087a5078cfa732a5e
