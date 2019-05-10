# reproducing poison min by using sigopt

## same noise seed and noise level (ecerything identical to before, just making it reproduces)
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

## different sigopt objective function
the objective is now quadratically penalizing anything less than perfect train acc, and lineraly penalizing the test acc from being high
RESULT: doesnt work as well

sigopt: https://app.sigopt.com/experiment/63037

comet: https://www.comet.ml/wronnyhuang/swissroll-5-sigopt

## rerun best hyperparam more epochs
comet experiment series link: https://www.comet.ml/wronnyhuang/swissroll-3-sigopt
rerun it for many more epoch (and many tries) get an even lower test acc

best result 100% train 7% test:
https://www.comet.ml/wronnyhuang/swissroll/54bcab88d3964dfeb359197400b3c619

## fix architecture and get good clean performance

sigopt link: https://app.sigopt.com/experiment/63024

comet sigopt series: https://www.comet.ml/wronnyhuang/swissroll-10-sigopt

comet best: https://www.comet.ml/wronnyhuang/swissroll/6970be46b0254730bc85b9901252badd/images


